"""Export ALIKED (aliked-n16) to ONNX for the C++ odometry.

ALIKED: Zhao et al., "ALIKED: A Lighter Keypoint and Descriptor Extraction
Network via Deformable Transformation", IEEE TIM 2023. Weights from
https://github.com/Shiaoming/ALIKED; the PyTorch model is the pure-PyTorch
implementation in LightGlue (lightglue/aliked.py). Both pinned in
Dockerfile.export.

Two ONNX models; the keypoint detection in between (DKD: NMS, top-k, sub-pixel
soft-argmax, scores) is done in C++ (src/deep_features.h), as in aliked.py:

  aliked-n16_dense.onnx (Hp, Wp dynamic, multiples of 32; C++ pads the image by
  edge replication like InputPadder, and crops the score map):
    input  "image":       1 x 1 x Hp x Wp, float32, grayscale, values 0-1
    input  "crop":        4, int64: top, left, H, W of the image in the padded image
    output "feature_map": 1 x 128 x Hp x Wp, L2-normalized, zero outside the crop
    output "score_map":   1 x 1 x Hp x Wp, keypoint scores 0-1

  aliked-n16_desc.onnx (SDDH descriptor head; N and the map size dynamic):
    input  "feature_map": 1 x 128 x Hp x Wp, the dense model's output
    input  "crop":        4, int64, as above
    input  "keypoints":   N x 2, float32, pixel coordinates (x, y) in the image
    output "descriptors": N x 128, L2-normalized

  The feature map goes straight from one model to the other (it stays on the
  GPU) and is never cropped (at 960 x 608 it is 300 MB; a crop is one more
  copy). SDDH only reads 3 x 3 patches and bilinear samples of it: they are
  taken at the shifted positions in the padded map, which is zero outside the
  crop, exactly like grid_sample's zero padding of the cropped map.

Changes for the export, numerically checked against the original below:
  - Deformable convolutions (torchvision.ops.deform_conv2d, blocks 3 and 4)
    are written as 9 bilinear grid_samples + one convolution: ONNX Runtime has
    no DeformConv kernel.
  - Sizes come from tensor shapes (not Python ints), so H, W and N stay dynamic.

Usage (inside the export container, see export_models.sh):
  python export_aliked.py [--reference IMAGE --top-k N]
With --reference, the original ALIKED output (top-k mode) for IMAGE is written
to aliked-n16.reference.txt (x y score descriptor... per line).
"""

import argparse
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, "/opt/LightGlue")
import lightglue.aliked as aliked_module  # noqa: E402

WEIGHTS = "/opt/aliked-n16.pth"
MODEL = "aliked-n16"


def load_aliked(top_k: int):
    # ALIKED.__init__ downloads the weights: use the pinned local copy.
    torch.hub.load_state_dict_from_url = lambda *a, **k: torch.load(WEIGHTS, map_location="cpu")
    # detection_threshold <= 0 -> top-k mode (DKD top_k = max_num_keypoints)
    return aliked_module.ALIKED(model_name=MODEL, max_num_keypoints=top_k,
                                detection_threshold=-1).eval()


def shape_hw(x):
    s = torch._shape_as_tensor(x)  # traced as an ONNX Shape: stays dynamic
    return s[2], s[3]


def pixel_grid(x):
    """Column and row index of every pixel of x (1 x 1 x H x W each), dynamic."""
    ones = torch.ones_like(x[:, :1])
    return ones.cumsum(3) - 1, ones.cumsum(2) - 1


class DeformConvGS(nn.Module):
    """DeformableConv2d (3x3, stride 1, padding 1, no mask) without DeformConv.

    torchvision.ops.deform_conv2d samples, for kernel tap k = 3*i + j, the
    input at (y - 1 + i + offset[2k], x - 1 + j + offset[2k+1]) with bilinear
    interpolation (zero outside the image) and applies the weights. Here: one
    grid_sample (align_corners=True, zeros padding: the same interpolation) per
    tap, then the weights as a 1x1 convolution over the 9*C sampled channels.
    """

    def __init__(self, dcn: aliked_module.DeformableConv2d):
        super().__init__()
        assert not dcn.mask and dcn.regular_conv.kernel_size == (3, 3)
        assert tuple(dcn.padding) == (1, 1) and dcn.regular_conv.stride == (1, 1)
        self.offset_conv = dcn.offset_conv
        w = dcn.regular_conv.weight  # O x C x 3 x 3
        self.weight = nn.Parameter(w.reshape(w.shape[0], -1, 1, 1).detach())  # O x 9C (c-major)
        self.bias = dcn.regular_conv.bias

    def forward(self, x):
        h, w = shape_hw(x)
        max_offset = torch.maximum(h, w).to(x.dtype) / 4.0
        offset = self.offset_conv(x)
        offset = torch.minimum(torch.maximum(offset, -max_offset), max_offset)
        gx, gy = pixel_grid(x)
        sx = 2.0 / (w.to(x.dtype) - 1)
        sy = 2.0 / (h.to(x.dtype) - 1)
        taps = []
        for k in range(9):
            i, j = divmod(k, 3)
            py = gy + (i - 1) + offset[:, 2 * k:2 * k + 1]
            px = gx + (j - 1) + offset[:, 2 * k + 1:2 * k + 2]
            grid = torch.cat([px * sx - 1, py * sy - 1], dim=1).permute(0, 2, 3, 1)
            taps.append(F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                                      align_corners=True))
        cols = torch.stack(taps, dim=2).flatten(1, 2)  # B x (C*9) x H x W, index c*9 + k
        return F.conv2d(cols, self.weight, self.bias)


def replace_dcn(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, aliked_module.DeformableConv2d):
            setattr(module, name, DeformConvGS(child))
        else:
            replace_dcn(child)


class AlikedDense(nn.Module):
    """ALIKED.extract_dense_map without the padding (done in C++).

    The feature map is not unpadded but set to zero outside `crop`, see the
    module docstring.
    """

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image, crop):
        m = self.m
        image = image.repeat(1, 3, 1, 1)  # grayscale -> RGB, as kornia grayscale_to_rgb
        x1 = m.block1(image)
        x2 = m.block2(m.pool2(x1))
        x3 = m.block3(m.pool4(x2))
        x4 = m.block4(m.pool4(x3))
        parts = [m.gate(m.conv1(x1)), m.upsample2(m.gate(m.conv2(x2))),
                 m.upsample8(m.gate(m.conv3(x3))), m.upsample32(m.gate(m.conv4(x4)))]
        x1234 = torch.cat(parts, dim=1)
        score_map = torch.sigmoid(m.score_head(x1234))
        # F.normalize(x1234, p=2, dim=1), written out to save GPU memory: as
        # ReduceL2 it asks ONNX Runtime's CUDA provider for ~4x the map size of
        # workspace, and x1234 * x1234 would be one more full-size map. The sum
        # of squares is taken per 32-channel part instead.
        sum_sq = sum((p * p).sum(dim=1, keepdim=True) for p in parts)
        # 1 / norm inside the crop, 0 in the padding.
        gx, gy = pixel_grid(score_map)
        inside = ((gy >= crop[0]) & (gy < crop[0] + crop[2]) &
                  (gx >= crop[1]) & (gx < crop[1] + crop[3])).to(x1234.dtype)
        scale = inside / torch.sqrt(sum_sq).clamp(min=1e-12)
        return x1234 * scale, score_map


class AlikedDesc(nn.Module):
    """SDDH.forward for one image, keypoints in pixels, sizes from tensors."""

    def __init__(self, sddh: aliked_module.SDDH):
        super().__init__()
        assert not sddh.mask and not sddh.conv2D and sddh.kernel_size == 3
        self.s = sddh

    def forward(self, feature_map, crop, keypoints):
        s = self.s
        c = feature_map.shape[1]  # constant (128)
        # feature_map: padded, zero outside crop = [top, left, h, w]. Sizes and
        # positions below are those of the cropped map (as in the original);
        # the map is read at (position + [left, top]).
        top, left, h, w = crop[0], crop[1], crop[2], crop[3]
        hp, wp = shape_hw(feature_map)
        dtype = feature_map.dtype
        max_offset = torch.maximum(h, w).to(dtype) / 4.0
        # get_patches(): 3x3 patch whose corner is (kpt - 1.5 + 1).long(), clamped
        kl = keypoints.long()
        corner = (kl.to(dtype) - 0.5).long()
        cx = torch.minimum(torch.clamp(corner[:, 0], min=0), w - 4)
        cy = torch.minimum(torch.clamp(corner[:, 1], min=0), h - 4)
        r = torch.arange(3, device=keypoints.device)
        index = ((top + cy)[:, None, None] + r[None, :, None]) * wp + \
            ((left + cx)[:, None, None] + r[None, None, :])
        flat = feature_map.reshape(c, -1)  # batch 1; not feature_map[0]: a Gather = full copy
        patch = flat[:, index.reshape(-1)].reshape(c, -1, 3, 3).permute(1, 0, 2, 3)  # N x C x 3 x 3
        offset = s.offset_conv(patch)
        offset = torch.minimum(torch.maximum(offset, -max_offset), max_offset)
        offset = offset[:, :, 0, 0].reshape(-1, 2, s.n_pos).permute(0, 2, 1)  # N x M x 2
        pos = keypoints[:, None, :] + offset  # cropped-map pixels
        # grid_sample(align_corners=True) on the padded map at the same pixels:
        # neighbours outside the crop are 0 there, as outside the cropped map.
        shift = torch.stack([left, top]).to(dtype)
        whp = torch.stack([wp - 1, hp - 1]).to(dtype)
        pos = 2.0 * (pos + shift) / whp - 1
        features = F.grid_sample(feature_map, pos.reshape(1, -1, 1, 2), mode="bilinear",
                                 align_corners=True)  # 1 x C x (N*M) x 1
        features = features.reshape(c, -1, s.n_pos, 1).permute(1, 0, 2, 3)  # N x C x M x 1
        features = torch.selu(s.sf_conv(features)).squeeze(-1)  # N x C x M
        descs = torch.einsum("ncp,pcd->nd", features, s.agg_weights)
        return F.normalize(descs, p=2.0, dim=1)


def pad32(image):
    """InputPadder(h, w, 32).pad, as numbers (the C++ side does the same)."""
    h, w = image.shape[-2:]
    ph = ((h // 32 + 1) * 32 - h) % 32
    pw = ((w // 32 + 1) * 32 - w) % 32
    pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
    return F.pad(image, pad, mode="replicate"), pad


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", help="image for the C++ reference output")
    parser.add_argument("--top-k", type=int, default=2000)
    args = parser.parse_args()

    original = load_aliked(args.top_k)
    model = load_aliked(args.top_k)
    replace_dcn(model)
    dense = AlikedDense(model).eval()
    desc = AlikedDesc(model.desc_head).eval()

    dense_path, desc_path = f"{MODEL}_dense.onnx", f"{MODEL}_desc.onnx"
    with torch.no_grad():
        torch.onnx.export(dense, (torch.rand(1, 1, 480, 640), torch.tensor([0, 4, 480, 632])),
                          dense_path, opset_version=17,
                          input_names=["image", "crop"], output_names=["feature_map", "score_map"],
                          dynamic_axes={"image": {2: "Hp", 3: "Wp"},
                                        "feature_map": {2: "Hp", 3: "Wp"},
                                        "score_map": {2: "Hp", 3: "Wp"}})
        fm = torch.rand(1, 128, 128, 160)
        fm = F.normalize(fm, dim=1)
        crop = torch.tensor([4, 0, 120, 160])
        kp = torch.rand(50, 2) * torch.tensor([159.0, 119.0])
        torch.onnx.export(desc, (fm, crop, kp), desc_path, opset_version=17,
                          input_names=["feature_map", "crop", "keypoints"],
                          output_names=["descriptors"],
                          dynamic_axes={"feature_map": {2: "Hp", 3: "Wp"}, "keypoints": {0: "N"},
                                        "descriptors": {0: "N"}})
    print(f"Wrote {dense_path}, {desc_path}")

    s_dense = ort.InferenceSession(dense_path, providers=["CPUExecutionProvider"])
    s_desc = ort.InferenceSession(desc_path, providers=["CPUExecutionProvider"])
    for h, w in [(480, 752), (360, 500)]:
        rgb = torch.rand(1, 1, h, w).repeat(1, 3, 1, 1)
        with torch.no_grad():
            fm_ref, sm_ref = original.extract_dense_map(rgb)  # pads and unpads itself
            kpts_n, _, _ = original.dkd(sm_ref)
            desc_ref = original.desc_head(fm_ref, kpts_n)[0][0]
        padded, pad = pad32(rgb[:, :1])
        crop = np.array([pad[2], pad[0], h, w], dtype=np.int64)
        fm_padded, sm = s_dense.run(None, {"image": padded.numpy(), "crop": crop})
        outside = np.ones(fm_padded.shape[2:], bool)
        outside[pad[2]:pad[2] + h, pad[0]:pad[0] + w] = False
        assert not np.any(fm_padded[:, :, outside]), "feature map not zero outside the crop"
        fm = fm_padded[:, :, pad[2]:pad[2] + h, pad[0]:pad[0] + w]
        sm = sm[:, :, pad[2]:pad[2] + h, pad[0]:pad[0] + w]
        kpts_px = ((kpts_n[0] + 1) / 2 * torch.tensor([w - 1, h - 1])).numpy()
        d = s_desc.run(None, {"feature_map": fm_padded, "crop": crop, "keypoints": kpts_px})[0]
        err_fm = float(np.abs(fm - fm_ref.numpy()).max())
        err_sm = float(np.abs(sm - sm_ref.numpy()).max())
        err_d = float(np.abs(d - desc_ref.numpy()).max())
        print(f"{h}x{w}: max |onnx - original| feature_map {err_fm:.2e}, score_map {err_sm:.2e}, "
              f"descriptors {err_d:.2e} ({len(kpts_px)} keypoints)")
        assert max(err_fm, err_sm, err_d) < 1e-3

    if args.reference:
        gray = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(gray).float()[None, None] / 255.0
        with torch.no_grad():
            result = original({"image": image})
        path = f"{MODEL}.reference.txt"
        with open(path, "w") as f:
            for k, s, d in zip(result["keypoints"][0].numpy(), result["keypoint_scores"][0].numpy(),
                               result["descriptors"][0].numpy()):
                f.write(" ".join(f"{v:.6f}" for v in [*k, s, *d]) + "\n")
        print(f"Reference: {len(result['keypoints'][0])} keypoints -> {path}")


if __name__ == "__main__":
    main()
