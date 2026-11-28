"""Export XFeat (sparse features) to ONNX for the C++ odometry.

XFeat: Potje et al., "XFeat: Accelerated Features for Lightweight Image
Matching", CVPR 2024. Code and weights: https://github.com/verlab/accelerated_features
(pinned in Dockerfile.export, cloned to /opt/xfeat).

The ONNX model holds the network only. The sparse part of
XFeat.detectAndCompute (NMS, scores, top-k, descriptor sampling) is done in C++
(src/deep_features.h), exactly as in modules/xfeat.py.

ONNX model (H, W dynamic, multiples of 32; C++ resizes the image like
XFeat.preprocess_tensor):
  input  "image":       1 x 1 x H x W, float32, grayscale, values 0-255
  output "feats":       1 x 64 x H/8 x W/8, L2-normalized dense descriptors
  output "heatmap":     1 x 1 x H x W, keypoint heatmap (softmax, 8x8 unshuffled)
  output "reliability": 1 x 1 x H/8 x W/8, reliability map

Usage (inside the export container, see export_models.sh):
  python export_xfeat.py --out xfeat.onnx [--reference IMAGE --top-k N]
With --reference, the official XFeat.detectAndCompute output for IMAGE is
written to <out>.reference.txt (x y score descriptor... per line), to test the
C++ implementation.
"""

import argparse
import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

sys.path.insert(0, "/opt/xfeat")
from modules.model import XFeatModel  # noqa: E402
from modules.xfeat import XFeat  # noqa: E402

WEIGHTS = "/opt/xfeat/weights/xfeat.pt"


def unfold2d(self, x, ws=2):
    # Same as XFeatModel._unfold2d (channel = c*ws*ws + row*ws + col), written
    # with pixel_unshuffle so the exported graph keeps H and W dynamic.
    return F.pixel_unshuffle(x, ws)


class XFeatDense(torch.nn.Module):
    def __init__(self, net: XFeatModel):
        super().__init__()
        self.net = net

    def forward(self, image):
        feats, kpts_logits, reliability = self.net(image)
        feats = F.normalize(feats, dim=1)
        # XFeat.get_kpts_heatmap: softmax over 65 bins, drop the dustbin, and
        # put the 8x8 cell back to full resolution (= pixel_shuffle).
        heatmap = F.pixel_shuffle(F.softmax(kpts_logits, 1)[:, :64], 8)
        return feats, heatmap, reliability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="xfeat.onnx")
    parser.add_argument("--reference", help="image for the C++ reference output")
    parser.add_argument("--top-k", type=int, default=2000)
    args = parser.parse_args()

    net = XFeatModel().eval()
    net.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))

    # pixel_unshuffle == the original _unfold2d
    x = torch.rand(1, 1, 64, 96)
    original = XFeatModel._unfold2d(net, x, ws=8)
    XFeatModel._unfold2d = unfold2d
    assert torch.equal(original, net._unfold2d(x, ws=8)), "pixel_unshuffle != _unfold2d"

    model = XFeatDense(net).eval()
    dummy = torch.rand(1, 1, 480, 640) * 255
    torch.onnx.export(
        model, (dummy,), args.out, opset_version=17,
        input_names=["image"], output_names=["feats", "heatmap", "reliability"],
        dynamic_axes={"image": {2: "H", 3: "W"}, "feats": {2: "h", 3: "w"},
                      "heatmap": {2: "H", 3: "W"}, "reliability": {2: "h", 3: "w"}})
    print(f"Wrote {args.out}")

    # ONNX Runtime vs PyTorch, at two sizes (dynamic shapes).
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    xfeat = XFeat(weights=WEIGHTS)  # the official inference module
    for h, w in [(480, 640), (480, 768)]:
        image = torch.rand(1, 1, h, w) * 255
        with torch.no_grad():
            ref = model(image)
            # heatmap as computed by the official code
            _, k1, _ = xfeat.net(image)
            ref_heatmap = xfeat.get_kpts_heatmap(k1)
        out = session.run(None, {"image": image.numpy()})
        err = [float(np.abs(o - r.numpy()).max()) for o, r in zip(out, ref)]
        err_heatmap = float(np.abs(out[1] - ref_heatmap.numpy()).max())
        print(f"{h}x{w}: max |onnx - torch| feats {err[0]:.2e}, heatmap {err[1]:.2e} "
              f"(vs get_kpts_heatmap {err_heatmap:.2e}), reliability {err[2]:.2e}")
        assert max(err + [err_heatmap]) < 1e-4

    if args.reference:
        gray = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
        image = torch.from_numpy(gray).float()[None, None]
        result = xfeat.detectAndCompute(image, top_k=args.top_k)[0]
        path = args.out + ".reference.txt"
        with open(path, "w") as f:
            for k, s, d in zip(result["keypoints"].numpy(), result["scores"].numpy(),
                               result["descriptors"].numpy()):
                f.write(" ".join(f"{v:.6f}" for v in [*k, s, *d]) + "\n")
        print(f"Reference: {len(result['keypoints'])} keypoints -> {path}")


if __name__ == "__main__":
    main()
