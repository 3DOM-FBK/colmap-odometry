import os
import sys
import cv2
import time
import warnings
import torch
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from ..thirdparty.ALIKED.nets.aliked import ALIKED
#from thirdparty.SuperPoint_open import superpoint_pytorch
from ..thirdparty.SuperPoint_open import superpoint_pytorch

# LoMa ships as a src-layout package (thirdparty/LoMa/src/loma) that uses
# absolute "loma." imports. Expose its src directory so "import loma" works,
# mirroring how ALIKED / SuperPoint_open are bundled under thirdparty.
_LOMA_SRC = Path(__file__).resolve().parent / ".." / "thirdparty" / "LoMa" / "src"
if _LOMA_SRC.is_dir() and str(_LOMA_SRC) not in sys.path:
    sys.path.insert(0, str(_LOMA_SRC))

DEBUG = False

class LocalFeatures:
    def __init__(
            self,
            image_width: int,
            image_height: int,
            config_local_features: dict,
            verbose: bool = False,
            ) -> None:
        self.verbose = verbose
        self.feature_name = config_local_features['features_name']
        self.image_width = image_width
        self.image_height = image_height
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.size ={
            "height": config_local_features['resize_height'],
            "width": config_local_features['resize_width'],
        }
        self.config_sp = config_local_features.get('superpoint', {})
        self.config_sp_open = config_local_features.get('superpoint_open', {})
        config_aliked = config_local_features.get('aliked', {})
        config_loma = config_local_features.get('loma', {})
        self.loma_input_height = ((self.size['height'] + 13) // 14) * 14
        self.loma_input_width = ((self.size['width'] + 13) // 14) * 14

        if self.feature_name == "superpoint":
            from transformers import AutoImageProcessor, SuperPointForKeypointDetection

            self.processor = AutoImageProcessor.from_pretrained("magic-leap-community/superpoint", do_resize=self.config_sp['do_resize'], size=self.size, use_fast=True)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if self.device == torch.device("cuda"):
                self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint").cuda()
            else:
                self.model = SuperPointForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
            self.model.eval()
        
        elif self.feature_name == "aliked":
            self.model = ALIKED(model_name=config_aliked['model_name'], device=self.device, top_k=config_aliked['top_k'], scores_th=config_aliked['scores_th'], n_limit=config_aliked['n_limit'])

        elif self.feature_name == "loma":
            # LoMa bundles a DaD detector + DeDoDe descriptor + LoMa matcher.
            # Here we only need the detector/descriptor part for extraction;
            # the matching transformer is driven from odometry.py.
            from loma import LoMa, LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR
            loma_models = {
                "loma_B128": LoMaB128,
                "loma_B": LoMaB,
                "loma_L": LoMaL,
                "loma_G": LoMaG,
                "loma_R": LoMaR,
            }
            model_name = config_loma.get('model_name', 'loma_B')
            # Use TensorFloat32 tensor cores for the large float32 matmuls in
            # LoMa. This is the performance setting recommended by PyTorch;
            # it does not change the model's input/output precision.
            torch.set_float32_matmul_precision("high")
            loma_cfg = loma_models[model_name]()
            # dataclasses.replace lets us override num_keypoints on the frozen cfg
            import dataclasses
            loma_cfg = dataclasses.replace(
                loma_cfg,
                num_keypoints=config_loma.get('top_k', loma_cfg.num_keypoints),
                compile=config_loma.get('compile', False),
            )
            self.loma_num_keypoints = loma_cfg.num_keypoints
            self.loma_compile_enabled = bool(loma_cfg.compile)
            self.model = LoMa(loma_cfg)

        elif self.feature_name == "superpoint_open":
            if self.device == 'cuda':
                self.model = superpoint_pytorch.SuperPoint(detection_threshold=0.005, max_num_keypoints=config_local_features['superpoint_open']['top_k'], nms_radius=5).cuda().eval()
            else:
                self.model = superpoint_pytorch.SuperPoint(detection_threshold=0.005, max_num_keypoints=config_local_features['superpoint_open']['top_k'], nms_radius=5).eval()
            try:
                current_dir = Path(__file__).parent
                weights_path = current_dir / ".." / "thirdparty" / "SuperPoint_open" / "weights" / "superpoint_v6_from_tf.pth"
                state_dict = torch.load(weights_path)
            except:
                state_dict = torch.load('/home/carla/thirdparty/COLMAP_SLAM/src/thirdparty/SuperPoint_open/weights/superpoint_v6_from_tf.pth')
            self.model.load_state_dict(state_dict)
            
        elif self.feature_name == "orb":
            config_orb = config_local_features['orb']
            self.orb_detector = cv2.ORB_create(
                nfeatures=config_orb.get('top_k', 1000),
                scaleFactor=config_orb.get('scaleFactor', 1.2),
                nlevels=config_orb.get('nlevels', 8),
                edgeThreshold=config_orb.get('edgeThreshold', 31),
                firstLevel=config_orb.get('firstLevel', 0),
                WTA_K=config_orb.get('WTA_K', 2),
                scoreType=config_orb.get('scoreType', cv2.ORB_HARRIS_SCORE),
                patchSize=config_orb.get('patchSize', 31),
                fastThreshold=config_orb.get('fastThreshold', 20)
            )
            
            #sp_th = superpoint_pytorch.SuperPoint(detection_threshold=0.005, nms_radius=5).eval()
            #state_dict = torch.load('/home/threedom/Desktop/github_3dom/COLMAP_SLAM/src/thirdparty/SuperPoint_open/weights/superpoint_v6_from_tf.pth')
            #sp_th.load_state_dict(state_dict)
            #print('Config:', sp_th.conf)
            #image_url = 'https://raw.githubusercontent.com/cvg/Hierarchical-Localization/master/datasets/sacre_coeur/mapping/03903474_1471484089.jpg'
            #image_path = 'image_sacre_coeur.jpg'
            #torch.hub.download_url_to_file(image_url, image_path)
#
            #image = cv2.imread(image_path).mean(-1) / 255
            #image = np.pad(image, [(0, int(np.ceil(s/8))*8 - s) for s in image.shape[:2]])
            #
            #with torch.no_grad():
            #    pred_th = sp_th({'image': torch.from_numpy(image[None,None]).float()})
            #points_th = pred_th['keypoints'][0]
            #print(points_th.shape)
            #descriptors_th = pred_th['descriptors']
            #print(descriptors_th[0].shape)
            #print('ok')
            #quit()

    @torch.inference_mode()
    def warmup_loma(self) -> float:
        """Compile and execute LoMa on representative fixed-size dummy inputs."""
        if self.feature_name != "loma" or not getattr(
            self, "loma_compile_enabled", False
        ):
            return 0.0
        started = time.monotonic()
        device = next(self.model.parameters()).device
        dummy = torch.rand(
            (1, 3, self.loma_input_height, self.loma_input_width),
            dtype=torch.float32,
            device=device,
        )
        # Exercise detector, descriptor, and the matching transformer. The
        # production path uses the same padded dimensions and keypoint budget.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Online softmax is disabled on the fly.*",
                module=r"torch\.\_inductor\.lowering",
            )
            keypoints0, descriptors0, _, _ = self.model.detect_and_describe(
                dummy, num_keypoints=self.loma_num_keypoints)
            keypoints1, descriptors1, _, _ = self.model.detect_and_describe(
                dummy, num_keypoints=self.loma_num_keypoints)
            self.model(keypoints0, keypoints1, descriptors0, descriptors1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        del dummy, keypoints0, descriptors0, keypoints1, descriptors1
        return time.monotonic() - started

    def superpoint(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        reading_image_status = True
        try:
            image = cv2.resize(image, (self.size['width'], self.size['height']))
        except:
            reading_image_status = False
            return {}, {}, reading_image_status

        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}
        
        with torch.no_grad():
            images = [image]
            inputs = self.processor(images, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            scores = outputs.scores[0]
            topk = self.config_sp['top_k']
            if scores.shape[0] < topk:
                topk = scores.shape[0]
            topk_indices = torch.topk(scores, topk).indices

            kpts = outputs['keypoints'][0]
            kpts[:, 0] *= self.size['width']  # Rescale first column (y) by resize height
            kpts[:, 1] *= self.size['height'] # Rescale second column (x) by resize width
            kpts = kpts[topk_indices]
            kpts[:, 0] *= self.image_width / self.size['width']
            kpts[:, 1] *= self.image_height / self.size['height']
            keypoints[img_name] = kpts.to("cpu")
            descriptors[img_name] = outputs['descriptors'][0][topk_indices].to("cpu")
                
            del inputs, outputs
            torch.cuda.empty_cache()


        if self.verbose==True and DEBUG==True:
            t1 = time.time()
            print(f"[CSLAM] Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors, reading_image_status

    # Old version of superpoint() working in batches:
    def superpoint_batch(self, imgs_dir: Path, image_files: list, batch_size: int) -> Tuple[dict, dict]:
        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}
        steps = len(image_files) // batch_size
        rest = len(image_files) % batch_size

        with torch.no_grad():
            for i in range(steps):
                images = []
                for k in range(batch_size):
                    img = image_files[i*batch_size+k]
                    image = Image.open(imgs_dir / img).convert("RGB")
                    images.append(image)
                inputs = self.processor(images, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)

                for k in range(batch_size):
                    kpts = outputs['keypoints'][k]
                    kpts[:, 0] *= self.image_width / self.size['width']
                    kpts[:, 1] *= self.image_height / self.size['height']
                    keypoints[image_files[i*batch_size+k]] = kpts.to("cpu")
                    descriptors[image_files[i*batch_size+k]] = outputs['descriptors'][k].to("cpu")
                
                del inputs, outputs
                torch.cuda.empty_cache()

            if rest > 0:
                images = []
                for k in range(rest):
                    img = image_files[steps*batch_size+k]
                    image = Image.open(imgs_dir / img).convert("RGB")
                    images.append(image)

                inputs = self.processor(images, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                for k in range(rest):
                    kpts = outputs['keypoints'][k]
                    kpts[:, 0] *= self.width / self.size['width']
                    kpts[:, 1] *= self.height / self.size['height']
                    keypoints[image_files[steps*batch_size+k]] = kpts.to("cpu")
                    descriptors[image_files[steps*batch_size+k]] = outputs['descriptors'][k].to("cpu")

                del inputs, outputs
                torch.cuda.empty_cache()

        if self.verbose==True and DEBUG==True:
            t1 = time.time()
            print(f"[CSLAM] Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors

    def aliked(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        reading_image_status = True
        try:
            image = cv2.resize(image, (self.size['width'], self.size['height']))
        except:
            reading_image_status = False
            return {}, {}, reading_image_status
        # ALIKED's backbone expects 3-channel images. Replicate grayscale
        # input (e.g. mono8 topics from the TUM-VI dataset) to 3 channels.
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        resize_factor = self.size['width'] / self.image_width
        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}
        
        with torch.no_grad():
            pred = self.model.run(image)
            keypoints[img_name] = torch.from_numpy(pred['keypoints']).to("cpu")/resize_factor
            descriptors[img_name] = torch.from_numpy(pred['descriptors']).to("cpu")

        if self.verbose==True and DEBUG==True:
            t1 = time.time()
            print(f"[CSLAM] Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors, reading_image_status

    def loma(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        reading_image_status = True
        try:
            image = cv2.resize(image, (self.size['width'], self.size['height']))
        except:
            reading_image_status = False
            return {}, {}, reading_image_status
        # DaD / DeDoDe expect a 3-channel RGB float tensor in [0, 1].
        # Replicate grayscale input (e.g. mono8 topics from TUM-VI) to 3
        # channels, mirroring the ALIKED grayscale handling above.
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.ndim == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        # DeDoDe's ViT backbone requires H and W to be multiples of the patch
        # size (14). Pad the resized image up to the next multiple of 14;
        # keypoints stay in normalized coords of the padded image, so the
        # pixel conversion below remains consistent.
        pad_h = self.loma_input_height - image.shape[0]
        pad_w = self.loma_input_width - image.shape[1]
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)))
        resize_factor = self.size['width'] / self.image_width
        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}

        with torch.no_grad():
            tensor = torch.from_numpy(image / 255.0).permute(2, 0, 1).float()[None]
            kpts, descs, h, w = self.model.detect_and_describe(
                tensor, num_keypoints=self.loma_num_keypoints
            )
            # kpts are normalized in [-1, 1]; convert to pixels of the resized
            # image, then scale back to the original image resolution.
            kpts_px = self.model._detector.to_pixel_coords(kpts[0], h, w)
            keypoints[img_name] = (kpts_px / resize_factor).to("cpu")
            descriptors[img_name] = descs[0].to("cpu")

        if self.verbose==True and DEBUG==True:
            t1 = time.time()
            print(f"[CSLAM] Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors, reading_image_status

    def superpoint_open(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        reading_image_status = True
        try:
            image = cv2.resize(image, (self.size['width'], self.size['height']))
        except:
            reading_image_status = False
            return {}, {}, reading_image_status
        resize_factor = self.size['width'] / self.image_width
        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}
        
        with torch.no_grad():
            # Normalize any input to a single grayscale channel.
            if image.ndim == 3 and image.shape[2] == 3:
                image = image.mean(-1) / 255
            elif image.ndim == 3 and image.shape[2] == 1:
                image = image[:, :, 0] / 255
            else:
                image = image / 255
            image = np.pad(image, [(0, int(np.ceil(s/8))*8 - s) for s in image.shape[:2]])
            pred = self.model({'image': torch.from_numpy(image[None,None]).to(self.device).float()})
            keypoints[img_name] = pred['keypoints'][0].to("cpu")/resize_factor
            descriptors[img_name] = pred['descriptors'][0].to("cpu")

        if self.verbose==True and DEBUG==True:
            t1 = time.time()
            print(f"[CSLAM] Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors, reading_image_status

    def orb(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        reading_image_status = True
        try:
            image = cv2.resize(image, (self.size['width'], self.size['height']))
        except:
            reading_image_status = False
            return {}, {}, reading_image_status
        resize_factor = self.size['width'] / self.image_width
        if self.verbose: t0 = time.time()
        keypoints = {}
        descriptors = {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect keypoints and compute descriptors
        kpts, descs = self.orb_detector.detectAndCompute(gray, None)
        
        if kpts is not None and descs is not None:
            # Convert keypoints to numpy array format
            kpts_array = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts], dtype=np.float32)
            
            # Scale keypoints back to original image size
            kpts_array = kpts_array / resize_factor
            
            # Convert to torch tensors
            keypoints[img_name] = torch.from_numpy(kpts_array)
            descriptors[img_name] = torch.from_numpy(descs.astype(np.float32))
        else:
            # Handle case where no keypoints are detected
            keypoints[img_name] = torch.empty((0, 2), dtype=torch.float32)
            descriptors[img_name] = torch.empty((0, 32), dtype=torch.float32)  # ORB descriptors are 32-dimensional

        if self.verbose==True and DEBUG==True:
            t1 = time.time()
            print(f"[CSLAM] Feature extraction time: {t1-t0:.2f} seconds")

        return keypoints, descriptors, reading_image_status

    def extract(self, img_name: str, image: np.ndarray) -> Tuple[dict, dict]:
        if self.feature_name == "superpoint":
            return self.superpoint(img_name, image)
        elif self.feature_name == "aliked":
            return self.aliked(img_name, image)
        elif self.feature_name == "loma":
            return self.loma(img_name, image)
        elif self.feature_name == "superpoint_open":
            return self.superpoint_open(img_name, image)
        elif self.feature_name == "orb":
            return self.orb(img_name, image)

    def extract_batch(
            self,
            items: List[Tuple[str, np.ndarray]],
            ) -> Tuple[dict, dict, dict]:
        """Extract a synchronized image set through the backend's image API."""
        keypoints = {}
        descriptors = {}
        statuses = {}
        for img_name, image in items:
            result = self.extract(img_name, image)
            if result is None or len(result) != 3:
                raise RuntimeError(
                    f"Feature backend '{self.feature_name}' returned an "
                    "invalid extraction result")
            kpts, descs, status = result
            keypoints.update(kpts)
            descriptors.update(descs)
            statuses[img_name] = bool(status)
        return keypoints, descriptors, statuses

    def decriptor_dim(self) -> int:
        if self.feature_name == "superpoint":
            return 256
        elif self.feature_name == "aliked":
            return 128
        elif self.feature_name == "loma":
            return 256
        elif self.feature_name == "superpoint_open":
            return 256
        elif self.feature_name == "orb":
            return 32
        
