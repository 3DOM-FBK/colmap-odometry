import os
import cv2
import time
import copy
import torch
import shutil
import numpy as np
import pycolmap
import kornia.feature as KF
from kornia.feature import match_nn, match_smnn

from typing import List, Tuple, Dict, Any
from pyquaternion import Quaternion
from pathlib import Path

from .local_features import LocalFeatures
from .db_colmap import COLMAPDatabase
from .stereo_temporal_covariance import (
    estimate_stereo_temporal_covariance,
    optimize_stereo_temporal_pose,
)
from ..rig_transforms import colmap_camera_from_rig_matrix


def _covariance_for_inverse(rigid3d: pycolmap.Rigid3d,
                            covariance: np.ndarray) -> np.ndarray:
    """Use the corrected PyCOLMAP inverse-covariance API across releases."""
    module_function = getattr(pycolmap, "get_covariance_for_inverse", None)
    if module_function is not None:
        return np.asarray(module_function(rigid3d, covariance), dtype=np.float64)
    method = getattr(rigid3d, "get_covariance_for_inverse", None)
    if method is not None:
        return np.asarray(method(covariance), dtype=np.float64)
    raise RuntimeError(
        "PyCOLMAP does not expose covariance propagation for pose inversion")

from pycolmap import Database, Camera, Image, Rigid3d, Rotation3d, TwoViewGeometry, logging

try:
    from pycolmap import ListPoint2D as _ListPoint2D
except ImportError:  # PyCOLMAP >= 3.12 renamed the container.
    from pycolmap import Point2DList as _ListPoint2D


def _empty_point2d_list():
    """Construct an empty image-point container across PyCOLMAP releases."""
    try:
        return _ListPoint2D(np.empty((0, 2), dtype=np.float64))
    except (TypeError, ValueError):
        return _ListPoint2D()


def _new_database_image(name: str, camera_id: int, image_id: int):
    """Create an unposed COLMAP database image across PyCOLMAP releases."""
    # ``Image.cam_from_world`` became read-only and ``id`` was renamed to
    # ``image_id`` in PyCOLMAP 3.12.  The database only needs the image name,
    # keypoint container and identifiers at this stage; reconstruction fills
    # the pose later.
    return Image(
        name=name,
        points2D=_empty_point2d_list(),
        camera_id=int(camera_id),
        image_id=int(image_id),
    )
import sqlite3
import time


def reconstruct(*args, **kwargs):
    from .custom_incremental_pipeline import reconstruct as reconstruct_impl
    return reconstruct_impl(*args, **kwargs)


def quat(colmap_quat: np.ndarray) -> Quaternion:
    x, y, z, w = colmap_quat
    return Quaternion(np.array([w, x, y, z]))


def rigid3d_to_matrix(transform: pycolmap.Rigid3d) -> np.ndarray:
    """Return COLMAP Rigid3d as a 4x4 matrix with the same direction."""
    x, y, z, w = transform.rotation.quat
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Quaternion(np.array([w, x, y, z])).rotation_matrix
    matrix[:3, 3] = np.asarray(transform.translation, dtype=np.float64).reshape(3)
    return matrix


def image_cam_from_world(image: pycolmap.Image) -> pycolmap.Rigid3d:
    """Read Image.cam_from_world across PyCOLMAP property/method APIs."""
    value = image.cam_from_world
    return value() if callable(value) else value


def matrix_to_rigid3d(transform: np.ndarray) -> pycolmap.Rigid3d:
    """Create a COLMAP Rigid3d without changing the matrix direction."""
    quaternion = Quaternion(matrix=transform[:3, :3]).normalised
    return pycolmap.Rigid3d(
        rotation=pycolmap.Rotation3d(
            [quaternion.x, quaternion.y, quaternion.z, quaternion.w]),
        translation=np.asarray(transform[:3, 3], dtype=np.float64),
    )


def triangulate_two_view(
    point0: np.ndarray,
    point1: np.ndarray,
    cam0_from_world: pycolmap.Rigid3d,
    cam1_from_world: pycolmap.Rigid3d,
    camera0: pycolmap.Camera,
    camera1: pycolmap.Camera,
) -> np.ndarray | None:
    """Deterministically triangulate two calibrated pixel observations.

    The fixed rig extrinsics are already known, so linear DLT provides an
    explicit two-view initializer. Callers perform cheirality, depth,
    reprojection, and ray-angle validation afterwards.
    """
    normalized0 = np.asarray(
        camera0.cam_from_img(np.asarray(point0, dtype=np.float64).reshape(1, 2)),
        dtype=np.float64,
    ).reshape(2)
    normalized1 = np.asarray(
        camera1.cam_from_img(np.asarray(point1, dtype=np.float64).reshape(1, 2)),
        dtype=np.float64,
    ).reshape(2)
    projection0 = rigid3d_to_matrix(cam0_from_world)[:3, :]
    projection1 = rigid3d_to_matrix(cam1_from_world)[:3, :]
    system = np.vstack((
        normalized0[0] * projection0[2] - projection0[0],
        normalized0[1] * projection0[2] - projection0[1],
        normalized1[0] * projection1[2] - projection1[0],
        normalized1[1] * projection1[2] - projection1[1],
    ))
    if not np.all(np.isfinite(system)):
        return None
    _, _, vh = np.linalg.svd(system)
    homogeneous = vh[-1]
    if not np.all(np.isfinite(homogeneous)) or abs(homogeneous[3]) <= 1.0e-12:
        return None
    point = homogeneous[:3] / homogeneous[3]
    return point if np.all(np.isfinite(point)) else None


def matrix_to_pose(transform: np.ndarray) -> Tuple[np.ndarray, Quaternion]:
    """Return translation and quaternion from a ROS-style pose matrix."""
    u, _, vh = np.linalg.svd(transform[:3, :3])
    rotation = u @ vh
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vh
    return transform[:3, 3].copy(), Quaternion(matrix=rotation)


def colmap_cam_from_world_to_pose(cam_from_world: pycolmap.Rigid3d) -> Tuple[np.ndarray, Quaternion]:
    """Convert COLMAP cam_from_world to camera pose in world coordinates."""
    return matrix_to_pose(np.linalg.inv(rigid3d_to_matrix(cam_from_world)))


def relative_camera_pose(
    ref_cam_from_world: pycolmap.Rigid3d,
    cur_cam_from_world: pycolmap.Rigid3d,
) -> Tuple[np.ndarray, Quaternion]:
    """Pose of current camera in the reference camera frame.

    COLMAP stores T_camera_world, i.e. X_camera = R * X_world + t.
    ROS poses usually describe T_world_camera, i.e. X_world = T * X_camera.
    The current camera pose in the reference camera frame is therefore:

        T_ref_cur = T_ref_world * inverse(T_cur_world)
    """
    T_ref_world = rigid3d_to_matrix(ref_cam_from_world)
    T_cur_world = rigid3d_to_matrix(cur_cam_from_world)
    return matrix_to_pose(T_ref_world @ np.linalg.inv(T_cur_world))


def compose_pose(
    cumulative_t: np.ndarray,
    cumulative_q: Quaternion,
    delta_t: np.ndarray,
    delta_q: Quaternion,
) -> Tuple[np.ndarray, Quaternion]:
    """Compose T_initial_current = T_initial_previous * T_previous_current."""
    delta_t = np.asarray(delta_t, dtype=np.float64).reshape(3)
    new_t = np.asarray(cumulative_t, dtype=np.float64).reshape(3) + cumulative_q.rotate(delta_t)
    new_q = cumulative_q * delta_q
    return new_t, new_q.normalised


class VisualOdometry:
    """
    Optimized version focusing on:
    - avoiding deep copies and redundant allocations
    - minimizing GPU<->CPU transfers
    - in-place dict updates
    - caching LightGlue LAFs per image
    - reducing repeated DB reloads
    - comprehensive logging data collection
    """

    def __init__(
        self,
        working_dir: Path,
        config: dict,
        camera_config: dict,
        rig_config: dict,
    ) -> None:
        logging.verbose_level = 0
        logging.minloglevel = 2

        # --- state ---
        self.snapshot_count = 0
        self.keyframes_names: Dict[str, int] = {}
        self.keyframes_ids: Dict[int, str] = {}
        self.keyframes_master_ids: List[int] = []
        self.config = config
        self.camera_config = camera_config
        self.use_min_solver = config.get('general', {}).get('min_solver', True)
        # The minimal stereo/PnP route is a frame-to-frame odometer, not an
        # incremental mapper.  It must therefore process every synchronized
        # rig sample rather than use the legacy feature-motion keyframe gate.
        self.process_every_frame = bool(config.get('general', {}).get(
            'process_every_frame', self.use_min_solver))
        # Incremental reconstruction requires COLMAP's SQLite database.  The
        # minimal route retains all state in memory, so writing the database
        # is optional (and disabled by default) there.
        self.write_colmap_database = bool(config.get('general', {}).get(
            'write_colmap_database', False)) or not self.use_min_solver
        self.baseline = config['mapping']['baseline']
        self.log = config['general']['log']
        self.rig_match_rule = config['mapping']['rig_match_rule']
        self.height, self.width = camera_config['cam0']['height'], camera_config['cam0']['width']
        self.N_reinit = 0
        self.run_BA = False  # control BA only after last stereo image is added
        self.current_status = 'initializing'
        # Read by the ROS wrapper's independent stall watchdog. Keep this a
        # plain string so updating it does not add synchronization overhead to
        # the hot path.
        self.performance_stage = 'initializing'
        self.frame_t0 = None
        self.frame_t1 = None
        self.feature_track_publish_min_length = max(
            2,
            int(config.get('general', {}).get(
                'feature_track_publish_min_length', 5)),
        )
        self.create_feature_tracks = bool(
            config.get('general', {}).get('create_feature_tracks', True))
        self._next_feature_track_id = 0
        self._feature_tracks: Dict[int, Dict[str, Any]] = {}
        self._active_feature_lookup: Dict[Tuple[str, int], int] = {}
        self._feature_frame_indices: Dict[str, int] = {}
        self._stereo_observations: Dict[str, Dict[int, Dict[str, Any]]] = {}
        # Retain geometry-valid stereo bearings even when metric triangulation
        # is rejected for shallow parallax or distant depth.  The final
        # inverse-depth temporal solve can still use them for rotation.
        self._stereo_bearing_observations: Dict[
            str, Dict[int, Dict[str, Any]]
        ] = {}
        self._temporal_matches: Dict[Tuple[str, str], np.ndarray] = {}
        # MAGSAC is the primary calibrated-essential temporal verifier. Retain
        # mutually matched candidates as a PnP-only recovery source because
        # essential estimation is degenerate during stationary/near-pure-
        # rotation intervals, whereas calibrated 2D--3D PnP remains well posed.
        self._temporal_raw_matches: Dict[Tuple[str, str], np.ndarray] = {}
        self._raw_feature_matches: Dict[Tuple[str, str], np.ndarray] = {}
        self.last_feature_track_updates: List[Dict[str, Any]] = []

        # Enhanced logging data structure
        self.log_data: Dict[str, Any] = {
            'frame_count': 0,
            'keyframe_count': 0,
            'total_features_extracted': 0,
            'total_matches_computed': 0,
            'total_feature_tracks_published': 0,
            'reinitializations': 0,
            'timing': {
                'feature_extraction_total': 0.0,
                'feature_matching_total': 0.0,
                'reconstruction_total': 0.0,
                'pose_estimation_total': 0.0,
                'frame_processing_total': 0.0,
                'database_operations_total': 0.0
            },
            'performance': {
                'avg_features_per_frame': 0.0,
                'avg_matches_per_pair': 0.0,
                'avg_frame_processing_time': 0.0,
                'keyframe_selection_ratio': 0.0
            },
            'reconstruction_stats': {
                'successful_reconstructions': 0,
                'failed_reconstructions': 0,
                'sliding_window_operations': 0,
                'bundle_adjustments': 0
            },
            'memory_stats': {
                'peak_gpu_memory_mb': 0.0,
                'current_cached_features': 0,
                'current_cached_lafs': 0
            },
            'current_frame': {
                'frame_name': '',
                'is_keyframe': False,
                'num_features': 0,
                'inlier_matches': 0.0,
                'inlier_ratio': 0.0,
                'median_match_distance': 0.0,
                'processing_time': 0.0,
                'feature_extraction_time': 0.0,
                'feature_model_time': 0.0,
                'feature_tensor_transfer_time': 0.0,
                'matching_time': 0.0,
                'matching_model_time': 0.0,
                'matching_geometry_time': 0.0,
                'matching_database_time': 0.0,
                'keyframe_selection_time': 0.0,
                'stereo_observation_time': 0.0,
                'database_time': 0.0,
                'reconstruction_time': 0.0,
                'covariance_matrix_angles_translations': None,
                'relative_pose_covariance_rotation_translation': None,
                'pnp_covariance_source': 'none',
                'pnp_covariance_points': 0,
                'pnp_covariance_rank': 0,
                'pnp_covariance_condition_number': None,
                'pnp_covariance_reprojection_rms_px': None,
                'feature_track_updates': 0,
                'stereo_matches_raw': 0,
                'stereo_matches_epipolar': 0,
                'stereo_points_candidates': 0,
                'stereo_points_triangulated': 0,
                'stereo_points_valid': 0,
                'stereo_rejected_indices': 0,
                'stereo_rejected_triangulation': 0,
                'stereo_rejected_cheirality': 0,
                'stereo_rejected_depth': 0,
                'stereo_rejected_reprojection': 0,
                'stereo_rejected_angle': 0,
            }
        }

        self.images_dir = working_dir / "images"

        self.test = self.config['general']['test']
        self.cameras = sorted(self.config['mapping']['cameras'], key=lambda x: int(x[3:]))
        self.n_cameras = len(self.cameras)
        self.cameras_for_baseline_estim = config['mapping']['cameras_for_baseline_estim']
        if "cam0" not in self.cameras_for_baseline_estim:
            raise ValueError("cam0 must be included in cameras_for_baseline_estim")
        # pick a second camera for scale estimation
        self.second_baseline_camera = next((c for c in self.cameras_for_baseline_estim if c != "cam0"), "cam1" if self.n_cameras > 1 else "cam0")

        # Rig YAML entries are ROS poses T_rig_camera. COLMAP projections,
        # epipolar geometry, and triangulation require the inverse direction,
        # T_camera_rig (called cam_from_world/cam_from_rig by COLMAP).
        self.cam0_pose_t0 = matrix_to_rigid3d(
            colmap_camera_from_rig_matrix(rig_config, 'cam0'))
        if self.n_cameras > 1:
            self.cam1_pose_t0 = matrix_to_rigid3d(
                colmap_camera_from_rig_matrix(rig_config, 'cam1'))
        else:
            self.cam1_pose_t0 = self.cam0_pose_t0  # dummy, won't be used

        self.camera0 = pycolmap.Camera(camera_config['cam0'])
        if self.n_cameras > 1:
            self.camera1 = pycolmap.Camera(camera_config['cam1'])

        mapping_config = config['mapping']
        self.stereo_epipolar_max_error_px = float(
            mapping_config.get('stereo_epipolar_max_error_px', 1.5))
        self.stereo_reprojection_max_error_px = float(
            mapping_config.get('stereo_reprojection_max_error_px', 2.0))
        self.stereo_min_triangulation_angle_deg = float(
            mapping_config.get('stereo_min_triangulation_angle_deg', 0.1))
        self.stereo_min_depth = float(
            mapping_config.get('stereo_min_depth', 0.1))
        self.stereo_max_depth = float(
            mapping_config.get('stereo_max_depth', 100.0))
        if self.stereo_epipolar_max_error_px <= 0.0:
            raise ValueError("stereo_epipolar_max_error_px must be positive")
        if self.stereo_reprojection_max_error_px <= 0.0:
            raise ValueError("stereo_reprojection_max_error_px must be positive")
        if self.stereo_min_triangulation_angle_deg < 0.0:
            raise ValueError("stereo_min_triangulation_angle_deg cannot be negative")
        if not 0.0 <= self.stereo_min_depth < self.stereo_max_depth:
            raise ValueError(
                "stereo depth limits must satisfy 0 <= min < max")
        covariance_config = config.get('covariance', {})
        self.stereo_covariance_pixel_std_px = float(
            covariance_config.get(
                'pixel_std_px',
                mapping_config.get('stereo_covariance_pixel_std_px', 1.0),
            ))
        self.stereo_covariance_min_points = max(
            6,
            int(covariance_config.get(
                'min_points',
                mapping_config.get('stereo_covariance_min_points', 6),
            )),
        )
        self.stereo_covariance_spatial_clusters = max(
            1,
            int(covariance_config.get(
                'spatial_clusters',
                mapping_config.get(
                    'stereo_covariance_spatial_clusters', 24),
            )),
        )
        if not np.isfinite(self.stereo_covariance_pixel_std_px) or \
                self.stereo_covariance_pixel_std_px <= 0.0:
            raise ValueError('covariance.pixel_std_px must be positive')
        self.structureless_max_iterations = max(
            1,
            int(mapping_config.get('structureless_max_iterations', 4)),
        )
        self.structureless_robust_delta_px = max(
            0.1,
            float(mapping_config.get('structureless_robust_delta_px', 3.0)),
        )

        # features storage
        self.keypoints: Dict[str, torch.Tensor] = {}
        self.descriptors: Dict[str, torch.Tensor] = {}
        self.lafs_cache: Dict[str, torch.Tensor] = {}  # cached LAFs to avoid recomputation
        self.last_feature_match_debug: List[Dict[str, Any]] = []

        self.database_path = working_dir / "database.db"
        self._remove_database_files()

        self.out_dir = working_dir / "out"
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
        #self.log_file = open(self.out_dir / "log.txt", "w")

        self.images: List[str] = []

        # Local feature extractor + LightGlue
        self.local_features = LocalFeatures(
            self.width,
            self.height,
            config['local_features'],
            self.log,
        )
        if config['local_features']['features_name'] == "aliked":
            self.lightglue_model = "aliked"
        elif config['local_features']['features_name'] == "superpoint":
            self.lightglue_model = "superpoint"
        elif config['local_features']['features_name'] == "superpoint_open":
            self.lightglue_model = "superpoint"
        #else:
        #    raise ValueError("Invalid local features model")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lightglue_dict = {
            "name": "lightglue",
            "input_dim": 256,
            "mp": True,  # enable mixed precision
            "filter_threshold": float(config['local_features'].get(
                'lightglue_filter_threshold', 0.1)),
            "depth_confidence": float(config['local_features'].get(
                'lightglue_depth_confidence', 0.95)),
            "width_confidence": float(config['local_features'].get(
                'lightglue_width_confidence', 0.99)),
        }
        if (
            config['local_features']['features_name'] == "superpoint_open"
            and 'lightglue_filter_threshold' not in config['local_features']
        ):
            lightglue_dict["filter_threshold"] = 0.00005 # default 0.00005

        self.matcher = config['local_features']['matcher']
        if self.matcher == "lightglue":
            self.lg_matcher = KF.LightGlueMatcher(self.lightglue_model, lightglue_dict).eval().to(self.device)
        elif self.matcher == "loma":
            # Reuse the LoMa model already built by LocalFeatures (it bundles
            # the matching transformer) instead of instantiating a second copy.
            self.loma_matcher = self.local_features.model
            self.loma_filter_threshold = float(
                config['local_features'].get('loma', {}).get(
                    'filter_threshold', self.loma_matcher.cfg.filter_threshold))
        self.loma_warmup_seconds = 0.0
        self.ratio_threshold = config['local_features']['ratio_threshold'];print(self.ratio_threshold)

        # Initialize database and odometry variables
        self.baseline_old = 0.0
        self.keyframe_count = 1
        self.keyframe_id = 1
        self.frame_t0 = None
        self.frame_t1 = None
        self.t_cumulative = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.q_cumulative = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))
        self.db = (Database(str(self.database_path))
                   if self.write_colmap_database else None)
        self.db_dirty = False  # track when DB has new data not yet loaded into controller

        # Mapper options
        self.sliding_window = self.config['mapping']['sliding_window']
        pycolmap.set_random_seed(0)
        self.options = pycolmap.IncrementalPipelineOptions()
        self.options.ba_refine_focal_length = config['bundle_adjustment']['refine_focal_length']
        self.options.ba_refine_principal_point = config['bundle_adjustment']['refine_principal_point']
        self.options.ba_refine_extra_params = config['bundle_adjustment']['refine_extra_params']
        self.options.extract_colors = False
        if hasattr(self.options, 'fix_existing_images'):
            self.options.fix_existing_images = False
        else:
            self.options.fix_existing_frames = False
        self.options.ba_global_max_num_iterations = config['bundle_adjustment']['global_max_num_iterations']
        self.options.ba_global_max_refinements = config['bundle_adjustment']['global_max_refinements']
        self.options.multiple_models = False
        self.max_cost_change_px = config['bundle_adjustment']['max_cost_change_px']

        self.reconstruction_manager = pycolmap.ReconstructionManager()
        self.controller = pycolmap.IncrementalPipeline(
            self.options, str(self.images_dir), str(self.database_path), self.reconstruction_manager
        )

        self.mapper_options = self.controller.options.get_mapper()
        self.mapper_options.init_max_forward_motion = config['mapping']['init_max_forward_motion'] # 0.99
        self.mapper_options.init_min_tri_angle = config['mapping']['init_min_tri_angle'] # 0.1
        self.mapper_options.init_max_error = config['mapping']['init_max_error'] # 100.0
        self.mapper_options.abs_pose_max_error = config['mapping']['abs_pose_max_error'] # 2.0
        self.mapper_options.abs_pose_min_num_inliers = config['mapping']['abs_pose_min_num_inliers'] # 8
        self.mapper_options.abs_pose_min_inlier_ratio = config['mapping']['abs_pose_min_inlier_ratio'] # 0.01
        self.mapper_options.filter_max_reproj_error = config['mapping']['filter_max_reproj_error'] # 1.0
        self.mapper_options.init_min_num_inliers = config['mapping']['init_min_num_inliers'] # 30
        self.mapper_options.filter_min_tri_angle = config['mapping']['filter_min_tri_angle'] # 0.0
        self.mapper_options.local_ba_min_tri_angle = config['mapping']['local_ba_min_tri_angle'] # 0.0
        self.tri_options = self.controller.options.get_triangulation()
        self.tri_options.ignore_two_view_tracks = config['mapping']['tri_ignore_two_view_tracks'] # 0
        self.tri_options.merge_max_reproj_error = 1 # 1.0
        self.tri_options.complete_max_reproj_error = 1 # 1.0
        self.tri_options.min_angle = 0.01 # 0.01
        #self.tri_options.complete_max_transitivity = 30
        #print(self.tri_options);quit()
        #print(self.mapper_options)
        #quit()

        self.PnP_options = pycolmap.AbsolutePoseEstimationOptions()
        self.PnP_options.ransac.max_error = float(
            config['mapping'].get('pnp_ransac_max_error_px', 2.0))

        # COLMAP's pycolmap bindings expose LO-RANSAC for absolute pose, but
        # do not expose a selectable MAGSAC estimator.  Use OpenCV's
        # USAC_MAGSAC explicitly for the 2D-2D geometric verification stage;
        # the resulting inliers are then passed to the calibrated
        # triangulation and COLMAP LO-RANSAC PnP stages.
        self.geometric_verification_method = str(
            config['mapping'].get('geometric_verification_method', 'magsac')
        ).strip().lower()
        self.magsac_confidence = float(
            config['mapping'].get('magsac_confidence', 0.999))
        self.magsac_max_iterations = int(
            config['mapping'].get('magsac_max_iterations', 10000))
        self.magsac_available = hasattr(cv2, 'USAC_MAGSAC')
        if self.geometric_verification_method == 'magsac' and not self.magsac_available:
            raise RuntimeError(
                'Configured geometric_verification_method=magsac, but this '
                'OpenCV build has no USAC_MAGSAC capability')

        self.epipolar_verification_options = pycolmap.TwoViewGeometryOptions()
        self.epipolar_verification_options.ransac.max_error = float(
            config['mapping'].get('temporal_epipolar_max_error_px', 1.5))

        # Precompute image size arrays (avoid realloc in matcher)
        self.hw_np = np.array([self.height, self.width])

        # for LightGlue: constant scale/orientation LAF base (ones) cached per image len
        self._ones_cache: Dict[int, torch.Tensor] = {}

    def _update_memory_stats(self):
        """Update memory statistics in log_data"""
        if torch.cuda.is_available():
            self.log_data['memory_stats']['peak_gpu_memory_mb'] = max(
                self.log_data['memory_stats']['peak_gpu_memory_mb'],
                torch.cuda.max_memory_allocated() / (1024 * 1024)
            )
        
        self.log_data['memory_stats']['current_cached_features'] = len(self.keypoints)
        self.log_data['memory_stats']['current_cached_lafs'] = len(self.lafs_cache)

    def _update_performance_stats(self):
        """Update performance statistics in log_data"""
        frame_count = self.log_data['frame_count']
        keyframe_count = self.log_data['keyframe_count']
        
        if frame_count > 0:
            self.log_data['performance']['avg_frame_processing_time'] = (
                self.log_data['timing']['frame_processing_total'] / frame_count
            )
            self.log_data['performance']['keyframe_selection_ratio'] = keyframe_count / frame_count
            
        if self.log_data['total_features_extracted'] > 0 and frame_count > 0:
            self.log_data['performance']['avg_features_per_frame'] = (
                self.log_data['total_features_extracted'] / frame_count
            )

    @staticmethod
    def _as_keypoint_array(keypoints: Any) -> np.ndarray:
        if hasattr(keypoints, "detach"):
            keypoints = keypoints.detach().cpu().numpy()
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.ndim != 2 or keypoints.shape[1] < 2:
            raise ValueError("feature keypoints must have shape (N, >=2)")
        return keypoints[:, :2]

    def _feature_observation(
        self,
        frame_name: str,
        keypoint_index: int,
        point: np.ndarray,
    ) -> Dict[str, Any]:
        image_name = frame_name.split("/", 1)[-1]
        observation = {
            "image_name": image_name,
            "frame_index": self._feature_frame_indices[image_name],
            "keypoint_index": int(keypoint_index),
            "x": float(point[0]),
            "y": float(point[1]),
            "has_stereo_point": False,
            "stereo_x": 0.0,
            "stereo_y": 0.0,
            "stereo_point_camera": [0.0, 0.0, 0.0],
            "triangulation_angle_rad": 0.0,
        }
        stereo = self._stereo_observations.get(image_name, {}).get(
            int(keypoint_index))
        if stereo is not None:
            observation.update(
                {
                    "has_stereo_point": True,
                    "stereo_x": float(stereo["stereo_point"][0]),
                    "stereo_y": float(stereo["stereo_point"][1]),
                    "stereo_point_camera": np.asarray(
                        stereo["point_camera"], dtype=np.float64).tolist(),
                    "triangulation_angle_rad": float(
                        stereo["triangulation_angle_rad"]),
                }
            )
        return observation

    def _update_stereo_observations(
        self,
        image_name: str,
        matches: Dict[Tuple[str, str], np.ndarray],
    ) -> None:
        """Triangulate verified cam0/cam1 matches once for tracks and PnP."""
        if self.n_cameras <= 1:
            return
        cam0_name = f"cam0/{image_name}"
        cam1_name = f"cam1/{image_name}"
        pair = (cam0_name, cam1_name)
        reverse_pair = (cam1_name, cam0_name)
        if pair in matches:
            stereo_matches = np.asarray(matches[pair], dtype=np.int64).reshape(-1, 2)
        elif reverse_pair in matches:
            stereo_matches = np.asarray(
                matches[reverse_pair], dtype=np.int64).reshape(-1, 2)[:, ::-1]
        else:
            # Record an explicit empty result so PnP cannot fall back to
            # unfiltered database matches for this live frame.
            self._stereo_observations[image_name] = {}
            self._stereo_bearing_observations[image_name] = {}
            return

        current_log = self.log_data['current_frame']
        current_log['stereo_points_candidates'] += len(stereo_matches)

        cam0_keypoints = self._as_keypoint_array(self.keypoints[cam0_name])
        cam1_keypoints = self._as_keypoint_array(self.keypoints[cam1_name])
        T_cam0_world = rigid3d_to_matrix(self.cam0_pose_t0)
        T_cam1_world = rigid3d_to_matrix(self.cam1_pose_t0)
        center0 = np.linalg.inv(T_cam0_world)[:3, 3]
        center1 = np.linalg.inv(T_cam1_world)[:3, 3]
        # This used to call camera-model conversion and a 4x4 SVD once for
        # every match.  Batched NumPy linear algebra avoids hundreds of Python
        # / pybind crossings per stereo frame.
        valid_indices = (
            (stereo_matches[:, 0] >= 0)
            & (stereo_matches[:, 0] < len(cam0_keypoints))
            & (stereo_matches[:, 1] >= 0)
            & (stereo_matches[:, 1] < len(cam1_keypoints))
        )
        current_log['stereo_rejected_indices'] += int((~valid_indices).sum())
        stereo_matches = stereo_matches[valid_indices]
        observations: Dict[int, Dict[str, Any]] = {}
        if len(stereo_matches) == 0:
            self._stereo_observations[image_name] = observations
            self._stereo_bearing_observations[image_name] = {}
            return

        indices0 = stereo_matches[:, 0].astype(np.int64, copy=False)
        indices1 = stereo_matches[:, 1].astype(np.int64, copy=False)
        image_points0 = cam0_keypoints[indices0]
        image_points1 = cam1_keypoints[indices1]
        normalized0 = np.asarray(self.camera0.cam_from_img(image_points0), dtype=np.float64)
        normalized1 = np.asarray(self.camera1.cam_from_img(image_points1), dtype=np.float64)
        projection0 = T_cam0_world[:3, :]
        projection1 = T_cam1_world[:3, :]
        systems = np.stack((
            normalized0[:, 0, None] * projection0[2] - projection0[0],
            normalized0[:, 1, None] * projection0[2] - projection0[1],
            normalized1[:, 0, None] * projection1[2] - projection1[0],
            normalized1[:, 1, None] * projection1[2] - projection1[1],
        ), axis=1)
        finite_systems = np.isfinite(systems).all(axis=(1, 2))
        homogeneous = np.full((len(systems), 4), np.nan, dtype=np.float64)
        if np.any(finite_systems):
            _, _, vh = np.linalg.svd(systems[finite_systems])
            homogeneous[finite_systems] = vh[:, -1, :]
        valid_homogeneous = (
            np.isfinite(homogeneous).all(axis=1)
            & (np.abs(homogeneous[:, 3]) > 1.0e-12)
        )
        current_log['stereo_rejected_triangulation'] += int((~valid_homogeneous).sum())
        current_log['stereo_points_triangulated'] += int(valid_homogeneous.sum())
        with np.errstate(invalid='ignore', divide='ignore'):
            points_world = homogeneous[:, :3] / homogeneous[:, 3, None]
        points_cam0 = points_world @ T_cam0_world[:3, :3].T + T_cam0_world[:3, 3]
        points_cam1 = points_world @ T_cam1_world[:3, :3].T + T_cam1_world[:3, 3]
        cheirality = (
            valid_homogeneous
            & np.isfinite(points_world).all(axis=1)
            & np.isfinite(points_cam0).all(axis=1)
            & np.isfinite(points_cam1).all(axis=1)
            & (points_cam0[:, 2] > 0.0)
            & (points_cam1[:, 2] > 0.0)
        )
        current_log['stereo_rejected_cheirality'] += int((valid_homogeneous & ~cheirality).sum())
        depth = (
            cheirality
            & (points_cam0[:, 2] >= self.stereo_min_depth)
            & (points_cam1[:, 2] >= self.stereo_min_depth)
            & (points_cam0[:, 2] <= self.stereo_max_depth)
            & (points_cam1[:, 2] <= self.stereo_max_depth)
        )
        current_log['stereo_rejected_depth'] += int((cheirality & ~depth).sum())
        reprojection = np.zeros(len(depth), dtype=bool)
        # Reprojection validity is independent of the configured metric depth
        # interval.  Preserve far/low-parallax but geometrically valid stereo
        # bearings for the inverse-depth temporal solve below.
        if np.any(cheirality):
            projected0 = np.asarray(
                self.camera0.img_from_cam(points_cam0[cheirality]), dtype=np.float64)
            projected1 = np.asarray(
                self.camera1.img_from_cam(points_cam1[cheirality]), dtype=np.float64)
            reprojection[cheirality] = (
                np.isfinite(projected0).all(axis=1)
                & np.isfinite(projected1).all(axis=1)
                & (np.maximum(
                    np.linalg.norm(projected0 - image_points0[cheirality], axis=1),
                    np.linalg.norm(projected1 - image_points1[cheirality], axis=1),
                ) <= self.stereo_reprojection_max_error_px)
            )
        current_log['stereo_rejected_reprojection'] += int((depth & ~reprojection).sum())
        ray0 = points_world - center0
        ray1 = points_world - center1
        denominator = np.linalg.norm(ray0, axis=1) * np.linalg.norm(ray1, axis=1)
        angles = np.zeros(len(depth), dtype=np.float64)
        valid_angle = reprojection & (denominator > 1.0e-12)
        angles[valid_angle] = np.arccos(np.clip(
            np.einsum('ij,ij->i', ray0[valid_angle], ray1[valid_angle])
            / denominator[valid_angle], -1.0, 1.0))
        accepted = depth & valid_angle & (
            angles >= np.deg2rad(self.stereo_min_triangulation_angle_deg))
        current_log['stereo_rejected_angle'] += int((reprojection & ~accepted).sum())
        bearing_observations: Dict[int, Dict[str, Any]] = {}
        for row in np.flatnonzero(cheirality & reprojection):
            bearing_observations[int(indices0[row])] = {
                "image_point": image_points0[row].copy(),
                "stereo_keypoint_index": int(indices1[row]),
                "stereo_point": image_points1[row].copy(),
                "point_world": points_world[row].copy(),
                "point_camera": points_cam0[row].copy(),
                "triangulation_angle_rad": float(angles[row]),
                "metric_usable": bool(accepted[row]),
            }
        self._stereo_bearing_observations[image_name] = bearing_observations
        for row in np.flatnonzero(accepted):
            observations[int(indices0[row])] = {
                "image_point": image_points0[row].copy(),
                "stereo_keypoint_index": int(indices1[row]),
                "stereo_point": image_points1[row].copy(),
                "point_world": points_world[row].copy(),
                "point_camera": points_cam0[row].copy(),
                "triangulation_angle_rad": float(angles[row]),
            }
        current_log['stereo_points_valid'] += len(observations)
        self._stereo_observations[image_name] = observations

    @staticmethod
    def _pnp_correspondences(
        temporal_matches: np.ndarray,
        cached_stereo: Dict[int, Dict[str, Any]],
        current_keypoints: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return finite 2D--3D PnP correspondences for one image interval."""
        points2d, points3d, _, _ = VisualOdometry._pnp_correspondences_with_stereo(
            temporal_matches, cached_stereo, current_keypoints)
        return points2d, points3d

    @staticmethod
    def _pnp_correspondences_with_stereo(
        temporal_matches: np.ndarray,
        cached_stereo: Dict[int, Dict[str, Any]],
        current_keypoints: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return PnP correspondences plus the source stereo pixels.

        The extra pixel arrays are kept in exactly the same order as the PnP
        arrays.  This lets the stereo covariance estimator account for the
        uncertainty of each triangulated landmark without attempting to match
        rows by floating-point coordinates later.
        """
        temporal_matches = np.asarray(
            temporal_matches, dtype=np.int64).reshape(-1, 2)
        if temporal_matches.size == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )

        valid_indices = (
            (temporal_matches[:, 0] >= 0)
            & (temporal_matches[:, 1] >= 0)
            & (temporal_matches[:, 1] < len(current_keypoints))
        )
        temporal_matches = temporal_matches[valid_indices]
        if temporal_matches.size == 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
            )

        points3d = np.full((len(temporal_matches), 3), np.nan, dtype=np.float64)
        previous_left = np.full((len(temporal_matches), 2), np.nan, dtype=np.float64)
        previous_right = np.full((len(temporal_matches), 2), np.nan, dtype=np.float64)
        for row, cam0_index in enumerate(temporal_matches[:, 0]):
            observation = cached_stereo.get(int(cam0_index))
            if observation is not None:
                points3d[row] = observation["point_world"]
                previous_left[row] = observation["image_point"]
                previous_right[row] = observation["stereo_point"]
        valid = (
            np.isfinite(points3d).all(axis=1)
            & np.isfinite(previous_left).all(axis=1)
            & np.isfinite(previous_right).all(axis=1)
        )
        return (
            np.asarray(current_keypoints[temporal_matches[valid, 1], :2],
                       dtype=np.float64),
            points3d[valid],
            previous_left[valid],
            previous_right[valid],
        )

    def _structureless_correspondences(
        self,
        temporal_matches: np.ndarray,
        previous_stereo: Dict[int, Dict[str, Any]],
        current_stereo: Dict[int, Dict[str, Any]],
        current_keypoints: np.ndarray,
        previous_keypoints: np.ndarray | None = None,
        previous_bearings: Dict[int, Dict[str, Any]] | None = None,
        current_bearings: Dict[int, Dict[str, Any]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        """Assemble all calibrated temporal observations in match order.

        Geometry-valid stereo bearings are retained even when metric
        triangulation was rejected for shallow parallax or distant depth.  A
        finite ``points_world`` row is only an inverse-depth initializer; NaN
        rows start at infinity.  NaN right pixels represent a valid temporal
        previous-left/current-left constraint.
        """
        temporal_matches = np.asarray(
            temporal_matches, dtype=np.int64).reshape(-1, 2)
        if temporal_matches.size == 0:
            return None
        left_points = np.asarray(current_keypoints, dtype=np.float64)
        points_world = []
        previous_left = []
        previous_right = []
        current_left = []
        current_right = []
        for previous_index_raw, current_index_raw in temporal_matches:
            previous_index = int(previous_index_raw)
            current_index = int(current_index_raw)
            if (
                current_index < 0
                or current_index >= len(left_points)
            ):
                continue
            observation = (previous_bearings or {}).get(previous_index)
            if observation is None:
                observation = previous_stereo.get(previous_index)
            if observation is None and previous_keypoints is not None:
                if previous_index < 0 or previous_index >= len(previous_keypoints):
                    continue
                observation = {
                    "image_point": previous_keypoints[previous_index, :2],
                    "stereo_point": np.array([np.nan, np.nan]),
                    "point_world": np.array([np.nan, np.nan, np.nan]),
                }
            if observation is None:
                continue
            point = np.asarray(
                observation.get("point_world", [np.nan, np.nan, np.nan]),
                dtype=np.float64,
            )
            left = np.asarray(observation.get("image_point"), dtype=np.float64)
            right = np.asarray(
                observation.get("stereo_point", [np.nan, np.nan]),
                dtype=np.float64,
            )
            if left.shape != (2,) or not np.all(np.isfinite(left)):
                continue
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                point = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            if right.shape != (2,) or not np.all(np.isfinite(right)):
                right = np.array([np.nan, np.nan], dtype=np.float64)
            current_observation = (current_bearings or {}).get(current_index)
            if current_observation is None:
                current_observation = current_stereo.get(current_index)
            if current_observation is None:
                right_current = np.array([np.nan, np.nan], dtype=np.float64)
            else:
                right_current = np.asarray(
                    current_observation.get(
                        "stereo_point", [np.nan, np.nan]), dtype=np.float64)
                if (
                    right_current.shape != (2,)
                    or not np.all(np.isfinite(right_current))
                ):
                    right_current = np.array([np.nan, np.nan], dtype=np.float64)
            points_world.append(point)
            previous_left.append(left)
            previous_right.append(right)
            current_left.append(left_points[current_index, :2])
            current_right.append(right_current)
        if len(points_world) < 6:
            return None
        return (
            np.asarray(previous_left, dtype=np.float64),
            np.asarray(previous_right, dtype=np.float64),
            np.asarray(current_left, dtype=np.float64),
            np.asarray(current_right, dtype=np.float64),
            np.asarray(points_world, dtype=np.float64),
        )

    def _snapshot_feature_track(
        self,
        track_id: int,
        is_complete: bool,
    ) -> None:
        track = self._feature_tracks[track_id]
        observations = track["observations"]
        published_count = int(track.get("published_observations", 0))
        if (
            published_count == 0
            and len(observations) < self.feature_track_publish_min_length
        ):
            return
        # Publish the complete history exactly once when the track becomes
        # mature, then only observations appended since the previous epoch.
        # Re-sending every prefix made both serialization and downstream
        # association grow with total track age instead of current-frame work.
        appended = observations[published_count:]
        if not appended and not is_complete:
            return
        self.last_feature_track_updates.append(
            {
                "track_id": track_id,
                "camera_name": track["camera_name"],
                "is_complete": is_complete,
                "observations": [dict(item) for item in appended],
            }
        )
        track["published_observations"] = len(observations)
        self.log_data['total_feature_tracks_published'] += 1

    def _finish_feature_track(self, track_id: int) -> None:
        if track_id not in self._feature_tracks:
            return
        self._snapshot_feature_track(track_id, is_complete=True)
        self._feature_tracks.pop(track_id, None)

    def finish_feature_tracks(self) -> List[Dict[str, Any]]:
        """Close all active tracks and return their publishable snapshots."""
        self.last_feature_track_updates = []
        if not self.create_feature_tracks:
            return self.last_feature_track_updates
        for track_id in list(self._feature_tracks):
            self._finish_feature_track(track_id)
        self._active_feature_lookup.clear()
        return self.last_feature_track_updates

    def _update_feature_tracks(
        self,
        previous_frame_name: str,
        current_frame_name: str,
        matches: Any,
    ) -> None:
        """Chain geometrically verified native feature indices across keyframes."""
        if not self.create_feature_tracks:
            return
        previous_keypoints = self._as_keypoint_array(
            self.keypoints[previous_frame_name])
        current_keypoints = self._as_keypoint_array(
            self.keypoints[current_frame_name])
        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)

        previous_track_ids = set(self._active_feature_lookup.values())
        extended_track_ids: set[int] = set()
        current_lookup: Dict[Tuple[str, int], int] = {}
        used_current_indices: set[int] = set()

        for previous_index_raw, current_index_raw in matches:
            previous_index = int(previous_index_raw)
            current_index = int(current_index_raw)
            if (
                previous_index < 0
                or previous_index >= len(previous_keypoints)
                or current_index < 0
                or current_index >= len(current_keypoints)
                or current_index in used_current_indices
            ):
                continue

            track_id = self._active_feature_lookup.get(
                (previous_frame_name, previous_index))
            if track_id is not None and track_id in extended_track_ids:
                continue
            if track_id is None:
                track_id = self._next_feature_track_id
                self._next_feature_track_id += 1
                camera_name = previous_frame_name.split("/", 1)[0]
                self._feature_tracks[track_id] = {
                    "camera_name": camera_name,
                    "published_observations": 0,
                    "observations": [
                        self._feature_observation(
                            previous_frame_name,
                            previous_index,
                            previous_keypoints[previous_index],
                        )
                    ],
                }

            self._feature_tracks[track_id]["observations"].append(
                self._feature_observation(
                    current_frame_name,
                    current_index,
                    current_keypoints[current_index],
                )
            )
            extended_track_ids.add(track_id)
            used_current_indices.add(current_index)
            current_lookup[(current_frame_name, current_index)] = track_id
            self._snapshot_feature_track(track_id, is_complete=False)

        for track_id in previous_track_ids - extended_track_ids:
            self._finish_feature_track(track_id)
        self._active_feature_lookup = current_lookup
        self.log_data['current_frame']['feature_track_updates'] = len(
            self.last_feature_track_updates)

    def _store_match_debug(
        self,
        pairs: List[Tuple[str, str]],
        matches: Dict[Tuple[str, str], np.ndarray],
    ) -> None:
        self.last_feature_match_debug = []
        for pair in pairs:
            img1, img2 = pair
            if img1 not in self.keypoints or img2 not in self.keypoints:
                continue
            idxs = matches.get(pair)
            if idxs is None:
                continue

            kps1 = self.keypoints[img1]
            kps2 = self.keypoints[img2]
            if hasattr(kps1, "detach"):
                kps1 = kps1.detach().cpu().numpy()
            else:
                kps1 = np.asarray(kps1)
            if hasattr(kps2, "detach"):
                kps2 = kps2.detach().cpu().numpy()
            else:
                kps2 = np.asarray(kps2)
            if hasattr(idxs, "detach"):
                idxs = idxs.detach().cpu().numpy()
            else:
                idxs = np.asarray(idxs)
            idxs = np.asarray(idxs, dtype=np.int64).reshape(-1, 2)
            kps1 = np.asarray(kps1, dtype=np.float32).reshape(-1, 2)
            kps2 = np.asarray(kps2, dtype=np.float32).reshape(-1, 2)

            self.last_feature_match_debug.append(
                {
                    "pair": pair,
                    "keypoints1": kps1[:, :2],
                    "keypoints2": kps2[:, :2],
                    "matches": idxs[:, :2],
                }
            )

    def _log_timing(self, operation: str, duration: float):
        """Log timing for specific operations"""
        if operation in self.log_data['timing']:
            self.log_data['timing'][operation] += duration

        # Also log to the current frame.  The detailed fields are useful for
        # explaining the otherwise broad ``visual_odometry`` wrapper stage.
        if operation == 'feature_extraction':
            self.log_data['current_frame']['feature_extraction_time'] += duration
        elif operation == 'feature_matching':
            self.log_data['current_frame']['matching_time'] += duration
        elif operation == 'reconstruction':
            self.log_data['current_frame']['reconstruction_time'] = duration
        elif operation in self.log_data['current_frame']:
            self.log_data['current_frame'][operation] += duration
        elif operation == 'database_operations_total':
            self.log_data['current_frame']['database_time'] += duration

    def compute_backend_summary(self) -> str:
        """Describe the compute devices selected by the active VO pipeline."""
        solver_provenance = (
            f"solver=anchored_inverse_depth_stereo_temporal; module={__file__}; "
            f"structureless={'enabled' if self.use_min_solver else 'disabled'}"
        )
        feature_device = (
            "CPU (OpenCV)"
            if self.local_features.feature_name == "orb"
            else str(self.local_features.device)
        )
        matcher_device = str(self.device) if self.matcher in ("lightglue", "loma") else "CPU"
        if self.device.type == "cuda":
            gpu_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(gpu_index)
            cuda_version = torch.version.cuda or "unknown"
            return (
                "COLMAP-SLAM compute backend: GPU "
                f"({gpu_name}, CUDA {cuda_version}); feature_extractor={feature_device}, "
                f"matcher={matcher_device}, geometric_solver=CPU (pycolmap); "
                f"{solver_provenance}."
                f" 2D-2D verification={self.geometric_verification_method.upper()}"
                f" ({'available' if self.magsac_available else 'unavailable'}); "
                "calibrated essential geometry; absolute pose=COLMAP LO-RANSAC."
            )
        return (
            "COLMAP-SLAM compute backend: CPU; PyTorch CUDA is unavailable, "
            f"feature_extractor={feature_device}, matcher={matcher_device}, "
            "geometric_solver=CPU (pycolmap); "
            f"{solver_provenance}; "
            f"2D-2D verification={self.geometric_verification_method.upper()} "
            f"({'available' if self.magsac_available else 'unavailable'}); "
            "calibrated essential geometry; absolute pose=COLMAP LO-RANSAC."
        )

    def warmup_loma(self) -> float:
        """Warm LoMa on the same worker thread that executes image frames."""
        if (
            self.local_features.feature_name == "loma"
            and getattr(self.local_features, "loma_compile_enabled", False)
        ):
            self.loma_warmup_seconds = self.local_features.warmup_loma()
        return self.loma_warmup_seconds

    def stereo_filter_summary(self) -> str:
        """Describe the fixed-rig checks protecting stereo triangulation."""
        if self.n_cameras <= 1 or not self.rig_match_rule:
            return "COLMAP-SLAM calibrated stereo filtering: disabled (mono input)."
        return (
            "COLMAP-SLAM calibrated stereo filtering: "
            f"epipolar<={self.stereo_epipolar_max_error_px:g}px, "
            f"reprojection<={self.stereo_reprojection_max_error_px:g}px, "
            f"angle>={self.stereo_min_triangulation_angle_deg:g}deg, "
            f"depth=[{self.stereo_min_depth:g}, "
            f"{self.stereo_max_depth:g}]m."
        )

    def stereo_points_for_frame(self, image_name: str) -> list[dict[str, Any]]:
        """Return accepted stereo points in the current cam0 optical frame."""
        return [
            {
                "image_point": np.asarray(
                    observation["image_point"], dtype=np.float64).copy(),
                "point_camera": np.asarray(
                    observation["point_camera"], dtype=np.float64).copy(),
            }
            for observation in self._stereo_observations.get(image_name, {}).values()
        ]

    # -------------------- utils --------------------
    def _laf_from_kps_cached(self, name: str, kps: torch.Tensor) -> torch.Tensor:
        """Cache LAF tensors per image to avoid recomputation in matching."""
        if name in self.lafs_cache:
            return self.lafs_cache[name]
        n = kps.shape[0]
        if n not in self._ones_cache:
            self._ones_cache[n] = torch.ones(1, n, 1, 1, device=self.device)
        lafs = KF.laf_from_center_scale_ori(kps[None], self._ones_cache[n])
        self.lafs_cache[name] = lafs
        return lafs

    def rig_match_pairs(self, img_name: str) -> List[Tuple[str, str]]:
        return [(f"{c1}/{img_name}", f"{c2}/{img_name}") for c1, c2 in self.rig_match_rule]

    def _known_rig_pair(
        self,
        image1: str,
        image2: str,
    ) -> Tuple[pycolmap.Camera, pycolmap.Rigid3d,
               pycolmap.Camera, pycolmap.Rigid3d] | None:
        """Return calibrated camera data for a synchronized cam0/cam1 pair."""
        camera1_name, separator1, timestamp1 = image1.partition("/")
        camera2_name, separator2, timestamp2 = image2.partition("/")
        if not separator1 or not separator2 or timestamp1 != timestamp2:
            return None
        configured_pairs = {
            (str(first), str(second))
            for first, second in self.rig_match_rule
        }
        if (
            (camera1_name, camera2_name) not in configured_pairs
            and (camera2_name, camera1_name) not in configured_pairs
        ):
            return None

        cameras = {"cam0": self.camera0}
        poses = {"cam0": self.cam0_pose_t0}
        if self.n_cameras > 1:
            cameras["cam1"] = self.camera1
            poses["cam1"] = self.cam1_pose_t0
        if camera1_name not in cameras or camera2_name not in cameras:
            return None
        return (
            cameras[camera1_name],
            poses[camera1_name],
            cameras[camera2_name],
            poses[camera2_name],
        )

    def _filter_known_rig_matches(
        self,
        image1: str,
        image2: str,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        matches: np.ndarray,
    ) -> np.ndarray | None:
        """Filter a stereo pair with the fixed calibrated essential matrix.

        Temporal pairs deliberately return ``None`` and continue through
        COLMAP's free two-view estimation because their relative pose is the
        unknown that odometry must solve.
        """
        rig_pair = self._known_rig_pair(image1, image2)
        if rig_pair is None:
            return None

        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
        current_log = self.log_data['current_frame']
        current_log['stereo_matches_raw'] += len(matches)
        if len(matches) == 0:
            return matches

        camera1, cam1_from_rig, camera2, cam2_from_rig = rig_pair
        T_cam2_cam1 = (
            rigid3d_to_matrix(cam2_from_rig)
            @ np.linalg.inv(rigid3d_to_matrix(cam1_from_rig))
        )
        rotation = T_cam2_cam1[:3, :3]
        translation = T_cam2_cam1[:3, 3]
        tx = np.array(
            [
                [0.0, -translation[2], translation[1]],
                [translation[2], 0.0, -translation[0]],
                [-translation[1], translation[0], 0.0],
            ],
            dtype=np.float64,
        )
        essential = tx @ rotation

        pixels1 = np.asarray(keypoints1, dtype=np.float64)[matches[:, 0], :2]
        pixels2 = np.asarray(keypoints2, dtype=np.float64)[matches[:, 1], :2]
        normalized1 = np.asarray(
            camera1.cam_from_img(pixels1), dtype=np.float64).reshape(-1, 2)
        normalized2 = np.asarray(
            camera2.cam_from_img(pixels2), dtype=np.float64).reshape(-1, 2)
        homogeneous1 = np.column_stack(
            (normalized1, np.ones(len(normalized1), dtype=np.float64)))
        homogeneous2 = np.column_stack(
            (normalized2, np.ones(len(normalized2), dtype=np.float64)))

        Ex1 = (essential @ homogeneous1.T).T
        Etx2 = (essential.T @ homogeneous2.T).T
        numerator = np.sum(homogeneous2 * Ex1, axis=1)
        denominator = np.sqrt(
            Ex1[:, 0] ** 2
            + Ex1[:, 1] ** 2
            + Etx2[:, 0] ** 2
            + Etx2[:, 1] ** 2
        )
        sampson_error = np.full(len(matches), np.inf, dtype=np.float64)
        valid_denominator = denominator > np.finfo(np.float64).eps
        sampson_error[valid_denominator] = np.abs(
            numerator[valid_denominator]) / denominator[valid_denominator]
        normalized_threshold = max(
            float(camera1.cam_from_img_threshold(
                self.stereo_epipolar_max_error_px)),
            float(camera2.cam_from_img_threshold(
                self.stereo_epipolar_max_error_px)),
        )
        keep = np.isfinite(sampson_error) & (
            sampson_error <= normalized_threshold)
        filtered = matches[keep]
        current_log['stereo_matches_epipolar'] += len(filtered)
        return filtered

    def check_stereo(self, img1: pycolmap.Image, img2: pycolmap.Image) -> None:
        if img1.name.split("/")[1] != img2.name.split("/")[1]:
            raise RuntimeError("Stereo timestamps mismatch")

    def _magsac_filter_matches(
        self,
        keypoints1: np.ndarray,
        keypoints2: np.ndarray,
        matches: np.ndarray,
        maximum_error_px: float,
        camera1: pycolmap.Camera | None = None,
        camera2: pycolmap.Camera | None = None,
    ) -> np.ndarray:
        """Return calibrated 2D-2D matches surviving OpenCV USAC-MAGSAC.

        COLMAP cameras in this package may be fisheye models.  Therefore a
        free fundamental matrix on raw image pixels is not valid: epipolar
        curves in a distorted image are not straight lines.  We undistort to
        calibrated normalized bearings and estimate an essential matrix with
        USAC-MAGSAC instead.  The returned indices remain in the original
        keypoint arrays and the later PnP/structureless solver still performs
        the final calibrated pose estimate.
        """
        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
        if len(matches) < 8:
            return matches
        if not self.magsac_available:
            raise RuntimeError(
                "MAGSAC requested but this OpenCV build has no USAC_MAGSAC")
        points1 = np.asarray(keypoints1, dtype=np.float64)[matches[:, 0], :2]
        points2 = np.asarray(keypoints2, dtype=np.float64)[matches[:, 1], :2]
        if camera1 is None:
            camera1 = self.camera0
        if camera2 is None:
            camera2 = self.camera0
        try:
            normalized1 = np.asarray(
                camera1.cam_from_img(points1), dtype=np.float64).reshape(-1, 2)
            normalized2 = np.asarray(
                camera2.cam_from_img(points2), dtype=np.float64).reshape(-1, 2)
            normalized_threshold = max(
                float(camera1.cam_from_img_threshold(maximum_error_px)),
                float(camera2.cam_from_img_threshold(maximum_error_px)),
                1.0e-6,
            )
        except (RuntimeError, TypeError, ValueError, AttributeError):
            return np.empty((0, 2), dtype=np.int64)
        if (
            not np.all(np.isfinite(normalized1))
            or not np.all(np.isfinite(normalized2))
        ):
            return np.empty((0, 2), dtype=np.int64)
        try:
            _, mask = cv2.findEssentialMat(
                normalized1,
                normalized2,
                cameraMatrix=np.eye(3, dtype=np.float64),
                method=cv2.USAC_MAGSAC,
                # OpenCV requires a strict probability in (0, 1); keeping a
                # tiny margin also makes hand-written configs with 1.0 safe.
                prob=min(max(self.magsac_confidence, 1.0e-6), 1.0 - 1.0e-9),
                threshold=normalized_threshold,
                maxIters=max(self.magsac_max_iterations, 100),
            )
        except cv2.error as error:
            # A free essential matrix is undefined for pure rotation or a
            # stationary interval.  Preserve the existing raw-match recovery
            # path so PnP remains responsible for those valid cases.
            current_log = self.log_data.get('current_frame', {})
            current_log['magsac_degenerate'] = True
            current_log['magsac_failure_reason'] = str(error)
            return np.empty((0, 2), dtype=np.int64)
        if mask is None:
            return np.empty((0, 2), dtype=np.int64)
        mask = np.asarray(mask).reshape(-1).astype(bool, copy=False)
        if len(mask) != len(matches):
            return np.empty((0, 2), dtype=np.int64)
        filtered = matches[mask]
        self.log_data['current_frame']['magsac_inliers'] = int(len(filtered))
        return filtered

    def _estimate_calibrated_essential_rotation(
        self,
        keypoints_previous: np.ndarray,
        keypoints_current: np.ndarray,
        matches: np.ndarray,
    ) -> tuple[np.ndarray, int] | None:
        """Estimate current-from-previous rotation from calibrated bearings.

        Essential translation has an arbitrary scale, so it is not used as a
        metric measurement.  Its rotation is a useful independent initializer
        for the metric stereo solve, particularly when the available stereo
        depths are weak or have been rejected by the depth limits.
        """
        matches = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
        if len(matches) < 8:
            return None
        try:
            pixels_previous = np.asarray(
                keypoints_previous, dtype=np.float64)[matches[:, 0], :2]
            pixels_current = np.asarray(
                keypoints_current, dtype=np.float64)[matches[:, 1], :2]
            bearings_previous = np.asarray(
                self.camera0.cam_from_img(pixels_previous), dtype=np.float64
            ).reshape(-1, 2)
            bearings_current = np.asarray(
                self.camera0.cam_from_img(pixels_current), dtype=np.float64
            ).reshape(-1, 2)
            threshold = max(
                float(self.camera0.cam_from_img_threshold(
                    self.epipolar_verification_options.ransac.max_error)),
                1.0e-6,
            )
            essential, mask = cv2.findEssentialMat(
                bearings_previous,
                bearings_current,
                cameraMatrix=np.eye(3, dtype=np.float64),
                method=cv2.USAC_MAGSAC,
                prob=min(max(self.magsac_confidence, 1.0e-6), 1.0 - 1.0e-9),
                threshold=threshold,
                maxIters=max(self.magsac_max_iterations, 100),
            )
            if essential is None or mask is None:
                return None
            essential = np.asarray(essential, dtype=np.float64)
            if essential.ndim == 2 and essential.shape[1] == 3:
                if essential.shape[0] < 3:
                    return None
                essential = essential[:3, :3]
            if essential.shape != (3, 3):
                return None
            count, rotation, _, pose_mask = cv2.recoverPose(
                essential,
                bearings_previous,
                bearings_current,
                cameraMatrix=np.eye(3, dtype=np.float64),
                mask=np.asarray(mask, dtype=np.uint8),
            )
            count = int(count)
            rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
            if (
                count < 8
                or not np.all(np.isfinite(rotation))
                or np.linalg.det(rotation) <= 0.0
            ):
                return None
            return rotation, count
        except (cv2.error, RuntimeError, TypeError, ValueError,
                np.linalg.LinAlgError):
            return None

    def make_match_plot(
        self, img: np.ndarray, img2: np.ndarray, mpts1: np.ndarray, mpts2: np.ndarray, method: str = "flow",
    ) -> np.ndarray:
        # DEBUG ONLY (slow). Prefer disabling in production.
        if method == "flow":
            match_img = img.copy()
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(match_img, p1, p2, (0, 255, 0), 3, lineType=16)
                cv2.circle(match_img, p2, 1, (0, 0, 255), -1, lineType=16)
        else:  # "pair"
            img1_w = img.shape[1]
            match_img = np.concatenate((img, img2), axis=1)
            for pt1, pt2 in zip(mpts1, mpts2):
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0] + img1_w)), int(round(pt2[1])))
                cv2.line(match_img, p1, p2, (0, 255, 0), lineType=16)
                cv2.circle(match_img, p1, 1, (0, 0, 255), 3, lineType=16)
                cv2.circle(match_img, p2, 1, (0, 0, 255), 3, lineType=16)
        return match_img

    @torch.inference_mode()
    def match_features(self, keypoints: Dict[str, torch.Tensor], descriptors: Dict[str, torch.Tensor], pairs: List[Tuple[str, str]]):
        matches = {}
        total_matches = 0
        self._raw_feature_matches = {}

        loma_matches = {}
        if self.matcher == "loma" and pairs:
            model_start = time.perf_counter()
            loma_pairs = [
                (
                    keypoints[img1], keypoints[img2],
                    descriptors[img1], descriptors[img2],
                ) for img1, img2 in pairs]
            loma_results = self._match_loma_batch(loma_pairs)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._log_timing(
                'matching_model_time', time.perf_counter() - model_start)
            loma_matches = {
                pair: result for pair, result in zip(pairs, loma_results)}

        for img1, img2 in pairs:
            self.performance_stage = f'feature_matching:{img1}<->{img2}'
            kps1 = keypoints[img1]
            kps2 = keypoints[img2]
            descs1 = descriptors[img1]
            descs2 = descriptors[img2]

            # ensure on device once
            if kps1.device != self.device:
                kps1 = kps1.to(self.device, non_blocking=True)
                keypoints[img1] = kps1
            if kps2.device != self.device:
                kps2 = kps2.to(self.device, non_blocking=True)
                keypoints[img2] = kps2
            if descs1.device != self.device:
                descs1 = descs1.to(self.device, non_blocking=True)
                descriptors[img1] = descs1
            if descs2.device != self.device:
                descs2 = descs2.to(self.device, non_blocking=True)
                descriptors[img2] = descs2

            if self.matcher == "lightglue":
                lafs1 = self._laf_from_kps_cached(img1, kps1)
                lafs2 = self._laf_from_kps_cached(img2, kps2)
                model_start = time.perf_counter()
                dists, idxs = self.lg_matcher(descs1, descs2, lafs1, lafs2, hw1=self.hw_np, hw2=self.hw_np)
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                self._log_timing(
                    'matching_model_time', time.perf_counter() - model_start)
            elif self.matcher == "loma":
                idxs = loma_matches[(img1, img2)]
            elif self.matcher == "smnn":
                model_start = time.perf_counter()
                dists, idxs = match_smnn(descs1, descs2, th=self.ratio_threshold)
                self._log_timing(
                    'matching_model_time', time.perf_counter() - model_start)

            geometry_start = time.perf_counter()
            keypoints1_np = kps1[:, :2].detach().cpu().numpy()
            keypoints2_np = kps2[:, :2].detach().cpu().numpy()
            candidate_matches = (
                idxs[:, :2].detach().cpu().numpy().astype(np.int64, copy=False)
            )
            # Retain mutually matched candidates before geometric
            # verification. The minimal solver uses them only if MAGSAC
            # removes the stereo-supported correspondences required by PnP.
            self._raw_feature_matches[(img1, img2)] = candidate_matches.copy()
            calibrated_matches = self._filter_known_rig_matches(
                img1,
                img2,
                keypoints1_np,
                keypoints2_np,
                candidate_matches,
            )
            if calibrated_matches is not None:
                # A synchronized rig pair has a known pose. Estimating a new
                # free two-view geometry here can accept matches that are
                # impossible under the calibrated baseline used downstream.
                idxs = calibrated_matches
            else:
                if self.geometric_verification_method == 'magsac':
                    idxs = self._magsac_filter_matches(
                        keypoints1_np,
                        keypoints2_np,
                        candidate_matches,
                        self.epipolar_verification_options.ransac.max_error,
                        camera1=self.camera0
                        if img1.split("/")[0] == "cam0" else self.camera1,
                        camera2=self.camera0
                        if img2.split("/")[0] == "cam0" else self.camera1,
                    )
                else:
                    two_view_geom = pycolmap.estimate_two_view_geometry(
                        pycolmap.Camera(self.camera_config[img1.split("/")[0]]),
                        keypoints1_np,
                        pycolmap.Camera(self.camera_config[img2.split("/")[0]]),
                        keypoints2_np,
                        candidate_matches,
                        self.epipolar_verification_options,
                    )
                    idxs = two_view_geom.inlier_matches
            self._log_timing(
                'matching_geometry_time', time.perf_counter() - geometry_start)

            matches[(img1, img2)] = idxs  # keep on device
            total_matches += len(idxs)
        
        # Update logging stats
        self.log_data['total_matches_computed'] += total_matches
        if len(pairs) > 0:
            self.log_data['performance']['avg_matches_per_pair'] = total_matches / len(pairs)
        
        ## Clear GPU cache after matching
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()

        return matches

    def _match_loma(self, kps1: torch.Tensor, kps2: torch.Tensor,
                    descs1: torch.Tensor, descs2: torch.Tensor) -> torch.Tensor:
        """Match one pair with the batched LoMa implementation."""
        return self._match_loma_batch([(kps1, kps2, descs1, descs2)])[0]

    def _match_loma_batch(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor,
                          torch.Tensor, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Match compatible pairs in one LoMa transformer invocation."""
        from loma.loma import filter_matches

        # Cached keypoints have already been converted from the resized/padded
        # detector image to native camera pixels. Normalize them using the
        # native dimensions, otherwise the learned positional encoding sees
        # coordinates outside [-1, 1] (e.g. x=1440 / w=728 on GrandTour).
        h = float(self.height)
        w = float(self.width)

        def to_normalized(kps: torch.Tensor) -> torch.Tensor:
            xy = kps[:, :2].float()
            return torch.stack(
                (2 * xy[:, 0] / w - 1, 2 * xy[:, 1] / h - 1), dim=-1)

        if not pairs:
            return []

        # Detector top-k normally makes all pairs shape-compatible. Keep a
        # safe fallback for a backend returning variable-length detections.
        same_shape = all(
            kps1.shape == pairs[0][0].shape and
            kps2.shape == pairs[0][1].shape and
            descs1.shape == pairs[0][2].shape and
            descs2.shape == pairs[0][3].shape
            for kps1, kps2, descs1, descs2 in pairs)
        if not same_shape and len(pairs) > 1:
            return [self._match_loma_batch([pair])[0] for pair in pairs]

        k1 = torch.stack(
            [to_normalized(kps1) for kps1, _, _, _ in pairs], dim=0
        ).to(self.device, non_blocking=True)
        k2 = torch.stack(
            [to_normalized(kps2) for _, kps2, _, _ in pairs], dim=0
        ).to(self.device, non_blocking=True)
        # LoMa's DeDoDe descriptors are passed in their native scale. The
        # learned matcher was trained with these descriptors unnormalized.
        d1 = torch.stack(
            [descs1 for _, _, descs1, _ in pairs], dim=0
        ).to(self.device, non_blocking=True)
        d2 = torch.stack(
            [descs2 for _, _, _, descs2 in pairs], dim=0
        ).to(self.device, non_blocking=True)

        # ``self.loma_matcher`` is torch.compile()-wrapped. Clone its output
        # before filtering so the next compiled invocation cannot overwrite a
        # CUDA-graph-owned score tensor.
        scores = self.loma_matcher(k1, k2, d1, d2)["scores"].clone()
        m0, _, _, _ = filter_matches(scores, self.loma_filter_threshold)
        results = []
        for batch_index in range(len(pairs)):
            valid = m0[batch_index] > -1
            idx1 = torch.where(valid)[0]
            idx2 = m0[batch_index][valid]
            results.append(torch.stack((idx1, idx2), dim=-1))
        return results

    @torch.inference_mode()
    def match_distance(self, keypoints: Dict[str, torch.Tensor], matches: Dict[Tuple[str, str], torch.Tensor], keyframe_name: str, frame_name: str) -> float:
        # compute on GPU to avoid transfers
        idxs = matches[(keyframe_name, frame_name)]
        mpts1 = keypoints[keyframe_name][idxs[:, 0]]
        mpts2 = keypoints[frame_name][idxs[:, 1]]
        # Torch median of L2 distances
        match_dist = torch.linalg.norm(mpts1 - mpts2, dim=1)
        median_match_dist = torch.median(match_dist).item()
        
        # Log the median match distance
        self.log_data['current_frame']['median_match_distance'] = median_match_dist
        
        return float(median_match_dist)

    def write_keypoints_to_db(self, db: COLMAPDatabase, keyframe_name: str, image_id: int, camera_id: int, keypoints: Dict[str, torch.Tensor]) -> None:
        db_start = time.time()
        
        image = _new_database_image(keyframe_name, camera_id, image_id)
        db.write_image(image, use_image_id=True)
        db.write_keypoints(image_id=image_id, keypoints=keypoints[keyframe_name].detach().cpu().numpy())
        self.db_dirty = True
        
        db_time = time.time() - db_start
        self._log_timing('database_operations_total', db_time)

    def _extract_and_store(self, name: str, img: np.ndarray) -> int:
        """Extract and store features, return number of features extracted"""
        self.performance_stage = f'feature_extraction:{name}'
        extraction_start = time.time()
        
        ## Clear GPU cache before feature extraction
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        
        model_start = time.perf_counter()
        new_kps, new_descs, reading_images_status = self.local_features.extract(name, img)
        if (self.local_features.feature_name == "loma" and
                self.device.type == "cuda"):
            torch.cuda.synchronize(self.device)
        model_duration = time.perf_counter() - model_start
        self._log_timing('feature_model_time', model_duration)
        if not reading_images_status:
        #    # Clear cache even on failure
        #    if torch.cuda.is_available():
        #        torch.cuda.empty_cache()
            return 0, reading_images_status
        
        # Keep tensors on device to avoid later copies
        transfer_start = time.perf_counter()
        for k, v in new_kps.items():
            new_kps[k] = v.to(self.device, non_blocking=True)
        for k, v in new_descs.items():
            new_descs[k] = v.to(self.device, non_blocking=True)
        self._log_timing(
            'feature_tensor_transfer_time', time.perf_counter() - transfer_start)
        
        self.keypoints.update(new_kps)
        self.descriptors.update(new_descs)
        
        # invalidate LAF cache if this name exists
        for k in new_kps.keys():
            self.lafs_cache.pop(k, None)
        
        extraction_time = time.time() - extraction_start
        self._log_timing('feature_extraction_total', extraction_time)
        self._log_timing('feature_extraction', extraction_time)
        
        # Count features and update stats
        num_features = sum(len(kps) for kps in new_kps.values())
        self.log_data['total_features_extracted'] += num_features
        self.log_data['current_frame']['num_features'] = num_features
        
        # Clear GPU cache after processing
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
        
        return num_features, reading_images_status

    def _extract_and_store_batch(
        self,
        items: List[Tuple[str, np.ndarray]],
    ) -> Tuple[Dict[str, int], Dict[str, bool]]:
        """Extract and cache a synchronized image batch.

        LoMa uses one detector/descriptor invocation for the whole rig. Other
        feature backends retain their existing per-image behavior through
        ``LocalFeatures.extract_batch``.
        """
        if not items:
            return {}, {}
        self.performance_stage = 'feature_extraction:batch'
        extraction_start = time.time()
        model_start = time.perf_counter()
        new_kps, new_descs, statuses = self.local_features.extract_batch(items)
        if (self.local_features.feature_name == "loma" and
                self.device.type == "cuda"):
            torch.cuda.synchronize(self.device)
        self._log_timing(
            'feature_model_time', time.perf_counter() - model_start)
        transfer_start = time.perf_counter()
        for key, value in new_kps.items():
            new_kps[key] = value.to(self.device, non_blocking=True)
        for key, value in new_descs.items():
            new_descs[key] = value.to(self.device, non_blocking=True)
        self._log_timing(
            'feature_tensor_transfer_time', time.perf_counter() - transfer_start)
        self.keypoints.update(new_kps)
        self.descriptors.update(new_descs)
        for key in new_kps:
            self.lafs_cache.pop(key, None)

        self._log_timing(
            'feature_extraction', time.time() - extraction_start)
        counts = {name: int(len(new_kps.get(name, ()))) for name, _ in items}
        self.log_data['total_features_extracted'] += sum(counts.values())
        self.log_data['current_frame']['num_features'] = sum(counts.values())
        return counts, statuses

    def reinitialize(self, reset_odometry: bool = False) -> None:
        """Reset local reconstruction state.

        The published odometry is an accumulated stream, so normal recovery
        should keep the accumulated pose and only reset COLMAP's local state.
        """
        self.current_status = 'initializing'
        if self.log:
            print('[CSLAM:] Reinitializing..')
        
        # Update reinitalization count
        self.log_data['reinitializations'] += 1
        self.N_reinit += 1

        if self.create_feature_tracks:
            for track_id in list(self._feature_tracks):
                self._finish_feature_track(track_id)
        self._active_feature_lookup.clear()
        self._feature_frame_indices.clear()
        self._stereo_observations.clear()
        self._stereo_bearing_observations.clear()
        self._temporal_matches.clear()
        self._temporal_raw_matches.clear()
        self._raw_feature_matches.clear()
        
        self.baseline_old = 0.0
        self.keyframe_count = 1
        self.keyframe_id = 1
        self.frame_t0 = None
        self.frame_t1 = None
        if reset_odometry:
            self.t_cumulative = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            self.q_cumulative = Quaternion(np.array([1.0, 0.0, 0.0, 0.0]))

        # Reset the legacy database only when that optional output is active.
        if self.db is not None:
            self.db.close()
            self._remove_database_files()
            self.db = Database(str(self.database_path))
        self.db_dirty = False

        # reset outputs
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.images.clear()
        self.keyframes_names.clear()
        self.keyframes_ids.clear()
        self.keyframes_master_ids.clear()
        self.keypoints.clear()
        self.descriptors.clear()
        self.lafs_cache.clear()
        self._ones_cache.clear()

        self.reconstruction_manager = pycolmap.ReconstructionManager()
        self.controller = pycolmap.IncrementalPipeline(
            self.options, str(self.images_dir), str(self.database_path), self.reconstruction_manager
        )

    def _remove_database_files(self) -> None:
        """Remove a database and SQLite sidecars before a clean reconstruction."""
        for suffix in ("", "-wal", "-shm"):
            Path(f"{self.database_path}{suffix}").unlink(missing_ok=True)

    def _run_min_solver_fallback(
        self,
        image: str,
        pose_prior: Dict[str, Any] | None = None,
    ) -> List[List]:
        if self.n_cameras <= 1 or not self.use_min_solver:
            return [[image, None, None, None, None, None]]
        if self.frame_t0 is None or self.frame_t1 is None:
            return [[image, None, None, None, None, None]]

        result = self.min_solver(
            image,
            self.keyframe_id,
            self.frame_t0,
            self.frame_t1,
            pose_prior=pose_prior,
        )
        if result[0][1] is not None:
            self.frame_t0 = copy.deepcopy(self.frame_t1)
        return result

    def _maybe_load_db(self):
        if self.db_dirty:
            self.performance_stage = 'database_load'
            db_start = time.time()
            self.controller.load_database()
            self.db_dirty = False
            db_time = time.time() - db_start
            self._log_timing('database_operations_total', db_time)

    def _pose_candidate_quality(
        self,
        pose: Dict[str, Any] | None,
        points2D: np.ndarray,
        points3D: np.ndarray,
        maximum_error_px: float,
    ) -> Tuple[int, float, np.ndarray]:
        if pose is None or pose.get('cam_from_world') is None:
            return 0, float('inf'), np.zeros(len(points2D), dtype=bool)
        transform = rigid3d_to_matrix(pose['cam_from_world'])
        points_camera = (
            points3D @ transform[:3, :3].T + transform[:3, 3]
        )
        valid = np.isfinite(points_camera).all(axis=1) & (points_camera[:, 2] > 0.0)
        projected = np.full((len(points3D), 2), np.nan, dtype=np.float64)
        if np.any(valid):
            projected[valid] = np.asarray(
                self.camera0.img_from_cam(points_camera[valid]),
                dtype=np.float64,
            ).reshape(-1, 2)
        errors = np.linalg.norm(projected - points2D, axis=1)
        inliers = valid & np.isfinite(errors) & (errors <= maximum_error_px)
        median_error = (
            float(np.median(errors[inliers])) if np.any(inliers) else float('inf')
        )
        return int(np.count_nonzero(inliers)), median_error, inliers

    @staticmethod
    def _valid_pnp_covariance(covariance: Any) -> np.ndarray | None:
        """Return a finite positive-definite [rotation, translation] covariance."""
        if covariance is None:
            return None
        matrix = np.asarray(covariance, dtype=np.float64)
        if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
            return None
        matrix = 0.5 * (matrix + matrix.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(eigenvalues)) or eigenvalues[-1] <= 0.0:
            return None
        # Small negative eigenvalues occur from round-off in covariance
        # transport. Repair those, but reject a genuinely indefinite matrix.
        if eigenvalues[0] < -1.0e-8 * eigenvalues[-1]:
            return None
        floor = max(1.0e-15, eigenvalues[-1] * 1.0e-12)
        eigenvalues = np.maximum(eigenvalues, floor)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _reprojection_pnp_covariance(
        self,
        pose: Dict[str, Any],
        points2D: np.ndarray,
        points3D: np.ndarray,
        inlier_mask: np.ndarray,
    ) -> np.ndarray | None:
        """Reconstruct PnP covariance from the final visual normal matrix.

        The tangent order matches pycolmap's pose covariance:
        [rotation_x, rotation_y, rotation_z, translation_x, ...]. Weakly
        observable directions retain large variance instead of making an
        otherwise valid pose lose its covariance message.
        """
        mask = np.asarray(inlier_mask, dtype=bool).reshape(-1)
        if mask.size != len(points2D) or np.count_nonzero(mask) < 6:
            return None
        observed = np.asarray(points2D[mask], dtype=np.float64).reshape(-1, 2)
        world_points = np.asarray(points3D[mask], dtype=np.float64).reshape(-1, 3)
        transform = rigid3d_to_matrix(pose['cam_from_world'])
        rotation = transform[:3, :3]
        translation = transform[:3, 3]

        def project(rotation_matrix: np.ndarray, translation_vector: np.ndarray):
            camera_points = (
                world_points @ rotation_matrix.T + translation_vector)
            if (
                not np.all(np.isfinite(camera_points))
                or np.any(camera_points[:, 2] <= 1.0e-9)
            ):
                return None
            pixels = np.asarray(
                self.camera0.img_from_cam(camera_points), dtype=np.float64
            ).reshape(-1, 2)
            return pixels if np.all(np.isfinite(pixels)) else None

        predicted = project(rotation, translation)
        if predicted is None:
            return None
        residual = (predicted - observed).reshape(-1)
        jacobian = np.empty((residual.size, 6), dtype=np.float64)
        rotation_step = 1.0e-6
        translation_step = 1.0e-5
        for column in range(6):
            axis = np.zeros(3, dtype=np.float64)
            if column < 3:
                axis[column] = rotation_step
                plus_rotation = cv2.Rodrigues(axis)[0] @ rotation
                minus_rotation = cv2.Rodrigues(-axis)[0] @ rotation
                plus = project(plus_rotation, translation)
                minus = project(minus_rotation, translation)
                denominator = 2.0 * rotation_step
            else:
                axis[column - 3] = translation_step
                plus = project(rotation, translation + axis)
                minus = project(rotation, translation - axis)
                denominator = 2.0 * translation_step
            if plus is None or minus is None:
                return None
            jacobian[:, column] = ((plus - minus) / denominator).reshape(-1)

        information = jacobian.T @ jacobian
        if not np.all(np.isfinite(information)):
            return None
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(
                0.5 * (information + information.T))
        except np.linalg.LinAlgError:
            return None
        maximum = float(eigenvalues[-1])
        if not np.isfinite(maximum) or maximum <= 1.0e-12:
            return None
        # Cap the numerical condition number, not the uncertainty. Replacing
        # an unobservable information eigenvalue by a tiny positive value
        # yields appropriately large covariance along that direction.
        information_floor = max(1.0e-12, maximum * 1.0e-12)
        inverse_information = (
            eigenvectors
            @ np.diag(1.0 / np.maximum(eigenvalues, information_floor))
            @ eigenvectors.T
        )
        degrees_of_freedom = max(1, residual.size - 6)
        pixel_variance = max(
            1.0, float(residual @ residual) / float(degrees_of_freedom))
        return self._valid_pnp_covariance(
            pixel_variance * inverse_information)

    def _pose_covariance_or_reconstruct(
        self,
        pose: Dict[str, Any],
        points2D: np.ndarray,
        points3D: np.ndarray,
        inlier_mask: np.ndarray,
        stereo_pixels: Tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        # The minimal solver triangulates each world point from the previous
        # stereo pair.  When those source pixels are available, marginalize
        # the point/depth uncertainty before falling back to COLMAP's fixed-
        # point covariance.  This is still a per-frame calculation (no window
        # bundle adjustment) and preserves the full 6x6 covariance.
        if stereo_pixels is not None:
            stereo_result = estimate_stereo_temporal_covariance(
                pose=pose['cam_from_world'],
                camera0=self.camera0,
                camera1=self.camera1,
                cam0_from_previous_world=self.cam0_pose_t0,
                cam1_from_previous_world=self.cam1_pose_t0,
                previous_left_pixels=stereo_pixels[0],
                previous_right_pixels=stereo_pixels[1],
                current_pixels=points2D,
                points_world=points3D,
                inlier_mask=inlier_mask,
                pixel_std_px=self.stereo_covariance_pixel_std_px,
                min_points=self.stereo_covariance_min_points,
            )
            if stereo_result is not None:
                current_log = self.log_data['current_frame']
                current_log['pnp_covariance_source'] = stereo_result.source
                current_log['pnp_covariance_points'] = stereo_result.used_points
                current_log['pnp_covariance_rank'] = stereo_result.rank
                current_log['pnp_covariance_condition_number'] = (
                    stereo_result.condition_number)
                current_log['pnp_covariance_reprojection_rms_px'] = (
                    stereo_result.reprojection_rms_px)
                covariance = self._valid_pnp_covariance(stereo_result.covariance)
                if covariance is not None:
                    return covariance

        covariance = self._valid_pnp_covariance(pose.get('covariance'))
        if covariance is not None:
            current_log = self.log_data['current_frame']
            current_log['pnp_covariance_source'] = 'pycolmap_fixed_points'
            current_log['pnp_covariance_points'] = int(
                np.count_nonzero(np.asarray(inlier_mask, dtype=bool)))
            try:
                eigenvalues = np.linalg.eigvalsh(covariance)
                positive = eigenvalues[eigenvalues > 1.0e-15]
                current_log['pnp_covariance_rank'] = int(len(positive))
                current_log['pnp_covariance_condition_number'] = float(
                    eigenvalues[-1] / max(eigenvalues[0], 1.0e-15))
            except np.linalg.LinAlgError:
                pass
            return covariance
        covariance = self._reprojection_pnp_covariance(
            pose, points2D, points3D, inlier_mask)
        if covariance is None:
            raise RuntimeError(
                'PnP pose succeeded but its covariance normal matrix could '
                'not be reconstructed')
        self.log_data['current_frame']['pnp_covariance_reconstructed'] = True
        self.log_data['current_frame']['pnp_covariance_source'] = (
            'reprojection_fixed_points')
        return covariance

    def _refine_pose_prior(
        self,
        pose_prior: Dict[str, Any] | None,
        points2D: np.ndarray,
        points3D: np.ndarray,
    ) -> Tuple[Dict[str, Any] | None, int, float]:
        if not pose_prior:
            return None, 0, float('inf')
        try:
            world_from_camera = np.asarray(
                pose_prior['world_from_camera'], dtype=np.float64).reshape(4, 4)
            maximum_error_px = float(pose_prior.get('maximum_error_px', 4.0))
            minimum_inliers = max(6, int(pose_prior.get('minimum_inliers', 12)))
            initial_pose = {
                'cam_from_world': matrix_to_rigid3d(
                    np.linalg.inv(world_from_camera))
            }
            count, _, inliers = self._pose_candidate_quality(
                initial_pose, points2D, points3D, maximum_error_px)
            if count < minimum_inliers:
                return None, count, float('inf')
            refined = pycolmap.refine_absolute_pose(
                cam_from_world=initial_pose['cam_from_world'],
                points2D=points2D,
                points3D=points3D,
                inlier_mask=inliers.reshape(-1, 1),
                camera=self.camera0,
                return_covariance=True,
            )
            refined_count, median_error, _ = self._pose_candidate_quality(
                refined, points2D, points3D, maximum_error_px)
            if refined_count < minimum_inliers:
                return None, refined_count, median_error
            return refined, refined_count, median_error
        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
        ):
            return None, 0, float('inf')

    def min_solver(
        self,
        image: str,
        keyframe_id: str,
        frame_t0: str,
        frame_t1: str,
        pose_prior: Dict[str, Any] | None = None,
    ) -> List[List]:
        """
        Minimal consecutive-frame solver using calibrated stereo geometry,
        PnP initialization, and structureless four-view refinement.
        
        Args:
            image: Current frame name
            frame_t0: Previous frame name
            frame_t1: Current frame name
            
        Returns:
            List containing pose change data or None values if failed
        """
        self.performance_stage = 'minimal_solver'
        try:
            self.log_data['current_frame']['status'] = True
            # Minimal stereo/PnP has no dependency on COLMAP's SQLite store.
            # Keeping the observations in memory also avoids synchronizing the
            # CUDA tensors merely to serialize and immediately deserialize
            # them again.
            kpts_t1_cam0 = self._as_keypoint_array(
                self.keypoints[f"cam0/{frame_t1}"])
            inlier_matches_t0_cam0_t1_cam0 = self._temporal_matches.get(
                (frame_t0, frame_t1))
            if inlier_matches_t0_cam0_t1_cam0 is None:
                raise RuntimeError("missing MAGSAC-verified consecutive-frame matches")
            inlier_matches_t0_cam0_t1_cam0 = np.asarray(
                inlier_matches_t0_cam0_t1_cam0, dtype=np.int64).reshape(-1, 2)
            solver_temporal_matches = inlier_matches_t0_cam0_t1_cam0

            self.log_data['current_frame']['inlier_matches'] = inlier_matches_t0_cam0_t1_cam0.shape[0]
            self.log_data['current_frame']['inlier_ratio'] = inlier_matches_t0_cam0_t1_cam0.shape[0] / self.log_data['current_frame']['num_features']

            # Stereo triangulation at t0
            self.performance_stage = 'minimal_solver:triangulation'
            stereo_cache_available = frame_t0 in self._stereo_observations
            cached_stereo = self._stereo_observations.get(frame_t0, {})
            if not stereo_cache_available:
                raise RuntimeError("missing stereo observations for previous frame")

            # PnP at t1. Calibrated essential MAGSAC remains the primary
            # temporal verifier. On nearly stationary or pure-rotation
            # intervals, essential estimation is mathematically degenerate
            # even though fixed stereo depth at t0 makes the actual PnP
            # problem well posed. In that case, use LoMa's mutual candidates
            # and let PnP's own LO-RANSAC reject outliers.
            points2D, points3D, previous_left_pixels, previous_right_pixels = \
                self._pnp_correspondences_with_stereo(
                inlier_matches_t0_cam0_t1_cam0, cached_stereo, kpts_t1_cam0)
            correspondence_source = 'magsac'
            raw_matches = self._temporal_raw_matches.get((frame_t0, frame_t1))
            if len(points2D) < 6 and raw_matches is not None:
                (raw_points2D, raw_points3D, raw_previous_left_pixels,
                 raw_previous_right_pixels) = self._pnp_correspondences_with_stereo(
                    raw_matches, cached_stereo, kpts_t1_cam0)
                if len(raw_points2D) >= 6:
                    points2D, points3D = raw_points2D, raw_points3D
                    previous_left_pixels = raw_previous_left_pixels
                    previous_right_pixels = raw_previous_right_pixels
                    solver_temporal_matches = np.asarray(
                        raw_matches, dtype=np.int64).reshape(-1, 2)
                    correspondence_source = 'magsac_degeneracy_fallback'

            current_log = self.log_data['current_frame']
            current_log['pnp_correspondences'] = int(len(points2D))
            current_log['pnp_temporal_match_source'] = correspondence_source

            if len(points2D) < 6:
                current_log['pnp_failure_reason'] = (
                    'insufficient_stereo_supported_temporal_matches:'
                    f'{len(points2D)}')
                if not self.process_every_frame:
                    self.reinitialize()
                return [[image, None, None, None, None, None]]

            self.performance_stage = 'minimal_solver:pnp'
            def estimate_visual_pose(
                image_points: np.ndarray,
                world_points: np.ndarray,
            ):
                return pycolmap.estimate_and_refine_absolute_pose(
                    points2D=image_points,
                    points3D=world_points,
                    camera=self.camera0,
                    estimation_options=self.PnP_options,
                    return_covariance=True,
                )

            ransac_pose = estimate_visual_pose(points2D, points3D)
            # A calibrated-essential model can still remove the scene subset
            # that has stereo depth (for example, repeated texture at high
            # frame rate). Retry only that failed visual PnP solve with
            # LoMa's mutual candidates; PnP LO-RANSAC remains the final
            # geometric arbiter.
            if (
                ransac_pose is None
                and correspondence_source == 'magsac'
                and raw_matches is not None
            ):
                (raw_points2D, raw_points3D, raw_previous_left_pixels,
                 raw_previous_right_pixels) = self._pnp_correspondences_with_stereo(
                    raw_matches, cached_stereo, kpts_t1_cam0)
                if len(raw_points2D) >= 6:
                    recovered_pose = estimate_visual_pose(
                        raw_points2D, raw_points3D)
                    if recovered_pose is not None:
                        points2D, points3D = raw_points2D, raw_points3D
                        previous_left_pixels = raw_previous_left_pixels
                        previous_right_pixels = raw_previous_right_pixels
                        solver_temporal_matches = np.asarray(
                            raw_matches, dtype=np.int64).reshape(-1, 2)
                        ransac_pose = recovered_pose
                        correspondence_source = 'magsac_pnp_recovery'
                        current_log['pnp_correspondences'] = int(len(points2D))
                        current_log['pnp_temporal_match_source'] = (
                            correspondence_source)
            seeded_pose, seeded_inliers, seeded_error = self._refine_pose_prior(
                pose_prior, points2D, points3D)
            ransac_inliers, ransac_error, ransac_inlier_mask = self._pose_candidate_quality(
                ransac_pose,
                points2D,
                points3D,
                self.PnP_options.ransac.max_error,
            )
            seeded_inlier_mask = np.zeros(len(points2D), dtype=bool)
            if seeded_pose is not None:
                seeded_inliers, seeded_error, seeded_inlier_mask = self._pose_candidate_quality(
                    seeded_pose, points2D, points3D,
                    self.PnP_options.ransac.max_error)
            # Both candidates below are visual-only refinements. The IMU pose
            # merely supplies the initial iterate for ``seeded_pose``; it is
            # not a residual in the PnP objective and is therefore not counted
            # again when the result is fused by the iEKF.
            pose = ransac_pose
            pose_source = 'ransac_visual'
            pose_inlier_mask = ransac_inlier_mask
            if seeded_pose is not None and (
                pose is None
                or seeded_inliers > ransac_inliers
                or (
                    seeded_inliers == ransac_inliers
                    and seeded_error < ransac_error
                )
            ):
                pose = seeded_pose
                pose_source = 'imu_seeded_visual'
                pose_inlier_mask = seeded_inlier_mask
            if pose is None:
                raise RuntimeError('visual PnP refinement failed')
            current_log['pnp_pose_source'] = pose_source
            current_log['pnp_ransac_inliers'] = ransac_inliers
            current_log['pnp_prior_inliers'] = seeded_inliers
            current_log['pnp_ransac_median_error_px'] = ransac_error
            current_log['pnp_prior_median_error_px'] = seeded_error
            pose_covariance = None

            # Refine the metric PnP result with the same calibrated stereo
            # observations that define its uncertainty.  This removes the
            # errors-in-variables bias caused by treating the previous DLT
            # points as exact, while remaining a strictly consecutive-frame
            # solver (temporary landmarks are Schur-eliminated immediately).
            structure_data = self._structureless_correspondences(
                solver_temporal_matches,
                cached_stereo,
                self._stereo_observations.get(frame_t1, {}),
                kpts_t1_cam0,
                previous_keypoints=self._as_keypoint_array(
                    self.keypoints[f"cam0/{frame_t0}"]
                ),
                previous_bearings=self._stereo_bearing_observations.get(
                    frame_t0, {}
                ),
                current_bearings=self._stereo_bearing_observations.get(
                    frame_t1, {}
                ),
            )
            if structure_data is not None:
                current_log['inertial_rotation_factor_used'] = bool(
                    pose_prior is not None
                    and pose_prior.get('rotation_prior') is not None
                    and pose_prior.get('rotation_covariance') is not None)
                current_log['inertial_gravity_factor_used'] = bool(
                    pose_prior is not None
                    and pose_prior.get('gravity_previous') is not None
                    and pose_prior.get('gravity_current') is not None
                    and pose_prior.get('gravity_sigma_rad') is not None)
                (structure_previous_left, structure_previous_right,
                 structure_current_left, structure_current_right,
                 structure_points) = structure_data
                # The temporal matches have already passed calibrated MAGSAC
                # verification.  They intentionally form a superset of the
                # metric PnP rows, so do not apply the PnP mask by position.
                if len(structure_points) >= 6:
                    initial_matrix = rigid3d_to_matrix(pose['cam_from_world'])
                    essential_result = self._estimate_calibrated_essential_rotation(
                        self._as_keypoint_array(
                            self.keypoints[f"cam0/{frame_t0}"]),
                        kpts_t1_cam0,
                        solver_temporal_matches,
                    )
                    if essential_result is not None:
                        essential_rotation, essential_inliers = essential_result
                        # Essential translation is scale-free.  Keep the
                        # metric PnP translation and replace only its rotation
                        # for the structureless initializer.  Reject an
                        # essential model that disagrees grossly with PnP;
                        # that is the expected degeneracy at near-stationary
                        # frames, where an arbitrary translation direction can
                        # produce an arbitrary essential rotation.
                        rotation_difference = cv2.Rodrigues(
                            essential_rotation @ initial_matrix[:3, :3].T
                        )[0]
                        rotation_difference_norm = float(
                            np.linalg.norm(rotation_difference))
                        current_log['essential_rotation_inliers'] = int(
                            essential_inliers)
                        current_log['essential_pnp_rotation_difference_deg'] = (
                            np.rad2deg(rotation_difference_norm))
                        if rotation_difference_norm <= np.deg2rad(15.0):
                            initial_matrix[:3, :3] = essential_rotation
                            current_log['structureless_initialization'] = (
                                'essential_rotation_pnp_translation')
                        else:
                            current_log['structureless_initialization'] = (
                                'pnp_translation_rotation_essential_rejected')
                    try:
                        structure_result = optimize_stereo_temporal_pose(
                            initial_pose=initial_matrix,
                            camera0=self.camera0,
                            camera1=self.camera1,
                            cam0_from_previous_world=self.cam0_pose_t0,
                            cam1_from_previous_world=self.cam1_pose_t0,
                            previous_left_pixels=structure_previous_left,
                            previous_right_pixels=structure_previous_right,
                            current_left_pixels=structure_current_left,
                            current_right_pixels=structure_current_right,
                            points_world=structure_points,
                            pixel_std_px=self.stereo_covariance_pixel_std_px,
                            min_points=self.stereo_covariance_min_points,
                            covariance_spatial_clusters=(
                                self.stereo_covariance_spatial_clusters),
                            max_iterations=self.structureless_max_iterations,
                            robust_delta_px=self.structureless_robust_delta_px,
                            inertial_rotation_prior=(
                                pose_prior.get('rotation_prior')
                                if pose_prior is not None else None),
                            inertial_rotation_covariance=(
                                pose_prior.get('rotation_covariance')
                                if pose_prior is not None else None),
                            gravity_direction_previous=(
                                pose_prior.get('gravity_previous')
                                if pose_prior is not None else None),
                            gravity_direction_current=(
                                pose_prior.get('gravity_current')
                                if pose_prior is not None else None),
                            gravity_sigma_rad=(
                                pose_prior.get('gravity_sigma_rad')
                                if pose_prior is not None else None),
                        )
                    except (RuntimeError, TypeError, ValueError,
                            np.linalg.LinAlgError) as error:
                        current_log['structureless_failure_reason'] = (
                            f'{type(error).__name__}:{error}')
                        structure_result = None
                    if structure_result is not None:
                        pose['cam_from_world'] = matrix_to_rigid3d(
                            structure_result.pose)
                        pose_covariance = structure_result.covariance
                        current_log['pnp_covariance_points'] = int(
                            structure_result.used_points)
                        current_log['pnp_covariance_rank'] = int(
                            structure_result.rank)
                        current_log['pnp_covariance_condition_number'] = float(
                            structure_result.condition_number)
                        current_log['pnp_covariance_reprojection_rms_px'] = (
                            float(structure_result.reprojection_rms_px))
                        current_log['structureless_used_points'] = int(
                            structure_result.used_points)
                        current_log['structureless_right_observations'] = int(
                            structure_result.right_observations)
                        current_log['structureless_iterations'] = int(
                            structure_result.iterations)
                        current_log['structureless_initial_cost'] = float(
                            structure_result.initial_cost)
                        current_log['structureless_final_cost'] = float(
                            structure_result.final_cost)
                        current_log['structureless_reprojection_rms_px'] = float(
                            structure_result.reprojection_rms_px)
                        current_log['structureless_information_eigenvalues'] = list(
                            structure_result.information_eigenvalues)
                        current_log['structureless_visual_information_eigenvalues'] = list(
                            structure_result.visual_information_eigenvalues)
                        current_log['structureless_inertial_information_eigenvalues'] = list(
                            structure_result.inertial_information_eigenvalues)
                        current_log['structureless_view_residual_rms_px'] = list(
                            structure_result.view_residual_rms_px)
                        current_log['structureless_robust_weight_stats'] = list(
                            structure_result.robust_weight_stats)
                        current_log['structureless_covariance_estimator'] = (
                            structure_result.covariance_estimator)
                        current_log['structureless_view_residual_scales'] = list(
                            structure_result.view_residual_scales)
                        current_log['structureless_rotation_factor_used'] = (
                            pose_prior is not None
                            and pose_prior.get('rotation_prior') is not None
                            and pose_prior.get('rotation_covariance') is not None)
                        current_log['structureless_gravity_factor_used'] = (
                            pose_prior is not None
                            and pose_prior.get('gravity_previous') is not None
                            and pose_prior.get('gravity_current') is not None
                            and pose_prior.get('gravity_sigma_rad') is not None)
                        current_log['inertial_rotation_factor_used'] = (
                            current_log['structureless_rotation_factor_used'])
                        current_log['inertial_gravity_factor_used'] = (
                            current_log['structureless_gravity_factor_used'])
                    else:
                        current_log.setdefault(
                            'structureless_failure_reason',
                            'solver_returned_no_solution')
                        current_log['inertial_rotation_factor_used'] = False
                        current_log['inertial_gravity_factor_used'] = False
                        current_log['pnp_covariance_source'] = (
                            'anchored_inverse_depth_stereo_temporal')

            if pose_covariance is None:
                pose_covariance = self._pose_covariance_or_reconstruct(
                    pose,
                    points2D,
                    points3D,
                    pose_inlier_mask,
                    stereo_pixels=(previous_left_pixels, previous_right_pixels),
                )
            self.log_data['current_frame']['covariance_matrix_angles_translations'] = pose_covariance
            inverse_covariance = self._valid_pnp_covariance(
                _covariance_for_inverse(
                    pose['cam_from_world'], pose_covariance))
            if inverse_covariance is None:
                raise RuntimeError(
                    'PnP covariance became invalid while converting from '
                    'world-to-camera to camera-to-world coordinates')
            self.log_data['current_frame'][
                'relative_pose_covariance_rotation_translation'
            ] = inverse_covariance

            delta_t, delta_q = colmap_cam_from_world_to_pose(pose['cam_from_world'])
            self.t_cumulative, self.q_cumulative = compose_pose(
                self.t_cumulative,
                self.q_cumulative,
                delta_t,
                delta_q,
            )

            # Keep only keypoints for cam0/cam1 at t0 and t1; remove all others.
            keep_keys = {
                f"cam0/{frame_t0}",
                f"cam1/{frame_t0}",
                f"cam0/{frame_t1}",
                f"cam1/{frame_t1}",
            }

            for k in list(self.keypoints.keys()):
                if k not in keep_keys:
                    self.keypoints.pop(k, None)
                    # keep other caches consistent
                    self.descriptors.pop(k, None)
                    self.lafs_cache.pop(k, None)

            for cached_frame in list(self._stereo_observations):
                if cached_frame not in {frame_t0, frame_t1}:
                    self._stereo_observations.pop(cached_frame, None)
                    self._stereo_bearing_observations.pop(cached_frame, None)
            for match_key in list(self._temporal_matches):
                if frame_t1 not in match_key:
                    self._temporal_matches.pop(match_key, None)

            # refresh memory stats after pruning
            self._update_memory_stats()

            current_log['pnp_success'] = True

            #return [[image, self.keyframes_names[curr.name], delta_t, delta_q, self.t_cumulative, self.q_cumulative]]
            return [[image, keyframe_id, delta_t, delta_q, self.t_cumulative, self.q_cumulative]]
        
        except Exception as e:
            current_log = self.log_data['current_frame']
            current_log['status'] = False
            current_log['pnp_failure_reason'] = (
                f'{type(e).__name__}: {e}')
            return [[image, None, None, None, None, None]]
    
    def clean_database(self, image_id: int):
        database_path = self.database_path
        self.db.close()
        db_start = time.time()
        #try:
        # Connect to SQLite DB (create connection to the COLMAP DB file)
        conn = sqlite3.connect(str(database_path))
        cur = conn.cursor()
        # Ensure foreign keys are enforced (if schema uses them)
        cur.execute("PRAGMA foreign_keys = ON;")
        # Remove keypoints and descriptors for the image
        cur.execute("DELETE FROM keypoints WHERE image_id = ?;", (image_id,))
        cur.execute("DELETE FROM descriptors WHERE image_id = ?;", (image_id,))
        # Remove any two-view geometry entries that reference this image.
        # COLMAP encodes pair_id as: pair_id = image_id1 + image_id2 * 2147483647
        # Decode each pair_id and delete rows that include the image_id.
        cur.execute("SELECT pair_id FROM two_view_geometries;")
        pair_rows = cur.fetchall()
        to_delete = []
        for (pair_id,) in pair_rows:
            try:
                img1 = int(pair_id % 2147483647)
                img2 = int(pair_id // 2147483647)
            except Exception:
                # if pair_id is not an integer for some reason, skip
                continue
            if img1 == image_id or img2 == image_id:
                to_delete.append(pair_id)
        if to_delete:
            cur.executemany("DELETE FROM two_view_geometries WHERE pair_id = ?;", [(pid,) for pid in to_delete])
        # Finally remove the image record itself
        cur.execute("DELETE FROM images WHERE image_id = ?;", (image_id,))
        conn.commit()
        conn.close()
        # mark controller DB as dirty if needed and log timing
        self.db_dirty = True
        db_time = time.time() - db_start
        self._log_timing('database_operations_total', db_time)

        self.db = Database(str(self.database_path))

        #except Exception as e:
        #    try:
        #        conn.rollback()
        #        conn.close()
        #    except Exception:
        #        pass
        #    # minimal logging to stdout/file
        #    if self.log:
        #        print(f"[CSLAM] clean_database error for image_id={image_id}: {e}")
        #    db_time = time.time() - db_start
        #    self._log_timing('database_operations_total', db_time)

    def _write_minimal_frame_to_database(
        self,
        image: str,
        frame_id: int,
        matches: Dict[Tuple[str, str], np.ndarray],
    ) -> None:
        """Optionally mirror the in-memory minimal-solver state to COLMAP."""
        if not self.write_colmap_database or self.db is None:
            return
        database_start = time.perf_counter()
        if frame_id == 1:
            for camera_name in self.cameras:
                self.db.write_camera(Camera(self.camera_config[camera_name]))
        for camera_index, camera_name in enumerate(self.cameras):
            name = f"{camera_name}/{image}"
            image_id = frame_id + camera_index
            self.keyframes_names[name] = image_id
            self.keyframes_ids[image_id] = name
            self.write_keypoints_to_db(
                self.db, name, image_id, camera_index + 1, self.keypoints)
        for (name0, name1), inlier_matches in matches.items():
            self.db.write_two_view_geometry(
                self.keyframes_names[name0], self.keyframes_names[name1],
                TwoViewGeometry({"inlier_matches": inlier_matches}),
            )
        self.db_dirty = True
        self._log_timing('database_time', time.perf_counter() - database_start)

    def _run_every_frame_minimal_solver(
        self,
        image: str,
        images: List[np.ndarray],
        frame_start_time: float,
        pose_prior: Dict[str, Any] | None,
    ):
        """Run the minimal stereo/PnP odometer for every synchronized frame."""
        if self.n_cameras != 2:
            raise RuntimeError("every-frame minimal solver requires exactly two cameras")

        frame_names = {
            camera_name: f"{camera_name}/{image}" for camera_name in self.cameras
        }
        feature_counts, feature_statuses = self._extract_and_store_batch([
            (frame_names[camera_name], images[index])
            for index, camera_name in enumerate(self.cameras)
        ])
        if not all(feature_statuses.values()) or not all(feature_counts.values()):
            self.log_data['current_frame']['corrupted_master_image'] = True
            self.log_data['current_frame']['pnp_failure_reason'] = (
                'feature_extraction_failed')
            return [[image, None, None, None, None, None]], self.log_data

        is_first_frame = self.frame_t0 is None
        previous_frame = self.frame_t0
        pairs: List[Tuple[str, str]] = []
        if previous_frame is not None:
            pairs.append((f"cam0/{previous_frame}", frame_names['cam0']))
        pairs.extend(self.rig_match_pairs(image))

        self.performance_stage = 'feature_matching:batched'
        matching_start = time.perf_counter()
        matches = self.match_features(self.keypoints, self.descriptors, pairs)
        self._log_timing('feature_matching', time.perf_counter() - matching_start)
        self._store_match_debug(pairs, matches)

        stereo_start = time.perf_counter()
        self._update_stereo_observations(image, matches)
        self._log_timing('stereo_observation_time', time.perf_counter() - stereo_start)

        if is_first_frame:
            self.keyframe_id = 1
            self.keyframe_name = frame_names['cam0']
            self.frame_t0 = image
            self.keyframes_master_ids = [self.keyframe_id]
            self.log_data['current_frame']['is_keyframe'] = True
            # The first image defines the world origin. There is no preceding
            # stereo cloud yet, hence no PnP problem can be formed for this
            # one bootstrap sample.
            self.log_data['current_frame']['pnp_failure_reason'] = (
                'bootstrap_no_previous_stereo_frame')
            self.log_data['keyframe_count'] += 1
            self._write_minimal_frame_to_database(image, self.keyframe_id, matches)
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self.log_data['current_frame']['status'] = self.current_status
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()
            return [[image, None, None, None, None, None]], self.log_data

        temporal_pair = (f"cam0/{previous_frame}", frame_names['cam0'])
        self._temporal_matches[(previous_frame, image)] = np.asarray(
            matches[temporal_pair], dtype=np.int64).reshape(-1, 2).copy()
        self._temporal_raw_matches[(previous_frame, image)] = np.asarray(
            self._raw_feature_matches.get(temporal_pair, matches[temporal_pair]),
            dtype=np.int64).reshape(-1, 2).copy()
        self.frame_t1 = image
        self.keyframe_id += self.n_cameras
        self.keyframe_name = frame_names['cam0']
        self.keyframes_master_ids = [self.keyframe_id]
        self.log_data['current_frame']['is_keyframe'] = True
        self.log_data['keyframe_count'] += 1
        self._write_minimal_frame_to_database(image, self.keyframe_id, matches)

        if self.create_feature_tracks:
            self._update_feature_tracks(
                f"cam0/{previous_frame}", frame_names['cam0'], matches[temporal_pair])

        minimal_solver_start = time.perf_counter()
        result = self.min_solver(
            image, self.keyframe_id, previous_frame, image, pose_prior=pose_prior)
        self.log_data['current_frame']['minimal_solver_time'] = (
            time.perf_counter() - minimal_solver_start)

        # Advance even after a failed PnP estimate: the next image must be
        # matched against its actual predecessor, never against a hidden
        # keyframe several samples old.
        self.frame_t0 = image
        self.frame_t1 = None
        # PnP has consumed the interval cache. Retain only the current
        # stereo cloud for the next consecutive interval, regardless of
        # whether the current PnP attempt succeeded.
        self._temporal_matches.clear()
        self._temporal_raw_matches.clear()
        for cached_frame in list(self._stereo_observations):
            if cached_frame != image:
                self._stereo_observations.pop(cached_frame, None)
                self._stereo_bearing_observations.pop(cached_frame, None)
        keep_keys = {frame_names['cam0'], frame_names['cam1']}
        for name in list(self.keypoints):
            if name not in keep_keys:
                self.keypoints.pop(name, None)
                self.descriptors.pop(name, None)
                self.lafs_cache.pop(name, None)
        if not self.write_colmap_database:
            self.keyframes_names = {name: self.keyframe_id + index
                                    for index, name in enumerate(keep_keys)}
            self.keyframes_ids = {identifier: name
                                  for name, identifier in self.keyframes_names.items()}

        frame_time = time.time() - frame_start_time
        self.log_data['current_frame']['processing_time'] = frame_time
        self.log_data['current_frame']['status'] = result[0][1] is not None
        self._log_timing('frame_processing_total', frame_time)
        self._update_memory_stats()
        self._update_performance_stats()
        return result, self.log_data

    def run(
        self, image: str,
        images: List[np.ndarray],
        reinitialize: bool,
        pose_prior: Dict[str, Any] | None = None,
    ):
        self.performance_stage = 'frame_setup'
        frame_start_time = time.time()
        self.last_feature_match_debug = []
        
        # Initialize current frame data
        self.log_data['current_frame'] = {
            'status': self.current_status,
            'frame_name': image,
            'is_keyframe': False,
            'num_features': 0,
            'median_match_distance': 0.0,
            'inlier_matches': 0.0,
            'inlier_ratio': 0.0,
            'processing_time': 0.0,
            'feature_extraction_time': 0.0,
            'feature_model_time': 0.0,
            'feature_tensor_transfer_time': 0.0,
            'matching_time': 0.0,
            'matching_model_time': 0.0,
            'matching_geometry_time': 0.0,
            'matching_database_time': 0.0,
            'keyframe_selection_time': 0.0,
            'stereo_observation_time': 0.0,
            'database_time': 0.0,
            'reconstruction_time': 0.0,
            'minimal_solver_time': 0.0,
            'corrupted_master_image': False,
            'corrupted_slave_image': False,
            'not_enough_features_on_master': False,
            'covariance_matrix_angles_translations': None,
            'relative_pose_covariance_rotation_translation': None,
            'pnp_covariance_source': 'none',
            'pnp_covariance_points': 0,
            'pnp_covariance_rank': 0,
            'pnp_covariance_condition_number': None,
            'pnp_covariance_reprojection_rms_px': None,
            'pnp_pose_source': 'none',
            'pnp_ransac_inliers': 0,
            'pnp_prior_inliers': 0,
            'pnp_ransac_median_error_px': None,
            'pnp_prior_median_error_px': None,
            'pnp_covariance_reconstructed': False,
            'pnp_success': False,
            'pnp_failure_reason': '',
            'pnp_correspondences': 0,
            'pnp_temporal_match_source': 'none',
            'feature_track_updates': 0,
            'stereo_matches_raw': 0,
            'stereo_matches_epipolar': 0,
            'stereo_points_candidates': 0,
            'stereo_points_triangulated': 0,
            'stereo_points_valid': 0,
            'stereo_rejected_indices': 0,
            'stereo_rejected_triangulation': 0,
            'stereo_rejected_cheirality': 0,
            'stereo_rejected_depth': 0,
            'stereo_rejected_reprojection': 0,
            'stereo_rejected_angle': 0,
        }
        self.last_feature_track_updates = []
        self.images.append(image)
        self.log_data['frame_count'] += 1
        if self.create_feature_tracks:
            self._feature_frame_indices[image] = self.log_data['frame_count'] - 1

        if reinitialize:
            self.reinitialize()
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self._log_timing('frame_processing_total', frame_time)
            return [[image, None, None, None, None, None]], self.log_data

        if self.use_min_solver and self.process_every_frame:
            return self._run_every_frame_minimal_solver(
                image, images, frame_start_time, pose_prior)

        if len(self.keyframes_names.keys()) == 0:
            # first batch: register all cameras for the first timestamp
            self.frame_t0 = copy.deepcopy(image)
            status = []
            rig_num_features = []
            rig = {}
            for c, cam in enumerate(self.cameras):
                camera = Camera(self.camera_config[f"{cam}"])
                self.db.write_camera(camera)
                keyframe_name = f"{cam}/{image}"
                camera_id = 1 + c
                image_id = 1 + c
                rig[c] = (keyframe_name, image_id, camera_id)

            batch_items = [
                (rig[c][0], images[c]) for c in range(len(self.cameras))]
            feature_counts, feature_statuses = self._extract_and_store_batch(
                batch_items)
            for c in range(len(self.cameras)):
                status.append(feature_statuses.get(rig[c][0], False))
                rig_num_features.append(feature_counts.get(rig[c][0], 0))

            # Check if all images were read successfully and have features
            if not all(status):
                self.log_data['current_frame']['corrupted_master_image'] = True
                return [[image, None, None, None, None, None]], self.log_data
            elif not all(rig_num_features):
                self.log_data['current_frame']['not_enough_features_on_master'] = True
                return [[image, None, None, None, None, None]], self.log_data 
            else:
                for c, cam in enumerate(self.cameras):
                    keyframe_name = rig[c][0]
                    image_id = rig[c][1]
                    camera_id = rig[c][2]
                    self.write_keypoints_to_db(self.db, keyframe_name, image_id, camera_id, self.keypoints)
                    self.keyframes_names[keyframe_name] = image_id
                    self.keyframes_ids[image_id] = keyframe_name

            self.keyframe_name = f"cam0/{image}"
            self.keyframes_master_ids.append(1)
            self.log_data['current_frame']['is_keyframe'] = True
            self.log_data['keyframe_count'] += 1

            if self.n_cameras != 1:
                matching_start = time.time()
                pairs = self.rig_match_pairs(image)
                matches = self.match_features(self.keypoints, self.descriptors, pairs)
                self._store_match_debug(pairs, matches)
                stereo_start = time.perf_counter()
                self._update_stereo_observations(image, matches)
                self._log_timing(
                    'stereo_observation_time', time.perf_counter() - stereo_start)
                database_start = time.perf_counter()
                for pair in pairs:
                    kfrm1, kfrm2 = pair
                    inlier_matches = matches[pair]
                    self.db.write_two_view_geometry(
                        self.keyframes_names[kfrm1],
                        self.keyframes_names[kfrm2],
                        TwoViewGeometry({"inlier_matches": inlier_matches})
                    )
                self.db_dirty = True
                self._log_timing(
                    'matching_database_time', time.perf_counter() - database_start)
                matching_time = time.time() - matching_start
                self._log_timing('feature_matching_total', matching_time)
                self._log_timing('feature_matching', matching_time)
            
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()
            
            return [[image, None, None, None, None, None]], self.log_data

        # -------- Keyframe selection based on feature motion --------
        matching_start = time.time()
        frame_name = f"cam0/{image}"
        num_features, reading_images_status = self._extract_and_store(frame_name, images[0])
        if not reading_images_status:
                self.log_data['current_frame']['corrupted_master_image'] = True
                return [[image, None, None, None, None, None]], self.log_data
        elif num_features==0:
            self.log_data['current_frame']['not_enough_features_on_master'] = True
            return [[image, None, None, None, None, None]], self.log_data 

        pairs = [(self.keyframe_name, frame_name)]
        matches = self.match_features(self.keypoints, self.descriptors, pairs)
        self._store_match_debug(pairs, matches)
        median_match_dist = self.match_distance(self.keypoints, matches, self.keyframe_name, frame_name)
        matching_time = time.time() - matching_start
        self._log_timing('feature_matching_total', matching_time)
        self._log_timing('feature_matching', matching_time)

        keyframe_decision_start = time.perf_counter()
        is_keyframe = median_match_dist >= self.config['mapping']['max_match_distance']
        self._log_timing(
            'keyframe_selection_time', time.perf_counter() - keyframe_decision_start)
        if not is_keyframe:
            # not a keyframe: free memory of temporary frame
            self.keypoints.pop(frame_name, None)
            self.descriptors.pop(frame_name, None)
            self.lafs_cache.pop(frame_name, None)
            
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self.log_data['current_frame']['status'] = self.current_status
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()
            
            return [[image, None, None, None, None, None]], self.log_data

        # --- Promote to keyframe on master camera ---
        self.frame_t1 = image
        self.log_data['current_frame']['is_keyframe'] = True
        self.log_data['keyframe_count'] += 1
        
        previous_keyframe_name = self.keyframe_name
        temporal_inlier_matches = matches[(previous_keyframe_name, frame_name)]
        self.keyframe_count += 1
        self.keyframe_id += 1 * self.n_cameras
        self.keyframe_name = frame_name
        camera_id = 1
        self.write_keypoints_to_db(self.db, self.keyframe_name, self.keyframe_id, camera_id, self.keypoints)
        self.db.write_two_view_geometry(
            self.keyframe_id - 1 * self.n_cameras,
            self.keyframe_id,
            TwoViewGeometry({"inlier_matches": temporal_inlier_matches})
        )
        self.db_dirty = True
        self.keyframes_names[self.keyframe_name] = self.keyframe_id
        self.keyframes_ids[self.keyframe_id] = self.keyframe_name
        self.keyframes_master_ids.append(self.keyframe_id)


        # --- Match slave cameras ---
        if self.n_cameras != 1:
            slave_matching_start = time.time()
            slave_items = []
            for c, cam in enumerate(self.cameras):
                if c == 0:
                    continue
                slave_name = f"{cam}/{image}"
                slave_id = self.keyframe_id + 1 * c
                self.keyframes_names[slave_name] = slave_id
                self.keyframes_ids[slave_id] = slave_name
                camera_id = c + 1
                slave_items.append((slave_name, images[c]))

            _, slave_statuses = self._extract_and_store_batch(slave_items)
            for c, cam in enumerate(self.cameras):
                if c == 0:
                    continue
                slave_name = f"{cam}/{image}"
                slave_id = self.keyframes_names[slave_name]
                camera_id = c + 1
                if not slave_statuses.get(slave_name, False):
                    self.log_data['current_frame']['corrupted_slave_image'] = True
                    self.keypoints[slave_name] = torch.empty(
                        (0, 2), device=self.device, dtype=torch.float32)
                    self.descriptors[slave_name] = torch.empty(
                        (0, self.local_features.decriptor_dim()),
                        device=self.device, dtype=torch.float32)
                self.write_keypoints_to_db(
                    self.db, slave_name, slave_id, camera_id, self.keypoints)

            pairs = self.rig_match_pairs(image)
            matches = self.match_features(self.keypoints, self.descriptors, pairs)
            self._store_match_debug(pairs, matches)
            stereo_start = time.perf_counter()
            self._update_stereo_observations(image, matches)
            self._log_timing(
                'stereo_observation_time', time.perf_counter() - stereo_start)
            database_start = time.perf_counter()
            for pair in pairs:
                kfrm1, kfrm2 = pair
                inlier_matches = matches[pair]
                self.db.write_two_view_geometry(
                    self.keyframes_names[kfrm1],
                    self.keyframes_names[kfrm2],
                    TwoViewGeometry({"inlier_matches": inlier_matches})
                )
            self.db_dirty = True
            self._log_timing(
                'matching_database_time', time.perf_counter() - database_start)
            slave_matching_time = time.time() - slave_matching_start
            self._log_timing('feature_matching_total', slave_matching_time)

        if self.create_feature_tracks:
            self._update_feature_tracks(
                previous_keyframe_name, frame_name, temporal_inlier_matches)

        if self.log:
            print(f"[CSLAM] Matching time: {matching_time:.2f} seconds")


        # Fast stereo VO route. Triangulate the previous stereo pair and solve
        # the current master camera with PnP, bypassing reconstruction and BA.
        if self.use_min_solver:
            minimal_solver_start = time.time()
            result = self.min_solver(
                image,
                self.keyframe_id,
                self.frame_t0,
                self.frame_t1,
                pose_prior=pose_prior,
            )
            self.log_data['current_frame']['minimal_solver_time'] = (
                time.time() - minimal_solver_start)
            if result[0][1] is not None:
                self.frame_t0 = copy.deepcopy(self.frame_t1)

            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self.log_data['current_frame']['status'] = result[0][1] is not None
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()
            return result, self.log_data


            



        # --- Orientation / Reconstruction step ---
        reconstruction_start = time.time()

        if self.keyframe_count <= 5:
            return self._run_min_solver_fallback(
                image, pose_prior=pose_prior), self.log_data

        if 5 < self.keyframe_count < self.sliding_window:
            self.old_pairs = pairs

        if 5 < self.keyframe_count < self.sliding_window:
            self.current_status = 'reconstruction_initialization'
            self._maybe_load_db()
            #reconstruct(self.controller, self.mapper_options, self.tri_options, self.keyframe_id, False, run_BA=True, max_cost_change_px=self.max_cost_change_px, no_stereo_constraint_in_local_ba=self.config['general']['no_stereo_constraint_in_local_ba'], cam_poses=[self.cam0_pose_t0, self.cam1_pose_t0]);quit()
            try:
                if self.config['mapping']['method'] == 'custom':
                    self.performance_stage = 'reconstruction_initialization:custom'
                    reconstruct(self.controller, self.mapper_options, self.tri_options, self.keyframe_id, False, run_BA=True, max_cost_change_px=self.max_cost_change_px, no_stereo_constraint_in_local_ba=self.config['general']['no_stereo_constraint_in_local_ba'], cam_poses=[self.cam0_pose_t0, self.cam1_pose_t0])
                    self.log_data['reconstruction_stats']['bundle_adjustments'] += 1
                else:
                    self.performance_stage = 'reconstruction_initialization:pycolmap'
                    self.controller.reconstruct(self.mapper_options)
                self.log_data['reconstruction_stats']['successful_reconstructions'] += 1
            except Exception:
                self.log_data['reconstruction_stats']['failed_reconstructions'] += 1
                self.reinitialize()
                frame_time = time.time() - frame_start_time
                self.log_data['current_frame']['processing_time'] = frame_time
                return self._run_min_solver_fallback(
                    image, pose_prior=pose_prior), self.log_data
            if self.test:
                self.reconstruction_manager.write(self.out_dir)
            
            reconstruction_time = time.time() - reconstruction_start
            self._log_timing('reconstruction_total', reconstruction_time)
            self._log_timing('reconstruction', reconstruction_time)
            
            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()

            return self._run_min_solver_fallback(
                image, pose_prior=pose_prior), self.log_data

        elif self.keyframe_count >= self.sliding_window:
            self.current_status = 'orientation'
            #st = time.time()
            self._maybe_load_db()
            #ed = time.time()
            #self.log_file.write(f"Time to load DB: {ed - st:.2f} seconds\n")

            if self.config['mapping']['method'] == 'custom':
                # Add new keyframes (entire rig)
                for c in range(self.n_cameras):
                    try:
                        if c == self.n_cameras - 1:
                            self.run_BA = True
                        if self.log:
                            tt0 = time.time()
                        self.performance_stage = f'reconstruction_update:camera_{c}'
                        reconstruct(self.controller, self.mapper_options, self.tri_options, self.keyframe_id + c, True, run_BA=self.run_BA, max_cost_change_px=self.max_cost_change_px, no_stereo_constraint_in_local_ba=self.config['general']['no_stereo_constraint_in_local_ba'], cam_poses=[self.cam0_pose_t0, self.cam1_pose_t0])
                        if self.run_BA:
                            self.log_data['reconstruction_stats']['bundle_adjustments'] += 1
                        if self.log:
                            tt1 = time.time()
                            print('[CSLAM] Time for reconstruct:', tt1 - tt0)
                        self.run_BA = False
                    except Exception:
                        if self.log:
                            print('[CSLAM] Error in keyframe orientation')
                        self.log_data['reconstruction_stats']['failed_reconstructions'] += 1
                        self.reinitialize()
                        frame_time = time.time() - frame_start_time
                        self.log_data['current_frame']['processing_time'] = frame_time
                        return self._run_min_solver_fallback(
                            image, pose_prior=pose_prior), self.log_data

                # Sliding window: deregister oldest
                reconstruction = self.reconstruction_manager.get(idx=0)
                reg_image_ids = reconstruction.reg_image_ids()
                n_to_deregister = len(reg_image_ids) - self.n_cameras * self.sliding_window
                for _ in range(max(0, n_to_deregister)):
                    reg_image_ids = reconstruction.reg_image_ids()
                    reconstruction.deregister_image(image_id=min(reg_image_ids))
                    self.log_data['reconstruction_stats']['sliding_window_operations'] += 1
                    image_deregistered = self.keyframes_ids[min(reg_image_ids)]
                    self.keypoints.pop(image_deregistered, None)
                    # keep other caches consistent
                    self.descriptors.pop(image_deregistered, None)
                    self.lafs_cache.pop(image_deregistered, None)
                    self.keyframes_names.pop(image_deregistered, None)
                    self.keyframes_ids.pop(min(reg_image_ids), None)
                    #if _ == 0:
                    #    self.keyframes_master_ids.pop(min(reg_image_ids))
                    self.clean_database(image_id=min(reg_image_ids))
                    #print(self.keypoints.keys());quit()

                self.log_data['reconstruction_stats']['successful_reconstructions'] += 1

                # Scale the reconstruction (stereo rigs)
                if self.n_cameras != 1:
                    baselines = []
                    for img_id in list(set(self.keyframes_master_ids) & set(reconstruction.reg_image_ids())):
                        master = reconstruction.image(image_id=img_id)
                        timestamp = self.keyframes_ids[img_id].split("/")[1]
                        slave_id = self.keyframes_names.get(f"{self.second_baseline_camera}/{timestamp}")
                        if slave_id in reconstruction.reg_image_ids():
                            slave = reconstruction.image(image_id=slave_id)
                            baselines.append(np.linalg.norm(slave.projection_center() - master.projection_center()))
                    scale_factor = (np.median(baselines) / self.baseline) if baselines else 1.0
                else:
                    scale_factor = 1.0

                if self.test:
                    self.reconstruction_manager.write(self.out_dir)

            else:  # pycolmap reconstruct
                try:
                    self.performance_stage = 'reconstruction_update:pycolmap'
                    self.controller.reconstruct(self.mapper_options)
                    reconstruction = self.reconstruction_manager.get(idx=0)
                    self.log_data['reconstruction_stats']['successful_reconstructions'] += 1
                except Exception:
                    self.log_data['reconstruction_stats']['failed_reconstructions'] += 1
                    self.reinitialize()
                    frame_time = time.time() - frame_start_time
                    self.log_data['current_frame']['processing_time'] = frame_time
                    #return [[image, None, None, None, None, None]], self.log_data
                    return result, self.log_data

            reconstruction_time = time.time() - reconstruction_start
            self._log_timing('reconstruction_total', reconstruction_time)
            self._log_timing('reconstruction', reconstruction_time)

            if self.keyframe_count == self.config['mapping']['max_keyframes']:
                self.reconstruction_manager.write(self.out_dir)
                print("Max keyframes reached, exiting...")
                raise SystemExit

            if self.log:
                print(f"[CSLAM] Orientation time {reconstruction_time:.2f} seconds")

            # --- Extract change in pose ---
            self.performance_stage = 'pose_extraction'
            pose_start = time.time()
            try:
                if self.n_cameras == 1:
                    reconstruction = self.reconstruction_manager.get(idx=0)
                    last = reconstruction.image(image_id=self.keyframe_count - 1)
                    prev = reconstruction.image(image_id=self.keyframe_count - 2)
                    curr = reconstruction.image(image_id=self.keyframe_count)
                    delta_t, delta_q = relative_camera_pose(
                        image_cam_from_world(last), image_cam_from_world(curr))

                    if self.baseline_old == 0:
                        self.baseline_old = np.linalg.norm(curr.projection_center() - last.projection_center())
                        s = 1.0
                    else:
                        baseline = np.linalg.norm(last.projection_center() - prev.projection_center())
                        s = baseline / self.baseline_old if self.baseline_old != 0 else 1.0
                        self.baseline_old = np.linalg.norm(curr.projection_center() - last.projection_center())

                    delta_t = delta_t / (s if s != 0 else 1.0)

                    last_pose = image_cam_from_world(last)
                    t = last_pose.translation
                    r = last_pose.rotation
                    reconstruction.transform(pycolmap.Sim3d({'translation': t, 'rotation': r, 'scale': 1}))
                else:
                    reconstruction = self.reconstruction_manager.get(idx=0)
                    ref = reconstruction.image(image_id=self.keyframes_master_ids[-2])
                    curr = reconstruction.image(image_id=self.keyframes_master_ids[-1])
                    delta_t, delta_q = relative_camera_pose(
                        image_cam_from_world(ref), image_cam_from_world(curr))

                    # scale_factor computed above (if custom path). If using pycolmap path and scale_factor undefined, default to 1.0
                    if 'scale_factor' not in locals():
                        scale_factor = 1.0
                    delta_t = delta_t / (scale_factor if scale_factor != 0 else 1.0)

                    ref_pose = image_cam_from_world(ref)
                    t = ref_pose.translation
                    r = ref_pose.rotation
                    reconstruction.transform(pycolmap.Sim3d({'translation': t, 'rotation': r, 'scale': 1}))

                self.t_cumulative, self.q_cumulative = compose_pose(
                    self.t_cumulative,
                    self.q_cumulative,
                    delta_t,
                    delta_q,
                )
                
                pose_time = time.time() - pose_start
                self._log_timing('pose_estimation_total', pose_time)
                
            except Exception:
                if self.log:
                    print('[CSLAM] Error in estimate change pose')
                self.reinitialize()
                frame_time = time.time() - frame_start_time
                self.log_data['current_frame']['processing_time'] = frame_time
                return self._run_min_solver_fallback(
                    image, pose_prior=pose_prior), self.log_data

            if self.log:
                print(f"[CSLAM] Estimate change pose time {pose_time:.2f} seconds")
                print(f"[CSLAM] delta_t {delta_t}, delta_q [{delta_q}]")

            frame_time = time.time() - frame_start_time
            self.log_data['current_frame']['processing_time'] = frame_time
            self._log_timing('frame_processing_total', frame_time)
            self._update_memory_stats()
            self._update_performance_stats()

            self.log_data['current_frame']['covariance_matrix_angles_translations'] = np.eye(6, dtype=np.float64) * 9999

            return [[image, self.keyframes_names[curr.name], delta_t, delta_q, self.t_cumulative, self.q_cumulative]], self.log_data

        # else path handled above when not promoted to keyframe
        frame_time = time.time() - frame_start_time
        self.log_data['current_frame']['processing_time'] = frame_time
        self._log_timing('frame_processing_total', frame_time)
        self._update_memory_stats()
        self._update_performance_stats()
        
        #return [[image, None, None, None, None, None]], self.log_data
        return result, self.log_data

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary"""
        return {
            'frame_statistics': {
                'total_frames': self.log_data['frame_count'],
                'total_keyframes': self.log_data['keyframe_count'],
                'keyframe_ratio': self.log_data['performance']['keyframe_selection_ratio'],
                'total_reinitializations': self.log_data['reinitializations']
            },
            'timing_summary': {
                'total_processing_time': self.log_data['timing']['frame_processing_total'],
                'avg_frame_time': self.log_data['performance']['avg_frame_processing_time'],
                'feature_extraction_total': self.log_data['timing']['feature_extraction_total'],
                'feature_matching_total': self.log_data['timing']['feature_matching_total'],
                'reconstruction_total': self.log_data['timing']['reconstruction_total'],
                'pose_estimation_total': self.log_data['timing']['pose_estimation_total']
            },
            'feature_statistics': {
                'total_features_extracted': self.log_data['total_features_extracted'],
                'avg_features_per_frame': self.log_data['performance']['avg_features_per_frame'],
                'total_matches_computed': self.log_data['total_matches_computed']
            },
            'reconstruction_statistics': self.log_data['reconstruction_stats'],
            'memory_statistics': self.log_data['memory_stats']
        }
