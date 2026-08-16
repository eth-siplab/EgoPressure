"""Dataset-wide constants and naming conventions for EgoPressure.

All magic numbers, file-naming patterns, and camera/MANO layout facts live here
so the rest of the package reads declaratively.
"""

from __future__ import annotations

# ── Cameras ────────────────────────────────────────────────────────────────
# The rig has 8 synchronised cameras:
#   * one egocentric ("dynamic") camera, stored on disk as ``cam-d`` (1920x1080)
#   * seven static Azure Kinect cameras, ``cam-1`` .. ``cam-7`` (2560x1440)
#
# In the per-sequence config JSON the same cameras are keyed "0".."7", where
# "0" is the egocentric camera and "1".."7" are the static Kinects.  This map
# translates the on-disk camera token to its config key.
EGO_CAMERA: str = "d"
STATIC_CAMERAS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6", "7")
ALL_CAMERAS: tuple[str, ...] = (EGO_CAMERA,) + STATIC_CAMERAS

CAMERA_TO_CONFIG_KEY: dict[str, str] = {
    EGO_CAMERA: "0",
    "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
}

# ── Sensel Morph pressure sensor ───────────────────────────────────────────
# Each frame's ``force`` column stores the raw float32 pressure grid from the
# Sensel Morph touchpad: 19,425 cells = 105 rows x 185 columns (row-major).
FORCE_DTYPE = "float32"
FORCE_SHAPE: tuple[int, int] = (105, 185)  # (height, width)
FORCE_NUM_CELLS: int = FORCE_SHAPE[0] * FORCE_SHAPE[1]  # 19425

# ── MANO hand model layout ─────────────────────────────────────────────────
MANO_NUM_VERTICES: int = 778
MANO_NUM_JOINTS: int = 21
MANO_NUM_SHAPE_PARAMS: int = 10       # betas
MANO_POSE_DIM: int = 48               # full_pose: 16 joints x 3 axis-angle

# ── Pressure UV map (baked into each annotation) ───────────────────────────
PRESSURE_MAP_SIZE: tuple[int, int] = (224, 224)

# ── Visualization defaults ─────────────────────────────────────────────────
DEPTH_VIS_MAX_MM: float = 1500.0   # colormap range for depth rendering

# ── Parquet shard array schema ─────────────────────────────────────────────
# Flat float32 list columns in the annotation/pressure shards reshape to:
ARRAY_SHAPES = {
    "vertices": (MANO_NUM_VERTICES, 3),
    "joint_position": (MANO_NUM_JOINTS, 3),
    "normals": (MANO_NUM_VERTICES, 3),
    "displacement": (MANO_NUM_VERTICES, 1),
    "betas": (1, MANO_NUM_SHAPE_PARAMS),
    "full_pose": (MANO_POSE_DIM,),
    "transl": (3,),
    "ego_R": (3, 3),
    "ego_T": (3,),
    "force": FORCE_SHAPE,
    "pressure_map": (*PRESSURE_MAP_SIZE, 1),
    "pressure_map_range": (2,),
    "visible_vertices": (len(STATIC_CAMERAS), MANO_NUM_VERTICES),
}

# ── Distribution ────────────────────────────────────────────────────────────
#: Canonical Hugging Face dataset repository for EgoPressure.
DEFAULT_DATASET_REPO = "eth-siplab/EgoPressure"
