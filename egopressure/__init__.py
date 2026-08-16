"""EgoPressure toolkit — load, download, and visualize the EgoPressure dataset.

Quick start::

    from egopressure import EgoPressure

    ep = EgoPressure.from_hub(participants=["p_001"], cameras=["d"],
                              modalities=["rgb", "depth", "pressure", "pose"])
    ep.frame("p_001", "p_001_press_palm_low_x5_right", 60).show()
"""

from .annotation import Annotation
from .calibration import CameraCalibration
from .constants import (
    ALL_CAMERAS,
    ARRAY_SHAPES,
    EGO_CAMERA,
    FORCE_SHAPE,
    STATIC_CAMERAS,
)
from .dataset import (
    EgoPressureDataset,
    Frame,
    ModalityNotDownloaded,
    Participant,
    Sequence,
)
from .hub import download
from .mano import find_mano_models, mano_available
from .parquet import ParquetSequence
from .toolkit import EgoPressure
from .video import save_video
from .viewer import FrameViewer

__all__ = [
    "ALL_CAMERAS",
    "ARRAY_SHAPES",
    "Annotation",
    "CameraCalibration",
    "EGO_CAMERA",
    "EgoPressure",
    "EgoPressureDataset",
    "FORCE_SHAPE",
    "Frame",
    "FrameViewer",
    "ModalityNotDownloaded",
    "ParquetSequence",
    "Participant",
    "STATIC_CAMERAS",
    "Sequence",
    "download",
    "find_mano_models",
    "mano_available",
    "save_video",
]

__version__ = "0.1.0"
