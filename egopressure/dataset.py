"""Core data model: dataset discovery, sequence interface, frames.

The dataset is distributed as Parquet shards in the Hub layout
(:mod:`egopressure.layout`)::

    <root>/
      configs/<participant>/<sequence>.json     # calibration + metadata
      data/<participant>/<sequence>/*.parquet   # one shard per camera/modality

:class:`EgoPressureDataset` discovers participants/sequences under a root;
:class:`Sequence` defines the loading interface plus everything shared
(calibration, participant metadata, frame orchestration, visualization);
the concrete shard reader is :class:`egopressure.parquet.ParquetSequence`.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

from .annotation import Annotation
from .calibration import CameraCalibration
from .constants import ALL_CAMERAS, CAMERA_TO_CONFIG_KEY
from .layout import CONFIG_ROOT, DATA_ROOT


class ModalityNotDownloaded(FileNotFoundError):
    """A whole shard (camera/modality) is absent from the local download —
    as opposed to a single frame the camera dropped at capture time."""


# ── metadata containers ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class Participant:
    """Demographic + capture metadata from the sequence config."""

    id: str
    age: int | None = None
    height: float | None = None
    weight: float | None = None
    gender: str | None = None
    hand_side: str | None = None
    exposure: str | None = None
    light_tubes: str | None = None
    sequence_name: str | None = None

    @classmethod
    def from_config(cls, block: dict) -> Participant:
        """Build from the config JSON's ``participant`` block."""
        def _num(v, cast):
            try:
                return cast(v)
            except (TypeError, ValueError):
                return None

        return cls(
            id=str(block.get("id", "")),
            age=_num(block.get("age"), int),
            height=_num(block.get("height"), float),
            weight=_num(block.get("weight"), float),
            gender=block.get("gender"),
            hand_side=block.get("hand_side"),
            exposure=block.get("exposure"),
            light_tubes=block.get("light_tubes"),
            sequence_name=block.get("sequence_name"),
        )


@dataclass
class Frame:
    """One synchronised time step. Populated fields depend on the request.

    Attributes:
        index: Frame number.
        rgb: ``{camera: (H, W, 3) uint8}`` for requested cameras.
        mask: ``{camera: (H, W[, C]) uint8}`` hand masks.
        depth: ``{camera: (512, 512) uint16 mm}`` (a camera that dropped this
            capture has no entry).
        force: ``(105, 185) float32`` raw Sensel pressure, or ``None``.
        annotation: :class:`Annotation`, or ``None``.
    """

    index: int
    rgb: dict[str, np.ndarray] = field(default_factory=dict)
    mask: dict[str, np.ndarray] = field(default_factory=dict)
    depth: dict[str, np.ndarray] = field(default_factory=dict)
    force: np.ndarray | None = None
    annotation: Annotation | None = None
    sequence: Sequence | None = None   # back-ref for .show()

    def show(self, **kwargs):
        """Render this frame (kwargs -> ``FrameViewer.render``)."""
        if self.sequence is None:
            raise ValueError("Frame has no sequence reference.")
        return self.sequence.viewer().render(self.index, **kwargs)


# ── sequence interface ──────────────────────────────────────────────────────
class Sequence(ABC):
    """One gesture recording: frames, cameras, calibration, participant.

    Concrete backends implement the per-field loaders; everything else —
    config-derived calibration/metadata, whole-frame assembly, iteration, and
    visualization — is shared.
    """

    def __init__(self, directory: str | Path, config_path: str | Path | None = None):
        self.directory = Path(directory)
        self.name = self.directory.name
        self._config_path = Path(config_path) if config_path else None

    @property
    def config_path(self) -> Path | None:
        """Path to this sequence's config JSON (``None`` if not downloaded)."""
        return self._config_path

    # ── config-derived ────────────────────────────────────────────────────
    @cached_property
    def _frame_set(self) -> set[int]:
        return set(self.frames)

    @cached_property
    def _config(self) -> dict:
        if self._config_path is None or not self._config_path.exists():
            raise FileNotFoundError(
                f"No config JSON for sequence '{self.name}'. Calibration and "
                "participant metadata are unavailable."
            )
        with open(self._config_path) as f:
            return json.load(f)

    @cached_property
    def participant(self) -> Participant:
        """Participant metadata for this sequence."""
        return Participant.from_config(self._config["participant"])

    @cached_property
    def calibrations(self) -> dict[str, CameraCalibration]:
        """``{camera: CameraCalibration}`` keyed by camera token
        (``"d"``, ``"1"``..``"7"``)."""
        cc = self._config["camera_calibrations"]
        return {
            cam: CameraCalibration.from_config(cam, cc[CAMERA_TO_CONFIG_KEY[cam]])
            for cam in ALL_CAMERAS
            if CAMERA_TO_CONFIG_KEY[cam] in cc
        }

    def calibration(self, camera: str) -> CameraCalibration:
        """Calibration for a single camera token."""
        return self.calibrations[camera]

    # ── backend interface ─────────────────────────────────────────────────
    @property
    @abstractmethod
    def frames(self) -> list[int]:
        """Sorted frame indices present in this sequence."""

    @property
    @abstractmethod
    def cameras(self) -> list[str]:
        """Camera tokens with imagery in this sequence."""

    @abstractmethod
    def load_rgb(self, index: int, camera: str) -> np.ndarray:
        """RGB image ``(H, W, 3) uint8`` for one camera at one frame."""

    @abstractmethod
    def load_mask(self, index: int, camera: str) -> np.ndarray:
        """Hand mask for one camera at one frame."""

    @abstractmethod
    def load_force(self, index: int, reshape: bool = True) -> np.ndarray:
        """Raw Sensel pressure grid ``(105, 185) float32`` for one frame."""

    @abstractmethod
    def load_annotation(self, index: int) -> Annotation:
        """MANO + pressure annotation for one frame."""

    # ── optional depth interface (backends override when depth exists) ────
    def load_depth(self, index: int, camera: str) -> np.ndarray:
        """Depth map for one camera; backends without depth raise."""
        raise ModalityNotDownloaded(
            f"{self.name}: this data source provides no depth")

    def has_depth(self, camera: str) -> bool:
        """True if this sequence can serve depth for ``camera``."""
        return False

    # ── whole-frame loader ────────────────────────────────────────────────
    def load_frame(
        self,
        index: int,
        cameras: list[str] | None = None,
        rgb: bool = True,
        mask: bool = False,
        depth: bool = False,
        force: bool = True,
        annotation: bool = True,
    ) -> Frame:
        """Assemble a :class:`Frame` with the requested modalities.

        Args:
            index: Frame number (must be in :attr:`frames`).
            cameras: Camera tokens to load imagery for (default: all present).
            rgb, mask, depth: Per-camera image modalities. A camera that
                dropped this frame at capture simply has no entry for it;
                requesting a modality whose shard wasn't downloaded raises
                :class:`ModalityNotDownloaded`.
            force: Read the raw pressure grid (``None`` on the frame if the
                pressure shard wasn't downloaded).
            annotation: Read the MANO/pressure annotation (``None`` on the
                frame if the annotation shard wasn't downloaded).
        """
        if index not in self._frame_set:
            raise KeyError(f"Frame {index} not in sequence '{self.name}'.")
        cams = cameras if cameras is not None else self.cameras
        frame = Frame(index=index, sequence=self)
        for cam in cams:
            # a frame an individual camera dropped at capture (missing shard
            # row) is skipped; an entirely missing shard (modality/camera not
            # downloaded) raises ModalityNotDownloaded with a download hint
            if rgb:
                try:
                    frame.rgb[cam] = self.load_rgb(index, cam)
                except ModalityNotDownloaded:
                    raise
                except FileNotFoundError:
                    pass                    # camera dropped this frame
            if mask:
                try:
                    frame.mask[cam] = self.load_mask(index, cam)
                except ModalityNotDownloaded:
                    raise
                except FileNotFoundError:
                    pass
            if depth:
                try:
                    frame.depth[cam] = self.load_depth(index, cam)
                except ModalityNotDownloaded:
                    raise
                except FileNotFoundError:
                    pass                    # camera dropped this frame
        if force:
            try:
                frame.force = self.load_force(index)
            except ModalityNotDownloaded:
                frame.force = None          # partial download: no pressure
        if annotation:
            try:
                frame.annotation = self.load_annotation(index)
            except ModalityNotDownloaded:
                frame.annotation = None     # partial download: no annotation
        return frame

    # ── iteration / helpers ───────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int) -> Frame:
        """Positional access into :attr:`frames` (default modalities)."""
        return self.load_frame(self.frames[i])

    def iter_frames(self, **load_kwargs) -> Iterator[Frame]:
        """Yield every frame in order (kwargs -> :meth:`load_frame`)."""
        for idx in self.frames:
            yield self.load_frame(idx, **load_kwargs)

    def annotated_frames(self) -> list[int]:
        """Frame indices with ``has_annotation == True`` (labelled contact)."""
        return [i for i in self.frames if self.load_annotation(i).has_annotation]

    # ── visualization ─────────────────────────────────────────────────────
    def viewer(self, depth_provider=None):
        """A :class:`~egopressure.viewer.FrameViewer` for this sequence."""
        # imported lazily: viewer imports this module (circular otherwise)
        from .viewer import FrameViewer

        return FrameViewer(self, depth_provider=depth_provider)

    def show(self, frame_index: int, **kwargs):
        """Shortcut for ``self.viewer().render(frame_index, **kwargs)``."""
        return self.viewer().render(frame_index, **kwargs)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}: {len(self)} frames>"


# ── dataset root ────────────────────────────────────────────────────────────
class EgoPressureDataset:
    """Discover participants and sequences under a dataset root (Hub layout).

    Example:
        >>> ds = EgoPressureDataset("egopressure_data")
        >>> ds.participants
        ['p_001']
        >>> seq = ds.sequence("p_001", "p_001_press_palm_low_x5_right")
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        data_root = self.root / DATA_ROOT
        if not data_root.is_dir():
            raise FileNotFoundError(
                f"{self.root} is not a dataset root (expected a '{DATA_ROOT}/' "
                "directory, as produced by egopressure.hub.download)."
            )
        self._data_root = data_root
        self._config_root = self.root / CONFIG_ROOT

    @property
    def participants(self) -> list[str]:
        """Sorted participant ids."""
        return sorted(p.name for p in self._data_root.iterdir() if p.is_dir())

    def sequence_names(self, participant: str) -> list[str]:
        """Sorted sequence names for one participant."""
        pdir = self._data_root / participant
        if not pdir.is_dir():
            raise KeyError(f"No participant '{participant}' under {self.root}.")
        return sorted(p.name for p in pdir.iterdir() if p.is_dir())

    def sequence(self, participant: str, name: str) -> Sequence:
        """Open one sequence."""
        # imported lazily: parquet imports this module (circular otherwise)
        from .parquet import ParquetSequence

        directory = self._data_root / participant / name
        if not directory.is_dir():
            raise KeyError(f"No sequence '{name}' for participant '{participant}'.")
        config = self._config_root / participant / f"{name}.json"
        return ParquetSequence(directory, config if config.exists() else None)

    def sequences(self, participant: str | None = None) -> list[Sequence]:
        """All sequences, optionally for one participant."""
        pids = [participant] if participant else self.participants
        return [self.sequence(p, n) for p in pids for n in self.sequence_names(p)]

    def __iter__(self) -> Iterator[Sequence]:
        return iter(self.sequences())

    def __len__(self) -> int:
        return sum(len(self.sequence_names(p)) for p in self.participants)

    def __repr__(self) -> str:
        return (f"<EgoPressureDataset root={str(self.root)!r} "
                f"participants={len(self.participants)} sequences={len(self)}>")
