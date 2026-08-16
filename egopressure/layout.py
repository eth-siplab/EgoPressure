"""Canonical Hugging Face repo layout for EgoPressure.

Single source of truth for how the dataset is sharded on the Hub, so the
downloader (:mod:`egopressure.hub`) and the dataset tooling
never disagree. The layout is designed so a user can fetch exactly the slice they
need — by participant, sequence, camera, or modality — via glob patterns.

Repo tree::

    <repo>/
      README.md                                   # dataset card
      configs/<participant>/<sequence>.json       # calibration + metadata
      data/<participant>/<sequence>/
          cam-d.color.parquet     cam-1.color.parquet ... cam-7.color.parquet
          cam-d.depth.parquet     cam-1.depth.parquet ... # depth, all 8 cameras
          cam-d.mask.parquet      cam-1.mask.parquet  ...  # hand masks
          pressure.parquet                                 # force grid + UV pressure
          annotation.parquet                               # MANO pose/mesh + ego pose

Download granularity == file granularity, so cameras and modalities split into
separate Parquet files.
"""

from __future__ import annotations

from .constants import ALL_CAMERAS, EGO_CAMERA

CONFIG_ROOT = "configs"
DATA_ROOT = "data"

# Modalities the user can select, and how each maps to on-disk shard files.
# Per-camera modalities take a camera token; shared modalities are single files.
PER_CAMERA_MODALITIES = ("color", "mask", "depth")
SHARED_MODALITIES = ("pressure", "annotation")
# friendly aliases -> canonical modality
MODALITY_ALIASES = {
    "rgb": "color",
    "image": "color",
    "images": "color",
    "masks": "mask",
    "pose": "annotation",
    "mano": "annotation",
    "anno": "annotation",
    "force": "pressure",
}
ALL_MODALITIES = PER_CAMERA_MODALITIES + SHARED_MODALITIES


def canonical_modality(name: str) -> str:
    """Normalise a user-facing modality name to its canonical form."""
    m = MODALITY_ALIASES.get(name, name)
    if m not in ALL_MODALITIES:
        raise ValueError(f"Unknown modality {name!r}; choose from {ALL_MODALITIES} "
                         f"(or aliases {tuple(MODALITY_ALIASES)}).")
    return m


def shard_filename(modality: str, camera: str | None = None) -> str:
    """Filename of the Parquet shard for a modality (+ camera if per-camera)."""
    modality = canonical_modality(modality)
    if modality in PER_CAMERA_MODALITIES:
        if camera is None:
            raise ValueError(f"Modality {modality!r} is per-camera; pass `camera`.")
        return f"cam-{camera}.{modality}.parquet"
    return f"{modality}.parquet"


def config_path(participant: str, sequence: str) -> str:
    """Repo-relative path of a sequence's config JSON."""
    return f"{CONFIG_ROOT}/{participant}/{sequence}.json"


def data_dir(participant: str, sequence: str) -> str:
    """Repo-relative directory of a sequence's shards."""
    return f"{DATA_ROOT}/{participant}/{sequence}"


def shard_path(participant: str, sequence: str, modality: str,
               camera: str | None = None) -> str:
    """Repo-relative path of one shard file."""
    return f"{data_dir(participant, sequence)}/{shard_filename(modality, camera)}"


def allow_patterns(
    participants: list[str] | None = None,
    sequences: list[str] | None = None,
    cameras: list[str] | None = None,
    modalities: list[str] | None = None,
    include_configs: bool = True,
) -> list[str]:
    """Build ``allow_patterns`` globs for ``snapshot_download``.

    ``None`` means "all" for that axis. Cameras only constrain per-camera
    modalities; shared modalities (pressure/annotation) are always included when
    selected regardless of camera.
    """
    if cameras:
        bad = [c for c in cameras if c not in ALL_CAMERAS]
        if bad:
            raise ValueError(
                f"unknown camera token(s) {bad} — valid: "
                f"'{EGO_CAMERA}' (egocentric), '1'..'7' (static)")
    parts = participants or ["*"]
    seqs = sequences or ["*"]
    mods = ([canonical_modality(m) for m in modalities]
            if modalities else list(ALL_MODALITIES))
    cams = list(cameras) if cameras is not None else list(ALL_CAMERAS)

    patterns: list[str] = []
    for p in parts:
        for s in seqs:
            if include_configs:
                # scope configs to the selected sequences — a sequence has one
                # calibration JSON plus a k4a folder; pulling a participant's
                # whole config tree would fetch hundreds of unrelated files
                patterns.append(f"{CONFIG_ROOT}/{p}/{s}.json")
                patterns.append(f"{CONFIG_ROOT}/{p}/{s}_k4a/*.json")
            base = f"{DATA_ROOT}/{p}/{s}"
            for m in mods:
                if m in PER_CAMERA_MODALITIES:
                    for c in cams:
                        patterns.append(f"{base}/{shard_filename(m, c)}")
                else:
                    patterns.append(f"{base}/{shard_filename(m)}")
    return sorted(set(patterns))
