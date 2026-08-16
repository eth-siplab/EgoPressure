"""Partial download of EgoPressure from the Hugging Face Hub.

Fetch exactly the slice you need — by participant, sequence, camera, and
modality — instead of the whole dataset. Thin wrapper over
``huggingface_hub.snapshot_download`` that turns a friendly selection into the
right ``allow_patterns`` (see :mod:`egopressure.layout`).

    from egopressure.hub import download
    path = download(participants=["p_001"], cameras=["d"],
                    modalities=["rgb", "depth", "pressure", "pose"])
    from egopressure import EgoPressureDataset
    ds = EgoPressureDataset(path)
"""

from __future__ import annotations

from pathlib import Path

from . import layout
from .constants import DEFAULT_DATASET_REPO


def download(
    participants: list[str] | None = None,
    sequences: list[str] | None = None,
    cameras: list[str] | None = None,
    modalities: list[str] | None = None,
    out_dir: str | Path = "egopressure_data",
    repo: str = DEFAULT_DATASET_REPO,
    revision: str = "main",
    token: str | None = None,
    include_configs: bool = True,
    max_workers: int = 8,
) -> Path:
    """Download a subset of the dataset to ``out_dir``.

    Args:
        participants: e.g. ``["p_001", "p_002"]``; ``None`` = all.
        sequences: full sequence names *including the participant prefix*
            (e.g. ``["p_001_press_palm_low_x5_right"]``); ``None`` = all.
        cameras: camera tokens (``"d"``, ``"1"``..``"7"``); ``None`` = all. Only
            constrains per-camera modalities (color/mask/depth).
        modalities: any of ``color/rgb, mask, depth, pressure, annotation/pose``;
            ``None`` = all.
        out_dir: local destination directory.
        repo: Hub dataset repo id.
        revision: branch, tag, or commit.
        token: HF token for gated/private access (or use ``hf auth login``).
        include_configs: also fetch the small per-sequence config JSONs.
        max_workers: parallel download workers.

    Returns:
        Path to the local dataset root (pass to :class:`EgoPressureDataset`).
    """
    # huggingface_hub is imported lazily: it is slow to import and only
    # needed for network operations, keeping `import egopressure` fast
    from huggingface_hub import snapshot_download

    patterns = layout.allow_patterns(
        participants=participants,
        sequences=sequences,
        cameras=cameras,
        modalities=modalities,
        include_configs=include_configs,
    )
    local = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        revision=revision,
        token=token,
        allow_patterns=patterns,
        local_dir=str(out_dir),
        max_workers=max_workers,
    )
    root = Path(local)
    # guard against a silent empty selection (e.g. a sequence name without
    # its participant prefix matches nothing but "succeeds")
    for seq in sequences or []:
        if not list((root / layout.DATA_ROOT).glob(f"*/{seq}")):
            raise ValueError(
                f"selection matched no files for sequence '{seq}' — sequence "
                "names include the participant prefix, e.g. "
                "'p_001_press_palm_low_x5_right' (see `egopressure list -v`)")
    for pid in participants or []:
        if not (root / layout.DATA_ROOT / pid).is_dir():
            raise ValueError(
                f"selection matched no files for participant '{pid}' "
                "(expected ids like 'p_001' — see `egopressure list`)")
    return root


def list_available(repo: str = DEFAULT_DATASET_REPO, revision: str = "main",
                   token: str | None = None) -> dict:
    """Summarise what a Hub repo contains: participants, sequences, modalities.

    Returns a dict ``{participant: {sequence: [shard filenames]}}``.
    """
    # lazy for the same import-cost reason as `download`
    from huggingface_hub import list_repo_files

    files = list_repo_files(repo, repo_type="dataset", revision=revision, token=token)
    tree: dict[str, dict[str, list[str]]] = {}
    prefix = layout.DATA_ROOT + "/"
    for f in files:
        if not f.startswith(prefix):
            continue
        rel = f[len(prefix):].split("/")
        if len(rel) != 3:
            continue
        participant, sequence, shard = rel
        tree.setdefault(participant, {}).setdefault(sequence, []).append(shard)
    return tree
