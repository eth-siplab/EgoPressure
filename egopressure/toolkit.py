"""``EgoPressure`` — the one-stop facade for the toolkit.

Ties together partial download, data access, and visualization behind a small
surface that reads well in a notebook.

    from egopressure import EgoPressure

    # local data you already have
    ep = EgoPressure("data")

    # or pull just a slice from the Hub
    ep = EgoPressure.from_hub("eth-siplab/EgoPressure",
                              participants=["p_001"], cameras=["d"],
                              modalities=["rgb", "depth", "pressure", "pose"])

    ep.participants                                  # ['p_001', ...]
    frame = ep.frame("p_001", "p_001_press_palm_low_x5_right", 60)
    frame.show(camera="d", overlays=["mesh", "skeleton", "pressure"])
"""

from __future__ import annotations

from pathlib import Path

from .constants import DEFAULT_DATASET_REPO
from .dataset import EgoPressureDataset, Frame, Sequence
from .hub import download


class EgoPressure:
    """High-level entry point wrapping :class:`EgoPressureDataset`."""

    def __init__(self, root: str | Path):
        self.dataset = EgoPressureDataset(root)
        self.root = Path(root)

    # ── construction ───────────────────────────────────────────────────────
    @classmethod
    def from_hub(
        cls,
        repo: str | None = None,
        out_dir: str | Path = "egopressure_data",
        **selection,
    ) -> EgoPressure:
        """Download a subset from the Hub, then open it.

        ``selection`` is forwarded to :func:`egopressure.hub.download`
        (``participants=``, ``sequences=``, ``cameras=``, ``modalities=``, ...).
        """
        path = download(repo=repo or DEFAULT_DATASET_REPO, out_dir=out_dir,
                        **selection)
        return cls(path)

    # ── access ─────────────────────────────────────────────────────────────
    @property
    def participants(self) -> list[str]:
        return self.dataset.participants

    def sequence_names(self, participant: str) -> list[str]:
        return self.dataset.sequence_names(participant)

    def sequence(self, participant: str, name: str) -> Sequence:
        return self.dataset.sequence(participant, name)

    def sequences(self, participant: str | None = None) -> list[Sequence]:
        return self.dataset.sequences(participant)

    def frame(self, participant: str, sequence: str, index: int,
              **load_kwargs) -> Frame:
        """Load a fully-populated :class:`Frame` (carries a viewer via ``.show()``)."""
        return self.sequence(participant, sequence).load_frame(index, **load_kwargs)

    def show(self, participant: str, sequence: str, index: int, **kwargs):
        """Render a frame directly: ``ep.show(p, seq, i, camera=..., overlays=...)``."""
        return self.sequence(participant, sequence).show(index, **kwargs)

    def viewer(self, participant: str, sequence: str, depth_provider=None):
        """A :class:`~egopressure.viewer.FrameViewer` for a sequence."""
        return self.sequence(participant, sequence).viewer(
            depth_provider=depth_provider)

    def __repr__(self) -> str:
        return (f"<EgoPressure root={str(self.root)!r} "
                f"participants={len(self.participants)}>")
