"""Optional PyTorch ``Dataset`` wrapper over EgoPressure frames.

Kept in its own module so the core package has **no** hard dependency on
torch. Import this only when torch is installed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

try:  # resolve the base class at import time so the class is picklable
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # pragma: no cover
    _TorchDataset = object

from .dataset import EgoPressureDataset, Frame, Sequence


def collate_frames(batch: list) -> list:
    """Identity collate: keep a batch as a plain list of :class:`Frame`.

    A named module-level function (unlike a lambda) so
    ``DataLoader(num_workers>0)`` can pickle it under the *spawn* start
    method (the default on macOS and Windows).
    """
    return list(batch)


class EgoPressureFrames(_TorchDataset):
    """A flat ``torch.utils.data.Dataset`` over ``(sequence, frame)`` pairs.

    Instances are picklable, so ``DataLoader(num_workers>0)`` works under
    both the *fork* and *spawn* start methods.

    Args:
        root: Dataset root passed to :class:`EgoPressureDataset`.
        participants: Restrict to these participant ids (default: all).
        annotated_only: Keep only frames with ``has_annotation == True``.
        transform: Optional ``Frame -> Any`` callable applied per item. For
            multi-worker loading this must be a module-level function (or
            other picklable callable), not a lambda.
        load_kwargs: Forwarded to :meth:`Sequence.load_frame`
            (e.g. ``cameras=["d"], mask=True``).
    """

    def __init__(
        self,
        root: str | Path,
        participants: list[str] | None = None,
        annotated_only: bool = False,
        transform: Callable[[Frame], object] | None = None,
        **load_kwargs,
    ):
        if _TorchDataset is object:  # pragma: no cover
            raise ImportError(
                "EgoPressureFrames requires PyTorch. `pip install torch`."
            )
        self.dataset = EgoPressureDataset(root)
        self.transform = transform
        self.load_kwargs = load_kwargs
        self._items: list[tuple[Sequence, int]] = []

        pids = participants or self.dataset.participants
        for pid in pids:
            for seq in self.dataset.sequences(pid):
                indices = seq.annotated_frames() if annotated_only else seq.frames
                self._items.extend((seq, i) for i in indices)

    @property
    def items(self) -> list[tuple[Sequence, int]]:
        """The flat ``(sequence, frame_index)`` list — sequence boundaries
        for clip sampling can be derived from consecutive entries."""
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        seq, idx = self._items[i]
        frame = seq.load_frame(idx, **self.load_kwargs)
        return self.transform(frame) if self.transform else frame


def _contiguous_runs(indices: list[int]) -> list[list[int]]:
    """Split a sorted frame-index list into runs of consecutive numbers."""
    runs: list[list[int]] = []
    for i in indices:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def build_clip_index(
    frame_lists: list[list[int]],
    window: int,
    stride: int = 1,
    hop: int | None = None,
) -> list[list[list[int]]]:
    """Clip start positions per sequence (pure function — unit-testable).

    For each sequence's sorted frame list, clips are sampled inside runs of
    *consecutive* frame numbers only (dropped/unannotated frames break a run),
    so no clip spans a temporal gap or a sequence boundary. Short runs yield
    no clips — there is no padding.

    Args:
        frame_lists: One sorted frame-index list per sequence.
        window: Frames per clip.
        stride: Step between consecutive frames *within* a clip.
        hop: Start-to-start distance between clips (default: ``window * stride``
            — non-overlapping).

    Returns:
        Per sequence, a list of clips; each clip is a list of ``window``
        frame indices.
    """
    span = (window - 1) * stride + 1
    hop = hop if hop is not None else window * stride
    out: list[list[list[int]]] = []
    for indices in frame_lists:
        clips: list[list[int]] = []
        for run in _contiguous_runs(sorted(indices)):
            for s in range(0, len(run) - span + 1, hop):
                clips.append(run[s:s + span:stride])
        out.append(clips)
    return out


class EgoPressureClips(_TorchDataset):
    """Fixed-length temporal clips for sequence models.

    Each item is a list of ``window`` :class:`Frame` objects (or
    ``transform(frames)``), sampled with ``stride`` inside runs of
    consecutive frames — clips never span a sequence boundary or a gap of
    dropped/unannotated frames, and short runs are skipped (no padding).

    Args:
        root: Dataset root passed to :class:`EgoPressureDataset`.
        window: Frames per clip.
        stride: Frame step within a clip (2 = every other frame).
        hop: Start-to-start distance between clips (default: non-overlapping).
        participants: Restrict to these participant ids (default: all).
        annotated_only: Sample clips only inside annotated runs.
        transform: Optional ``list[Frame] -> Any`` (picklable) per item.
        load_kwargs: Forwarded to :meth:`Sequence.load_frame`.
    """

    def __init__(
        self,
        root: str | Path,
        window: int = 16,
        stride: int = 1,
        hop: int | None = None,
        participants: list[str] | None = None,
        annotated_only: bool = False,
        transform: Callable[[list], object] | None = None,
        **load_kwargs,
    ):
        if _TorchDataset is object:  # pragma: no cover
            raise ImportError(
                "EgoPressureClips requires PyTorch. `pip install torch`."
            )
        self.dataset = EgoPressureDataset(root)
        self.transform = transform
        self.load_kwargs = load_kwargs
        self._items: list[tuple[Sequence, list[int]]] = []

        pids = participants or self.dataset.participants
        seqs = [s for pid in pids for s in self.dataset.sequences(pid)]
        frame_lists = [s.annotated_frames() if annotated_only else s.frames
                       for s in seqs]
        for seq, clips in zip(seqs, build_clip_index(frame_lists, window,
                                                     stride, hop)):
            self._items.extend((seq, clip) for clip in clips)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        seq, indices = self._items[i]
        frames = [seq.load_frame(idx, **self.load_kwargs) for idx in indices]
        return self.transform(frames) if self.transform else frames
