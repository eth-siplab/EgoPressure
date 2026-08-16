"""Parquet shard reader — the concrete :class:`Sequence` backend.

Each sequence directory holds one shard per (camera, modality) plus shared
``pressure`` and ``annotation`` shards (see :mod:`egopressure.layout`). Image
columns store the original file bytes (lossless); array columns are flat
float lists with shapes documented in
:data:`egopressure.constants.ARRAY_SHAPES` (the toolkit returns float32).

Reads are **row-group granular**: only the row group containing the requested
frame is decoded (a small per-instance LRU keeps recent groups), so random
access stays cheap on shards written with small row groups, while legacy
single-row-group shards behave like the previous whole-table cache.
"""

from __future__ import annotations

import io
from collections import OrderedDict
from functools import cached_property

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from . import layout
from .annotation import Annotation
from .constants import ALL_CAMERAS, ARRAY_SHAPES
from .dataset import ModalityNotDownloaded, Sequence

_GROUP_CACHE_SIZE = 24     # decoded row groups kept per sequence instance
                           # (> the ~18 shards an all-camera frame touches)


class ParquetSequence(Sequence):
    """A sequence backed by Parquet shards (the Hub distribution format)."""

    # ── shard access ────────────────────────────────────────────────────────
    def _meta(self, filename: str):
        """``(ParquetFile, {frame: (group, offset)})`` or None if absent.

        Builds the frame index from the ``frame`` column only — no row data
        is decoded until a specific row group is requested.
        """
        cache = self.__dict__.setdefault("_meta_cache", {})
        if filename not in cache:
            path = self.directory / filename
            if not path.exists():
                cache[filename] = None
            else:
                pf = pq.ParquetFile(path)
                index: dict[int, tuple[int, int]] = {}
                row = 0
                frames = pf.read(columns=["frame"]).column("frame").to_pylist()
                for g in range(pf.metadata.num_row_groups):
                    n = pf.metadata.row_group(g).num_rows
                    for off in range(n):
                        index[int(frames[row + off])] = (g, off)
                    row += n
                cache[filename] = (pf, index)
        return cache[filename]

    def _group(self, filename: str, g: int):
        """Decoded row-group table (small per-instance LRU)."""
        cache = self.__dict__.setdefault("_group_cache", OrderedDict())
        key = (filename, g)
        if key in cache:
            cache.move_to_end(key)
        else:
            pf, _ = self._meta(filename)
            cache[key] = pf.read_row_group(g)
            while len(cache) > _GROUP_CACHE_SIZE:
                cache.popitem(last=False)
        return cache[key]

    def _require_meta(self, filename: str, hint: str):
        """Shard metadata, or raise :class:`ModalityNotDownloaded` with a
        download hint naming the missing modality."""
        meta = self._meta(filename)
        if meta is None:
            raise ModalityNotDownloaded(
                f"{self.name}: {filename} is not present locally — "
                f"download with a modality selection that includes {hint}")
        return meta

    def _cell(self, filename: str, frame: int, column: str):
        """One cell of a shard, or None if the shard/frame is missing."""
        meta = self._meta(filename)
        if meta is None:
            return None
        loc = meta[1].get(frame)
        if loc is None:
            return None
        g, off = loc
        return self._group(filename, g).column(column)[off].as_py()

    def _image_bytes(self, filename: str, frame: int, column: str) -> bytes:
        meta = self._require_meta(filename, "this camera/modality")
        loc = meta[1].get(frame)
        if loc is None:
            raise FileNotFoundError(
                f"{self.name}: frame {frame} missing from {filename} "
                "(the camera dropped this frame at capture)")
        g, off = loc
        return self._group(filename, g).column(column)[off].as_py()["bytes"]

    @staticmethod
    def _decode(data: bytes) -> np.ndarray:
        return np.asarray(Image.open(io.BytesIO(data)))

    # ── discovery ───────────────────────────────────────────────────────────
    @cached_property
    def frames(self) -> list[int]:
        candidates = [layout.shard_filename("annotation"),
                      layout.shard_filename("pressure")]
        candidates += [layout.shard_filename("color", c) for c in ALL_CAMERAS]
        candidates += [layout.shard_filename("depth", c) for c in ALL_CAMERAS]
        candidates += [layout.shard_filename("mask", c) for c in ALL_CAMERAS]
        for filename in candidates:
            meta = self._meta(filename)
            if meta is not None:
                return sorted(meta[1])
        raise FileNotFoundError(
            f"{self.directory}: no shards found — was anything downloaded "
            "for this sequence?")

    @cached_property
    def cameras(self) -> list[str]:
        return [c for c in ALL_CAMERAS
                if (self.directory / layout.shard_filename("color", c)).exists()]

    # ── per-field loaders ────────────────────────────────────────────────────
    def load_rgb(self, index: int, camera: str) -> np.ndarray:
        return self._decode(self._image_bytes(
            layout.shard_filename("color", camera), index, "image"))

    def load_mask(self, index: int, camera: str) -> np.ndarray:
        return self._decode(self._image_bytes(
            layout.shard_filename("mask", camera), index, "mask"))

    def load_depth(self, index: int, camera: str) -> np.ndarray:
        """Depth map ``(512, 512) uint16`` millimetres for one camera.

        Raises ``FileNotFoundError`` if the camera dropped this frame at
        capture time (recorded as a missing shard row) or
        :class:`ModalityNotDownloaded` if the depth shard is absent.
        """
        return self._decode(self._image_bytes(
            layout.shard_filename("depth", camera), index, "depth"))

    def has_depth(self, camera: str) -> bool:
        """True if this sequence has a depth shard for ``camera``."""
        return (self.directory / layout.shard_filename("depth", camera)).exists()

    def load_force(self, index: int, reshape: bool = True) -> np.ndarray:
        filename = layout.shard_filename("pressure")
        self._require_meta(filename, "'pressure'")
        flat = self._cell(filename, index, "force")
        if flat is None:
            raise FileNotFoundError(
                f"{self.name}: no pressure row for frame {index}")
        arr = np.asarray(flat, dtype=np.float32)
        return arr.reshape(ARRAY_SHAPES["force"]) if reshape else arr

    def load_annotation(self, index: int) -> Annotation:
        filename = layout.shard_filename("annotation")
        meta = self._require_meta(filename, "'pose'")
        loc = meta[1].get(index)
        if loc is None:
            raise KeyError(f"Frame {index} not in sequence '{self.name}'.")
        g, off = loc
        table = self._group(filename, g)
        row = {c: table.column(c)[off].as_py() for c in table.column_names}

        def arr(key):
            v = row.get(key)
            return None if v is None else \
                np.asarray(v, dtype=np.float32).reshape(ARRAY_SHAPES[key])

        d: dict = {
            "has_annotation": bool(row["has_annotation"]),
            "hand_side": row["hand_side"],
        }
        if row.get("ego_R") is not None:
            d["ego_camera_pose"] = {"R": arr("ego_R").tolist(),
                                    "T": arr("ego_T").tolist()}
        for k in ("vertices", "joint_position", "betas", "full_pose",
                  "transl", "displacement", "normals"):
            v = arr(k)
            if v is not None:
                d[k] = v
        vv = row.get("visible_vertices")
        if vv is not None:
            stack = np.asarray(vv, dtype=np.float32).reshape(
                ARRAY_SHAPES["visible_vertices"]).astype(np.int64)
            # keyed by int camera number 1..7 (matches DATA.md row order)
            d["visible_vertices"] = {i + 1: stack[i] for i in range(stack.shape[0])}
        # pressure lives in its own shard; merge it into the annotation view
        pm = self._cell(layout.shard_filename("pressure"), index, "pressure_map")
        pr = self._cell(layout.shard_filename("pressure"), index, "pressure_map_range")
        if pm is not None:
            d["pressure_map"] = np.asarray(pm, np.float32).reshape(
                ARRAY_SHAPES["pressure_map"])
        if pr is not None:
            d["pressure_map_range"] = np.asarray(pr, np.float32)
        return Annotation.from_dict(d)

    def annotated_frames(self) -> list[int]:
        """Frame indices with ``has_annotation == True`` — reads two columns
        once instead of one full row per frame."""
        filename = layout.shard_filename("annotation")
        meta = self._require_meta(filename, "'pose'")
        pf = meta[0]
        t = pf.read(columns=["frame", "has_annotation"])
        fr = t.column("frame").to_pylist()
        ha = t.column("has_annotation").to_pylist()
        return sorted(int(f) for f, h in zip(fr, ha) if h)

    # ── pickling ─────────────────────────────────────────────────────────────
    def __getstate__(self):
        # keep instances light + picklable (DataLoader workers under spawn);
        # ParquetFile handles are re-opened lazily in each process
        state = self.__dict__.copy()
        state.pop("_meta_cache", None)
        state.pop("_group_cache", None)
        return state
