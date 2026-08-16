"""Per-frame annotation container for EgoPressure.

Typed access over one annotation-shard row, with light validation and a
couple of convenience helpers (denormalised pressure, homogeneous ego pose).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Annotation:
    """MANO hand pose, geometry, and pressure ground truth for one frame.

    Only :attr:`has_annotation`, :attr:`ego_camera_pose` and :attr:`hand_side`
    are always present. Every other field is ``None`` on **unannotated** frames
    (pre-contact / dropped labels), i.e. whenever ``has_annotation is False``.
    Guard with :attr:`has_annotation` (or :meth:`require`) before using them.

    Attributes:
        vertices: ``(778, 3)`` MANO mesh vertices (metres, rig/world frame).
        joint_position: ``(21, 3)`` hand joints.
        transl: ``(3,)`` global MANO translation.
        betas: ``(1, 10)`` MANO shape parameters.
        full_pose: ``(48,)`` MANO pose — 16 joints x 3 axis-angle
            (global orientation + 15 finger joints).
        displacement: ``(778, 1)`` per-vertex normal displacement.
        normals: ``(778, 3)`` per-vertex normals.
        pressure_map: ``(224, 224, 1)`` **normalised** pressure baked onto the
            hand UV layout. Multiply by :meth:`pressure_scale` for real units.
        pressure_map_range: ``(2,)`` ``[min, max]`` used to normalise the map.
        visible_vertices: ``{static_cam: (778,) int}`` per-camera vertex
            visibility indicator (**int** keys ``1``..``7``).
        ego_camera_pose: ``{"R": 3x3, "T": 3}`` egocentric camera extrinsics.
        hand_side: ``"left"`` or ``"right"``.
        has_annotation: ``False`` for pre-contact / unlabelled frames.
    """

    ego_camera_pose: dict
    hand_side: str
    has_annotation: bool
    raw: dict  # the source field dict, for forward compatibility
    vertices: np.ndarray | None = None
    joint_position: np.ndarray | None = None
    transl: np.ndarray | None = None
    betas: np.ndarray | None = None
    full_pose: np.ndarray | None = None
    displacement: np.ndarray | None = None
    normals: np.ndarray | None = None
    pressure_map: np.ndarray | None = None
    pressure_map_range: np.ndarray | None = None
    visible_vertices: dict | None = None

    # ── constructors ───────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict) -> Annotation:
        """Build from one annotation-shard row (missing fields become None)."""
        return cls(
            ego_camera_pose=d.get("ego_camera_pose", {}),
            hand_side=d.get("hand_side", ""),
            has_annotation=bool(d.get("has_annotation", True)),
            raw=d,
            vertices=d.get("vertices"),
            joint_position=d.get("joint_position"),
            transl=d.get("transl"),
            betas=d.get("betas"),
            full_pose=d.get("full_pose"),
            displacement=d.get("displacement"),
            normals=d.get("normals"),
            pressure_map=d.get("pressure_map"),
            pressure_map_range=d.get("pressure_map_range"),
            visible_vertices=d.get("visible_vertices"),
        )

    def require(self) -> Annotation:
        """Return ``self`` if annotated, else raise — handy for fail-fast loops.

        Raises:
            ValueError: If ``has_annotation`` is ``False``.
        """
        if not self.has_annotation:
            raise ValueError("Frame is unannotated (has_annotation is False).")
        return self

    # ── convenience ────────────────────────────────────────────────────────
    def pressure_scale(self) -> float:
        """Return ``max`` of :attr:`pressure_map_range` (multiply the normalised
        ``pressure_map`` by this to recover physical units)."""
        return float(self.pressure_map_range[1])

    def denormalized_pressure_map(self) -> np.ndarray:
        """``pressure_map`` rescaled to physical units, clipped at zero."""
        return np.clip(self.pressure_map, 0.0, None) * self.pressure_scale()

    def ego_extrinsic(self) -> np.ndarray:
        """4x4 homogeneous egocentric camera pose from ``ego_camera_pose``."""
        R = np.asarray(self.ego_camera_pose["R"], dtype=np.float64).reshape(3, 3)
        T = np.asarray(self.ego_camera_pose["T"], dtype=np.float64).reshape(3)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = T
        return M
