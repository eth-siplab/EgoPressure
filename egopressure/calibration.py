"""Camera calibration + geometry for EgoPressure cameras.

Each camera has pinhole intrinsics, the Azure Kinect distortion coefficients, a
depth->color transform, and (for the 7 **static** cameras) a fixed world->camera
``ModelViewMatrix``. The egocentric camera moves, so its world pose is supplied
per frame from the annotation (``Annotation.ego_extrinsic()``) instead.

Conventions (all empirically validated against the data):
  * Mesh vertices and the ego pose are in **metres**; ``ModelViewMatrix`` is in
    **millimetres** (so world points are scaled x1000 before applying it).
  * Distributed images are **undistorted**, so projection uses the pinhole model
    (no distortion) by default. The stored ``dist`` coefficients describe the
    original lens and are kept only for reference.
  * The ego pose maps world->camera as ``P_cam = R @ P_world + T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# world points (metres) are scaled by this before applying a millimetre-space
# ModelViewMatrix (the static-camera extrinsic convention in the config).
_MVM_UNIT_SCALE = 1000.0


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsics, distortion, and extrinsics for a single camera.

    Attributes:
        camera: On-disk camera token (``"d"`` for ego, ``"1"``..``"7"`` static).
        fx, fy: Focal lengths in pixels.
        cx, cy: Principal point in pixels.
        dist: OpenCV-ordered distortion coefficients
            ``[k1, k2, p1, p2, k3, k4, k5, k6]`` (original lens; images shipped
            undistorted).
        depth_to_color: 4x4 intra-sensor depth->color transform.
        model_view_matrix: 4x4 world->camera extrinsic in **millimetres** for
            static cameras; ``None`` for the ego camera (per-frame pose instead).
        image_size: ``(width, height)`` of the color image in pixels.
        serial_no: Hardware serial number of the sensor.
    """

    camera: str
    fx: float
    fy: float
    cx: float
    cy: float
    dist: np.ndarray            # shape (8,)
    depth_to_color: np.ndarray  # shape (4, 4)
    model_view_matrix: np.ndarray | None  # shape (4, 4) or None (ego)
    image_size: tuple[int, int]
    serial_no: str

    @property
    def is_ego(self) -> bool:
        """True for the moving egocentric camera (no fixed extrinsic)."""
        return self.model_view_matrix is None

    # ── constructors ───────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, camera: str, entry: dict) -> CameraCalibration:
        """Build from a single ``camera_calibrations`` entry of the config JSON."""
        dist = np.array(
            [
                entry["k1"], entry["k2"], entry["p1"], entry["p2"],
                entry["k3"], entry["k4"], entry["k5"], entry["k6"],
            ],
            dtype=np.float64,
        )
        mvm = entry.get("ModelViewMatrix")
        return cls(
            camera=camera,
            fx=float(entry["fx"]),
            fy=float(entry["fy"]),
            cx=float(entry["cx"]),
            cy=float(entry["cy"]),
            dist=dist,
            depth_to_color=np.array(entry["DepthToColor"], dtype=np.float64),
            model_view_matrix=(np.array(mvm, dtype=np.float64)
                               if mvm is not None else None),
            image_size=(int(entry["ImageSizeX"]), int(entry["ImageSizeY"])),
            serial_no=str(entry.get("SerialNo", "")),
        )

    # ── derived quantities ─────────────────────────────────────────────────
    @property
    def K(self) -> np.ndarray:
        """3x3 pinhole intrinsic matrix."""
        return np.array(
            [[self.fx, 0.0, self.cx],
             [0.0, self.fy, self.cy],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def project(self, points_cam: np.ndarray,
                apply_distortion: bool = False) -> np.ndarray:
        """Project 3D points **already expressed in this camera's frame** to pixels.

        Note:
            ``points_cam`` must already be in this camera's coordinate system
            (use :meth:`world_to_cam` / :meth:`project_world` for world-frame
            points). Points behind the camera (``z <= 0``) yield ``nan``.

        Args:
            points_cam: ``(N, 3)`` points in camera coordinates (metres).
            apply_distortion: Apply the rational distortion model if ``True``.
                Released images are **undistorted**, so the default is
                ``False`` (matching :meth:`project_world`).

        Returns:
            ``(N, 2)`` pixel coordinates.
        """
        points_cam = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
        z = points_cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            xp = points_cam[:, 0] / z
            yp = points_cam[:, 1] / z

        if apply_distortion:
            k1, k2, p1, p2, k3, k4, k5, k6 = self.dist
            r2 = xp * xp + yp * yp
            r4, r6 = r2 * r2, r2 * r2 * r2
            radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) \
                / (1 + k4 * r2 + k5 * r4 + k6 * r6)
            xd = xp * radial + 2 * p1 * xp * yp + p2 * (r2 + 2 * xp * xp)
            yd = yp * radial + p1 * (r2 + 2 * yp * yp) + 2 * p2 * xp * yp
        else:
            xd, yd = xp, yp

        u = self.fx * xd + self.cx
        v = self.fy * yd + self.cy
        uv = np.stack([u, v], axis=-1)
        uv[z <= 0] = np.nan
        return uv

    # ── world <-> pixel ────────────────────────────────────────────────────
    def world_to_cam(
        self, points_world: np.ndarray, extrinsic: np.ndarray | None = None
    ) -> np.ndarray:
        """Transform world points (metres) into this camera's frame (metres).

        Args:
            points_world: ``(N, 3)`` points in the world/rig frame, in metres.
            extrinsic: 4x4 world->camera pose in **metres** (required for the ego
                camera; pass ``Annotation.ego_extrinsic()``). For static cameras
                leave ``None`` to use the config ``ModelViewMatrix``.

        Returns:
            ``(N, 3)`` points in camera coordinates, metres.
        """
        P = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        if extrinsic is not None:
            M = np.asarray(extrinsic, dtype=np.float64)
            scale = 1.0
        elif self.model_view_matrix is not None:
            M = self.model_view_matrix
            scale = _MVM_UNIT_SCALE            # world metres -> mm for the MVM
        else:
            raise ValueError(
                f"Camera {self.camera!r} is the ego camera: pass its per-frame "
                "`extrinsic` (Annotation.ego_extrinsic())."
            )
        Ph = np.c_[P * scale, np.ones(len(P))]
        cam = (M @ Ph.T).T[:, :3]
        return cam / scale                      # back to metres

    def project_world(
        self,
        points_world: np.ndarray,
        extrinsic: np.ndarray | None = None,
        apply_distortion: bool = False,
    ) -> np.ndarray:
        """Project world points (metres) to pixels.

        Handles the ego (per-frame ``extrinsic``) and static (config
        ``ModelViewMatrix``) cases uniformly. Distortion is off by default
        because the released images are undistorted.

        Returns:
            ``(N, 2)`` pixel coordinates (``nan`` for points behind the camera).
        """
        return self.project(
            self.world_to_cam(points_world, extrinsic),
            apply_distortion=apply_distortion,
        )

    def unproject_depth(
        self,
        depth_mm: np.ndarray,
        extrinsic: np.ndarray | None = None,
        stride: int = 1,
        near: float = 0.1,
        far: float = 3.0,
        return_uv: bool = False,
    ):
        """Back-project a depth map **registered to this camera's color
        frame** into the world.

        Accepts the full color resolution or any uniform integer downscale of
        it (as produced by
        :func:`egopressure.registration.register_depth_to_color`); the
        intrinsics are rescaled to match automatically. Raw EgoPressure depth
        maps (512x512) are in the **depth-sensor frame** and are rejected —
        register them first, or call
        :func:`egopressure.registration.world_points_from_depth` which does
        both steps.

        Args:
            depth_mm: ``(H, W)`` registered depth in millimetres (0 = invalid).
            extrinsic: 4x4 world->camera pose in metres (ego: pass the per-frame
                pose; static: leave ``None`` to use the ``ModelViewMatrix``).
            stride: Pixel subsampling factor (``>1`` thins the cloud).
            near, far: Keep points with ``near < z < far`` metres.
            return_uv: Also return the source ``(N, 2)`` pixel coordinates.

        Returns:
            ``(N, 3)`` world points in metres, or ``(points, uv)`` if
            ``return_uv``.
        """
        depth = np.asarray(depth_mm, dtype=np.float64)
        H, W = depth.shape
        W0, H0 = int(self.image_size[0]), int(self.image_size[1])
        sx, sy = W0 / W, H0 / H
        if abs(sx - sy) > 1e-6 or abs(sx - round(sx)) > 1e-6:
            raise ValueError(
                f"depth map {W}x{H} is not aligned to camera {self.camera!r}'s "
                f"color frame ({W0}x{H0}) or an integer downscale of it. Raw "
                "EgoPressure depth maps are in the depth-sensor frame — "
                "register first (egopressure.registration."
                "register_depth_to_color) or use egopressure.registration."
                "world_points_from_depth().")
        sc = round(sx)
        fx, fy, cx, cy = self.fx / sc, self.fy / sc, self.cx / sc, self.cy / sc
        vs, us = np.mgrid[0:H:stride, 0:W:stride]
        z = depth[::stride, ::stride] / 1000.0          # mm -> m
        keep = (z > near) & (z < far)
        u = us[keep].astype(np.float64)
        v = vs[keep].astype(np.float64)
        z = z[keep]
        # pinhole back-projection (undistorted image)
        x = (u - cx) / fx * z
        y = (v - cy) / fy * z
        cam = np.stack([x, y, z], axis=1)
        # camera -> world:  inverse of  P_cam = R @ P_world + T
        if extrinsic is not None:
            M = np.asarray(extrinsic, dtype=np.float64)
            R, T = M[:3, :3], M[:3, 3]
            world = (cam - T) @ R                        # R is orthonormal
        elif self.model_view_matrix is not None:
            M = self.model_view_matrix
            R, T = M[:3, :3], M[:3, 3]
            world = ((cam * _MVM_UNIT_SCALE) - T) @ R / _MVM_UNIT_SCALE
        else:
            raise ValueError(f"Camera {self.camera!r} needs an ego `extrinsic`.")
        if return_uv:
            return world, np.stack([u, v], axis=1)
        return world
