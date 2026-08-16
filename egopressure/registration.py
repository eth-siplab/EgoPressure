"""Depth → color registration using the shipped k4a factory calibrations.

Stored depth maps are **sensor-frame** measurements (512×512, unresampled;
coverage spans the region co-visible with the camera's color image). This
module reprojects them into a camera's color frame when you want
pixel-aligned RGB-D:

    from egopressure.registration import register_depth_to_color
    reg = register_depth_to_color(seq, frame_index, camera="d", scale=4)

The registration is a z-buffered point splat: each valid depth pixel is
unprojected with the depth camera's intrinsics (normalised Brown–Conrady from
the k4a calibration), transformed depth→color (``Rt``, millimetres), and
projected with the released color intrinsics. Output pixels without a sample
are 0 — increase ``splat`` or decrease ``scale`` for denser coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import Sequence

_DIRS_CACHE: dict = {}    # per-camera-model undistorted ray grids


def load_k4a_calibration(seq: Sequence, camera: str) -> dict:
    """The k4a factory calibration JSON for one camera of this sequence."""
    cfg = seq.config_path
    if cfg is None:
        raise FileNotFoundError(f"{seq.name}: no config path known")
    k4a = Path(cfg).parent / f"{seq.name}_k4a" / f"cam-{camera}.k4a_calibration.json"
    if not k4a.exists():
        raise FileNotFoundError(
            f"{k4a} missing — was the download made with include_configs=True?")
    return json.loads(k4a.read_text())


def _depth_intrinsics(cal: dict, img_w: int, img_h: int):
    cams = cal["CalibrationInformation"]["Cameras"]
    depth = next(c for c in cams if c["Location"].endswith("D0"))
    color = next(c for c in cams if c["Location"].endswith("PV0"))
    p = depth["Intrinsics"]["ModelParameters"]
    K = {"cx": p[0] * img_w, "cy": p[1] * img_h,
         "fx": p[2] * img_w, "fy": p[3] * img_h,
         "k": (p[4], p[5], p[6], p[7], p[8], p[9]),
         "p1": p[13], "p2": p[12]}
    # each camera's Rt maps points FROM the depth (D0, reference) frame INTO
    # that camera's frame, translation in metres — so the color entry gives
    # depth -> color directly; convert T to millimetres to match depth units
    Rt = (np.array(color["Rt"]["Rotation"]).reshape(3, 3),
          np.array(color["Rt"]["Translation"]) * 1000.0)
    return K, Rt



def _undistort_dirs(K: dict, w: int, h: int) -> np.ndarray:
    """Per-pixel unit ray directions for the distorted depth camera (iterative
    Brown–Conrady inversion, adequate to sub-pixel for the k4a models).
    Cached per camera model — the grid is identical for every frame."""
    key = (K["cx"], K["cy"], K["fx"], K["fy"], K["k"], K["p1"], K["p2"], w, h)
    if key in _DIRS_CACHE:
        return _DIRS_CACHE[key]
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    xd = (u - K["cx"]) / K["fx"]
    yd = (v - K["cy"]) / K["fy"]
    x, y = xd.copy(), yd.copy()
    k1, k2, k3, k4, k5, k6 = K["k"]
    for _ in range(6):                              # fixed-point iteration
        r2 = x * x + y * y
        radial = (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3) / \
                 (1 + k4 * r2 + k5 * r2**2 + k6 * r2**3)
        dx = 2 * K["p1"] * x * y + K["p2"] * (r2 + 2 * x * x)
        dy = K["p1"] * (r2 + 2 * y * y) + 2 * K["p2"] * x * y
        x = (xd - dx) / radial
        y = (yd - dy) / radial
    dirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    _DIRS_CACHE[key] = dirs
    return dirs


def register_depth_to_color(
    seq: Sequence,
    frame_index: int,
    camera: str = "d",
    scale: int = 4,
    splat: int = 2,
) -> np.ndarray:
    """Reproject a raw depth map into the camera's color frame.

    Args:
        seq: The sequence (needs the k4a configs, downloaded by default).
        frame_index: Frame number.
        camera: Camera token (``"d"``, ``"1"``..``"7"``).
        scale: Output downscale factor vs the full color resolution (4 →
            480×270 for the ego camera). Depth has far fewer samples than
            color pixels, so a downscaled target avoids a sparse result.
        splat: Half-width of the square splat per point (fills small holes).

    Returns:
        ``(H/scale, W/scale)`` float32 depth in **millimetres**, in the color
        camera's (undistorted pinhole) frame; 0 where no sample landed.
    """
    depth = seq.load_depth(frame_index, camera).astype(np.float64)
    h, w = depth.shape
    cal = load_k4a_calibration(seq, camera)
    K, (R, T) = _depth_intrinsics(cal, w, h)

    dirs = _undistort_dirs(K, w, h)                 # cached per camera model
    valid = depth > 0
    pts_d = dirs[valid] * depth[valid][:, None]     # depth-cam frame, mm
    pts_c = pts_d @ R.T + T                         # color-cam frame, mm

    ccal = seq.calibration(camera)
    W_out = int(ccal.image_size[0]) // scale
    H_out = int(ccal.image_size[1]) // scale
    z = pts_c[:, 2]
    front = z > 1e-3
    u = (ccal.fx * pts_c[front, 0] / z[front] + ccal.cx) / scale
    v = (ccal.fy * pts_c[front, 1] / z[front] + ccal.cy) / scale
    z = z[front]

    out = np.full((H_out, W_out), np.inf, dtype=np.float64)
    ui, vi = u.astype(int), v.astype(int)
    for du in range(-splat, splat + 1):
        for dv in range(-splat, splat + 1):
            uu, vv = ui + du, vi + dv
            ok = (uu >= 0) & (uu < W_out) & (vv >= 0) & (vv < H_out)
            # z-buffer: keep the nearest sample per output pixel
            np.minimum.at(out, (vv[ok], uu[ok]), z[ok])
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)


def world_points_from_depth(
    seq: Sequence,
    frame_index: int,
    camera: str = "d",
    scale: int = 4,
    stride: int = 1,
    extrinsic: np.ndarray | None = None,
) -> np.ndarray:
    """Raw sensor-frame depth -> **world-frame point cloud** (metres).

    Convenience wrapper: registers the depth map into the camera's color
    frame, then back-projects with the released color intrinsics and the
    camera's pose. For the ego camera the per-frame annotated pose is loaded
    automatically (or pass ``extrinsic``); static cameras use their fixed
    ``ModelViewMatrix``.

    Returns:
        ``(N, 3)`` world points in metres (the touchpad plane is ``z = 0``,
        +z pointing down through the pad).
    """
    reg = register_depth_to_color(seq, frame_index, camera, scale=scale)
    cal = seq.calibration(camera)
    if camera == "d" and extrinsic is None:
        extrinsic = seq.load_annotation(frame_index).ego_extrinsic()
    return cal.unproject_depth(reg, extrinsic=extrinsic, stride=stride)
