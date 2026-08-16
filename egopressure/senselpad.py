"""Sensel Morph touchpad geometry + physical-unit conversion.

Ported from the official EgoPressureVis release
(``models/sensel_projection.py`` / ``models/pressure_util.py``), which defines
the authoritative pad placement in the world frame:

* The pad is the **z = 0 plane**, centred at the origin, spanning
  ``x ∈ ±0.120 m``, ``y ∈ ±0.06875 m`` — the 240 × 137.5 mm pad rectangle
  the annotation pipeline is calibrated against.
* Sensor cells map to the pad corners as
  ``(col, row): (185,0) (185,105) (0,105) (0,0)`` — i.e. columns run along +x,
  rows along +y.
* Raw ``force`` counts convert to physical units via
  ``newtons = counts / 1736`` and ``kPa = N / pitch²/ 1000`` with a 1.25 mm
  cell pitch.

With a camera's intrinsics + extrinsics this yields a pad→image homography, so
the raw sensor grid can be rendered into **any** camera view (validated: the
projected pad outline matches the visible touchpad in ego and static views).
"""

from __future__ import annotations

import numpy as np

# ── official constants ──────────────────────────────────────────────────────
SENSEL_W = 0.240            # active width  (m), along x / columns
SENSEL_H = 0.1375           # active height (m), along y / rows
SENSEL_COUNTS_TO_NEWTON = 1736.0
SENSEL_PIXEL_PITCH = 0.00125     # m per cell
SENSEL_MAX_VIS_KPA = 20.0        # display normalisation used by the authors

#: Pad corners in the world frame (z = 0 plane), official ordering.
CORNERS_3D = np.array(
    [[+SENSEL_W / 2, -SENSEL_H / 2, 0.0],
     [+SENSEL_W / 2, +SENSEL_H / 2, 0.0],
     [-SENSEL_W / 2, +SENSEL_H / 2, 0.0],
     [-SENSEL_W / 2, -SENSEL_H / 2, 0.0]], dtype=np.float64)

#: Matching sensor-grid corners as (col, row).
CORNERS_2D = np.array([[185, 0], [185, 105], [0, 105], [0, 0]], dtype=np.float64)


def counts_to_kpa(counts: np.ndarray) -> np.ndarray:
    """Convert raw ``force`` counts to kilopascals (official conversion)."""
    newtons = np.asarray(counts, dtype=np.float64) / SENSEL_COUNTS_TO_NEWTON
    return newtons / (SENSEL_PIXEL_PITCH ** 2) / 1000.0


def _dlt_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3x3 homography mapping ``src`` -> ``dst`` (4-point DLT, no OpenCV dep)."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]


def pad_homography(calibration, extrinsic: np.ndarray | None = None) -> np.ndarray:
    """Homography mapping sensor cells (col, row) to pixels in a camera view.

    Args:
        calibration: A :class:`~egopressure.calibration.CameraCalibration`.
        extrinsic: Per-frame 4x4 world->camera pose in metres for the ego
            camera; ``None`` for static cameras (uses the config extrinsic).
    """
    corners_px = calibration.project_world(CORNERS_3D, extrinsic=extrinsic)
    return _dlt_homography(CORNERS_2D, corners_px)


def pad_outline(calibration, extrinsic: np.ndarray | None = None) -> np.ndarray:
    """``(5, 2)`` closed pixel polyline of the pad outline in a camera view."""
    c = calibration.project_world(CORNERS_3D, extrinsic=extrinsic)
    return np.vstack([c, c[:1]])


def warp_force_to_image(
    force_counts: np.ndarray,
    calibration,
    image_shape: tuple[int, int],
    extrinsic: np.ndarray | None = None,
    stride: int = 1,
) -> np.ndarray:
    """Inverse-warp the raw force grid into image space, in kPa.

    Args:
        force_counts: ``(105, 185)`` raw sensor counts.
        calibration: Camera calibration.
        image_shape: ``(H, W)`` of the target image.
        extrinsic: Ego per-frame pose (metres) or ``None`` for static cameras.
        stride: Compute every ``stride``-th pixel (output stays full-size,
            nearest-filled) for speed.

    Returns:
        ``(H, W)`` float kPa image (0 outside the pad).
    """
    H_img, W_img = image_shape
    kpa = counts_to_kpa(force_counts)
    Hm = pad_homography(calibration, extrinsic)
    Hinv = np.linalg.inv(Hm)

    ys, xs = np.mgrid[0:H_img:stride, 0:W_img:stride]
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
    src = Hinv @ pts
    cx = (src[0] / src[2]).reshape(xs.shape)
    cy = (src[1] / src[2]).reshape(xs.shape)
    valid = (cx >= 0) & (cx < kpa.shape[1]) & (cy >= 0) & (cy < kpa.shape[0])

    small = np.zeros(xs.shape, dtype=np.float64)
    vi = valid.nonzero()
    small[vi] = kpa[cy[vi].astype(int), cx[vi].astype(int)]
    if stride == 1:
        return small
    return np.kron(small, np.ones((stride, stride)))[:H_img, :W_img]


def overlay_force_on_image(
    image: np.ndarray,
    force_counts: np.ndarray,
    calibration,
    extrinsic: np.ndarray | None = None,
    max_kpa: float = SENSEL_MAX_VIS_KPA,
    cmap: str = "inferno",
    stride: int = 2,
) -> np.ndarray:
    """Blend the projected pressure onto an RGB image (alpha ∝ pressure).

    Returns a new ``(H, W, 3) uint8`` image.
    """
    # matplotlib is an optional extra (viz) — imported lazily on purpose
    import matplotlib

    kpa = warp_force_to_image(force_counts, calibration, image.shape[:2],
                              extrinsic=extrinsic, stride=stride)
    norm = np.clip(kpa / max_kpa, 0.0, 1.0)
    rgba = matplotlib.colormaps[cmap](norm)
    out = image.astype(np.float64).copy()
    alpha = norm[..., None]
    out = (1 - alpha) * out + alpha * rgba[..., :3] * 255.0
    return out.astype(np.uint8)
