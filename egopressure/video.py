"""Sequence video export: all cameras + live sensor panels -> compact MP4.

Renders a whole sequence as a tiled video — every available camera with the
MANO mesh, skeleton, and measured pressure (white glow) overlaid, plus the
raw Sensel force grid and the hand-UV pressure map as side panels — and
streams frames directly into ffmpeg
(H.264, CRF-compressed, yuv420p). Memory use is constant in sequence length:
one composed frame lives in RAM at a time.

    from egopressure.video import save_video
    save_video(seq, "sequence.mp4")                    # all cameras
    save_video(seq, "ego.mp4", cameras=["d"])          # ego only

or from the shell::

    egopressure video p_001 p_001_press_palm_low_x5_right --out seq.mp4

Requires ``ffmpeg`` on PATH and matplotlib (colormaps only).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .constants import ALL_CAMERAS, DEPTH_VIS_MAX_MM, EGO_CAMERA
from .dataset import Sequence
from .registration import register_depth_to_color
from .senselpad import SENSEL_MAX_VIS_KPA, counts_to_kpa, warp_force_to_image
from .viewer import (
    FINGER_COLORS,
    FINGERS,
    MESH_ALPHA,
    MESH_RGB,
    MESH_SHADE_RANGE,
    PRESSURE_GLOW_ALPHA,
    _mano_faces,
    uv_pressure_panel,
)

_TILE_W, _TILE_H = 480, 270          # per-camera tile (16:9, downscaled)
_PANEL_W = 480                       # sensor panel column width

_LABEL_FONTS: dict = {}              # TrueType font cache, keyed by size
_PRESSURE_WARP_STRIDE = 8            # px subsampling for the force->image warp

def _label_font(size: int = 14):
    """A readable TrueType font for tile labels, cached per size; falls back
    to the PIL bitmap font when no system font is found."""
    if size not in _LABEL_FONTS:
        for cand in ("/System/Library/Fonts/Helvetica.ttc",             # macOS
                     "/Library/Fonts/Arial.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                     "DejaVuSans.ttf", "arial.ttf"):                # PATH/Windows
            try:
                _LABEL_FONTS[size] = ImageFont.truetype(cand, size)
                break
            except OSError:
                continue
        else:
            _LABEL_FONTS[size] = ImageFont.load_default()
    return _LABEL_FONTS[size]


def _label(img: np.ndarray, text: str, xy=(6, 4), size: int = 14) -> None:
    """Burn a small label into ``img`` (black backing strip)."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    x, y = xy
    font = _label_font(size)
    left, top, right, bottom = d.textbbox((x, y), text, font=font)
    d.rectangle([left - 4, top - 3, right + 4, bottom + 3], fill=(0, 0, 0, 160))
    d.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    img[:] = np.asarray(im)


def _label_bottom_left(img: np.ndarray, text: str, size: int = 20) -> None:
    """Burn a label into the lower-left corner of ``img``."""
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    font = _label_font(size)
    left, top, right, bottom = d.textbbox((0, 0), text, font=font)
    x = 10
    y = img.shape[0] - (bottom - top) - 8
    left, top, right, bottom = d.textbbox((x, y), text, font=font)
    d.rectangle([left - 5, top - 4, right + 5, bottom + 4], fill=(0, 0, 0, 160))
    d.text((x, y), text, font=font, fill=(255, 255, 255, 255))
    img[:] = np.asarray(im)


def _draw_skeleton(tile: np.ndarray, uv: np.ndarray) -> None:
    """MANO 21-joint skeleton, per-finger colours (in place, tile pixels)."""
    im = Image.fromarray(tile.astype(np.uint8))
    d = ImageDraw.Draw(im)
    for finger, chain in FINGERS.items():
        pts = uv[chain]
        if np.isnan(pts).any():
            continue
        hexc = FINGER_COLORS[finger].lstrip("#")
        col = tuple(int(hexc[i:i + 2], 16) for i in (0, 2, 4))
        d.line([tuple(p) for p in pts], fill=col, width=2, joint="curve")
        for p in pts:
            d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=col)
    tile[:] = np.asarray(im).astype(tile.dtype)


def _draw_mesh_surface(tile: np.ndarray, uv: np.ndarray, z: np.ndarray,
                       faces: np.ndarray, alpha: float = MESH_ALPHA) -> None:
    """Painter-sorted flat-shaded MANO surface, composited onto the tile."""
    overlay = Image.new("RGBA", (tile.shape[1], tile.shape[0]), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    tri_uv = uv[faces]                              # (F, 3, 2)
    ok = ~np.isnan(tri_uv).any(axis=(1, 2))
    tri_uv, tri_z = tri_uv[ok], z[faces[ok]].mean(axis=1)
    order = np.argsort(-tri_z)                      # back to front
    zmin, zmax = tri_z.min(), tri_z.max() + 1e-9
    a = int(alpha * 255)
    for i in order:
        shade = 1.0 - MESH_SHADE_RANGE * (tri_z[i] - zmin) / (zmax - zmin)
        col = tuple(int(c * shade) for c in MESH_RGB) + (a,)
        d.polygon([tuple(p) for p in tri_uv[i]], fill=col)
    out = Image.alpha_composite(
        Image.fromarray(tile).convert("RGBA"), overlay).convert("RGB")
    tile[:] = np.asarray(out)


def _cmap(name: str):
    # matplotlib is an optional extra (viz) — imported lazily on purpose so
    # the module (and the CLI's non-video commands) work without it
    import matplotlib

    return matplotlib.colormaps[name]


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Nearest-neighbour resize via indexing (no extra dependencies)."""
    ys = (np.linspace(0, img.shape[0] - 1, h)).astype(int)
    xs = (np.linspace(0, img.shape[1] - 1, w)).astype(int)
    return img[ys][:, xs]


def _draw_points(img: np.ndarray, uv: np.ndarray, color, radius: int = 1) -> None:
    """Scatter small dots into ``img`` in place (uv in tile pixels)."""
    h, w = img.shape[:2]
    ok = (~np.isnan(uv).any(1)) & (uv[:, 0] >= radius) & (uv[:, 0] < w - radius) \
        & (uv[:, 1] >= radius) & (uv[:, 1] < h - radius)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            img[uv[ok, 1].astype(int) + dv, uv[ok, 0].astype(int) + du] = color


def _camera_tile(seq: Sequence, fi: int, cam: str, ann, overlays,
                 modality: str = "rgb", depth_view: str = "registered",
                 mesh_style: str = "surface") -> np.ndarray:
    name = "Egocentric Camera" if cam == EGO_CAMERA else f"Camera {cam}"
    if modality == "depth":
        # default: registered into the color frame via the k4a calibration
        # (viewing-friendly, comparable to the RGB tiles); the STORED data
        # stays raw sensor-frame — registration is computed on the fly
        if depth_view == "registered":
            try:
                scale = max(1, round(seq.calibration(cam).image_size[0] / _TILE_W))
                d = register_depth_to_color(
                    seq, fi, cam, scale=scale).astype(np.float64)
                lab = f"{name} - depth registered to color [mm]"
            except FileNotFoundError:
                d = seq.load_depth(fi, cam).astype(np.float64)
                lab = f"{name} - depth (sensor frame, mm)"
        else:
            d = seq.load_depth(fi, cam).astype(np.float64)
            lab = f"{name} - depth (sensor frame, mm)"
        norm = np.clip(d / DEPTH_VIS_MAX_MM, 0, 1)
        img = _cmap("turbo")(norm)[..., :3] * 255
        img[d == 0] = 0
        out = _resize(img.astype(np.uint8), _TILE_W, _TILE_H)
        _label(out, lab)
        return out

    rgb = seq.load_rgb(fi, cam)
    H, W = rgb.shape[:2]
    sx, sy = _TILE_W / W, _TILE_H / H
    tile = _resize(rgb, _TILE_W, _TILE_H).astype(np.float64)

    cal = seq.calibration(cam)
    extr = ann.ego_extrinsic() if (cam == EGO_CAMERA and ann.ego_camera_pose) else None
    # draw order: rgb base -> mesh -> skeleton -> pressure (white, on top)
    if "mesh" in overlays and ann.has_annotation and ann.vertices is not None:
        uv = cal.project_world(ann.vertices, extrinsic=extr) * np.array([sx, sy])
        mano = (_mano_faces(ann.hand_side or "right")
                if mesh_style == "surface" else None)
        if mano is not None:
            z = cal.world_to_cam(ann.vertices, extrinsic=extr)[:, 2]
            tile_u8 = tile.astype(np.uint8)
            _draw_mesh_surface(tile_u8, uv, z, mano["faces_v"])
            tile = tile_u8.astype(np.float64)
        else:
            _draw_points(tile, uv, (0, 229, 255), radius=1)
    if ("skeleton" in overlays and ann.has_annotation
            and ann.joint_position is not None):
        j2 = cal.project_world(ann.joint_position, extrinsic=extr) \
            * np.array([sx, sy])
        _draw_skeleton(tile, j2)
    if "pressure" in overlays:
        kpa = warp_force_to_image(seq.load_force(fi), cal, (H, W),
                                  extrinsic=extr, stride=_PRESSURE_WARP_STRIDE)
        kpa = _resize(kpa, _TILE_W, _TILE_H)
        norm = np.clip(kpa / SENSEL_MAX_VIS_KPA, 0, 1)
        # white glow, alpha ~ sqrt(pressure): clearly visible over mesh/pad
        alpha = np.sqrt(norm)[..., None] * PRESSURE_GLOW_ALPHA
        tile = (1 - alpha) * tile + alpha * 255.0
    out = tile.astype(np.uint8)
    _label(out, name)
    return out


def _sensor_panel(seq: Sequence, fi: int, ann, height: int) -> np.ndarray:
    """Force grid (top) + UV pressure map (bottom), colormapped, stacked."""
    half = height // 2
    kpa = counts_to_kpa(seq.load_force(fi))
    force_img = _cmap("inferno")(np.clip(kpa / SENSEL_MAX_VIS_KPA, 0, 1))[..., :3] * 255
    top = _resize(force_img.astype(np.uint8), _PANEL_W, half)

    _label(top, "Sensel force grid [kPa]")

    bh = height - half
    bottom = uv_pressure_panel(ann, _PANEL_W, bh).copy()
    _label(bottom, "Hand UV pressure map")
    return np.vstack([top, bottom])


def save_video(
    seq: Sequence,
    out_path: str | Path,
    cameras: list[str] | None = None,
    overlays: tuple[str, ...] = ("mesh", "skeleton", "pressure"),
    modality: str = "rgb",
    depth_view: str = "registered",
    mesh_style: str = "surface",
    fps: int = 30,
    crf: int = 28,
    frames: list[int] | None = None,
) -> Path:
    """Render the sequence to an H.264 MP4.

    Layout: available cameras tiled in rows of four (ego first), with the raw
    force grid and UV pressure map as a right-hand sensor column.

    Args:
        seq: The sequence (works for Hub-downloaded and local data alike).
        out_path: Output ``.mp4``.
        cameras: Camera subset (default: all with imagery).
        overlays: Any of ``"mesh"``, ``"skeleton"``, ``"pressure"`` (all on
            by default; pressure renders last as a white glow).
            Ignored for depth tiles.
        modality: ``"rgb"`` (default) or ``"depth"``; depth tiles are
            colormapped (turbo, 0 to ``DEPTH_VIS_MAX_MM``) and frames a
            camera dropped render black.
        depth_view: ``"registered"`` (default; reprojected into the color
            frame via the k4a calibration — comparable to the RGB tiles) or
            ``"sensor"`` (the raw 512x512 sensor-frame maps as stored).
        mesh_style: hand overlay as shaded MANO ``"surface"`` (default;
            needs the user-provided MANO files, falls back to points) or
            vertex ``"points"``.
        fps: Output frame rate (native capture is 30).
        crf: H.264 quality (higher = smaller; 28 is a compact default).
        frames: Explicit frame subset (default: all frames).
    """
    if cameras is not None:
        cams = list(cameras)
    elif modality == "depth":
        cams = [c for c in ALL_CAMERAS if seq.has_depth(c)]
    else:
        cams = list(seq.cameras)
    if not cams:
        raise ValueError(f"no cameras with {modality!r} data in this download")
    frame_list = frames if frames is not None else seq.frames
    cols = min(4, len(cams))
    rows = (len(cams) + cols - 1) // cols
    grid_w, grid_h = cols * _TILE_W, rows * _TILE_H
    W, H = grid_w + _PANEL_W, grid_h
    W += W % 2
    H += H % 2                                     # yuv420p needs even dims

    out_path = Path(out_path)
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
         "-pix_fmt", "yuv420p", str(out_path)],
        stdin=subprocess.PIPE)

    gesture = re.sub(r"^p_\d+_", "", seq.name).replace("_", " ")
    try:
        for fi in frame_list:
            ann = seq.load_annotation(fi)
            canvas = np.zeros((H, W, 3), dtype=np.uint8)
            for idx, cam in enumerate(cams):
                r, c = divmod(idx, cols)
                try:
                    tile = _camera_tile(seq, fi, cam, ann, overlays,
                                        modality, depth_view, mesh_style)
                except FileNotFoundError:
                    tile = np.zeros((_TILE_H, _TILE_W, 3), dtype=np.uint8)
                canvas[r * _TILE_H:(r + 1) * _TILE_H,
                       c * _TILE_W:(c + 1) * _TILE_W] = tile
            canvas[0:grid_h, grid_w:grid_w + _PANEL_W] = \
                _sensor_panel(seq, fi, ann, grid_h)
            _label_bottom_left(canvas, gesture)
            ff.stdin.write(canvas.tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
    if ff.returncode != 0:
        raise RuntimeError("ffmpeg failed while encoding the video")
    return out_path
