"""Notebook-friendly visual explorer for EgoPressure frames.

``FrameViewer`` is the visual analogue of the project webpage's interactive
viewer, but driven by the real data through the validated calibration: pick a
camera, a base modality (RGB or depth), and overlays (skeleton / mesh /
pressure), and it renders them together with the raw Sensel grid and the UV
pressure map. Everything is computed live from the annotation + calibration, so it
works for any frame, any camera, and doubles as a geometry sanity check.

Requires ``matplotlib``. Interactive scrubbing additionally uses ``ipywidgets``
when available (falls back to a static preview otherwise).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from collections.abc import Sequence as Seq

import numpy as np
from PIL import Image, ImageDraw

from .constants import EGO_CAMERA
from .dataset import ModalityNotDownloaded, Sequence
from .mano import load_mano_uv
from .registration import register_depth_to_color
from .senselpad import SENSEL_MAX_VIS_KPA, counts_to_kpa, warp_force_to_image

# MANO 21-joint skeleton, per-finger colours (matches the web viewer palette).
FINGERS = {
    "thumb":  [0, 1, 2, 3, 4],
    "index":  [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring":   [0, 13, 14, 15, 16],
    "pinky":  [0, 17, 18, 19, 20],
}
FINGER_COLORS = {
    "thumb": "#e6194b", "index": "#3cb44b", "middle": "#4363d8",
    "ring": "#f58231", "pinky": "#911eb4",
}
OVERLAYS = ("skeleton", "mesh", "pressure")

_MANO_CACHE: dict = {}
_UV_CHART_CACHE: dict = {}
_WARNED = {"mano": False}

MESH_RGB = (206, 206, 208)           # neutral-gray MANO surface base colour
MESH_ALPHA = 0.75                    # mesh surface opacity
MESH_SHADE_RANGE = 0.45              # depth-shading strength of the surface
PRESSURE_GLOW_ALPHA = 0.9            # peak opacity of the white pressure glow
CHART_GRAY = 165                     # UV wireframe intensity on white panels


def _mano_faces(hand_side: str):
    """MANO topology if the user installed the MANO files, else None (warn once)."""
    if hand_side not in _MANO_CACHE:
        try:
            _MANO_CACHE[hand_side] = load_mano_uv(hand_side)
        except FileNotFoundError:
            _MANO_CACHE[hand_side] = None
            if not _WARNED["mano"]:
                warnings.warn(
                    "MANO model files not found - rendering the hand as vertex "
                    "points and the UV panel without the hand chart. Install "
                    "them for true mesh-surface rendering (mano_models/README.md).")
                _WARNED["mano"] = True
    return _MANO_CACHE[hand_side]


def _uv_chart(hand_side: str, w: int, h: int):
    """Boolean edge mask of the MANO UV layout (cached), or None without MANO."""
    key = (hand_side, w, h)
    if key not in _UV_CHART_CACHE:
        mano = _mano_faces(hand_side)
        if mano is None:
            _UV_CHART_CACHE[key] = None
        else:
            im = Image.new("L", (w, h), 0)
            d = ImageDraw.Draw(im)
            uvpx = np.stack([mano["uv"][:, 0] * (w - 1),
                             (1.0 - mano["uv"][:, 1]) * (h - 1)], 1)
            for tri in mano["faces_vt"]:
                p = [tuple(uvpx[i]) for i in tri]
                d.line([p[0], p[1], p[2], p[0]], fill=255)
            _UV_CHART_CACHE[key] = np.asarray(im) > 0
    return _UV_CHART_CACHE[key]


def uv_pressure_panel(ann, w: int, h: int) -> np.ndarray:
    """White hand-UV pressure panel — the toolkit's standard UV visualization.

    Args:
        ann: :class:`~egopressure.annotation.Annotation` or ``None``.
        w, h: Output size in pixels.

    Returns:
        ``(h, w, 3) uint8``: white background, gray MANO-UV wireframe (when
        the MANO files are installed), pressure blended in inferno on a fixed
        kPa scale.
    """
    # matplotlib is an optional extra (viz) — imported lazily on purpose
    import matplotlib

    if ann is not None and ann.has_annotation and ann.pressure_map is not None:
        pm = ann.denormalized_pressure_map()[..., 0]
        norm = np.clip(counts_to_kpa(pm) / SENSEL_MAX_VIS_KPA, 0, 1)
        ys = (np.linspace(0, norm.shape[0] - 1, h)).astype(int)
        xs = (np.linspace(0, norm.shape[1] - 1, w)).astype(int)
        norm = norm[ys][:, xs]
        a = np.sqrt(norm)[..., None]
        panel = ((1 - a) * 255.0
                 + a * matplotlib.colormaps["inferno"](norm)[..., :3] * 255)
        panel = panel.astype(np.uint8)
    else:
        panel = np.full((h, w, 3), 255, dtype=np.uint8)
    chart = _uv_chart((ann.hand_side if ann is not None else None) or "right",
                      w, h)
    if chart is not None:
        panel[chart] = np.minimum(panel[chart], CHART_GRAY)
    return panel


class FrameViewer:
    """Render EgoPressure frames with selectable camera / modality / overlays.

    Args:
        sequence: The :class:`~egopressure.dataset.Sequence` to view.
        depth_provider: Optional ``(frame_index, camera) -> (H, W) depth_mm``
            callable (e.g. from the depth-merge tool). Enables the depth
            modality and depth-point overlays. If ``None``, depth is unavailable.

    Example:
        >>> viewer = FrameViewer(seq)
        >>> viewer.show(60, camera="d", overlays=["mesh", "skeleton"])
        >>> viewer.grid(60)          # all 8 cameras at once
        >>> viewer.interact()        # ipywidgets scrubber in a notebook
    """

    def __init__(self, sequence: Sequence, depth_provider: Callable | None = None):
        self.seq = sequence
        self.depth_provider = depth_provider

    # ── geometry helpers ───────────────────────────────────────────────────
    def _extrinsic(self, camera: str, ann):
        """Per-frame ego pose or ``None`` (static uses config MVM)."""
        if camera == EGO_CAMERA:
            return ann.ego_extrinsic() if ann is not None else None
        return None

    def _project(self, camera: str, ann, points):
        cal = self.seq.calibration(camera)
        return cal.project_world(points, extrinsic=self._extrinsic(camera, ann))

    # ── single-camera render ───────────────────────────────────────────────
    def render(
        self,
        frame_index: int,
        camera: str = EGO_CAMERA,
        modality: str = "rgb",
        overlays: Seq[str] = ("mesh", "skeleton", "pressure"),
        zoom: bool = True,
        panels: bool = True,
        ax=None,
    ):
        """Render one frame for one camera.

        Args:
            frame_index: Frame number.
            camera: Camera token (``"d"`` or ``"1"``..``"7"``).
            modality: ``"rgb"`` or ``"depth"`` (downloaded depth shards or a
                custom ``depth_provider``).
            overlays: Any of ``"skeleton"``, ``"mesh"``, ``"pressure"``.
            zoom: Crop to the hand when a pose overlay is available.
            panels: Also show the raw force grid and UV pressure map side panels.
            ax: Optional single Axes to draw the main image into (disables panels).

        Returns:
            The matplotlib ``Figure``.
        """
        # matplotlib is an optional extra (viz) — imported lazily on purpose
        import matplotlib.pyplot as plt

        try:
            ann = self.seq.load_annotation(frame_index)
        except ModalityNotDownloaded:
            ann = None          # rgb-only download: render without overlays
        if modality == "depth":
            base, dscale = self._depth_image(frame_index, camera)
        else:
            base, dscale = self.seq.load_rgb(frame_index, camera), None

        if ax is not None:
            fig = ax.figure
            main_axes = [ax]
            panels = False
        elif panels:
            fig = plt.figure(figsize=(18, 5))
            gs = fig.add_gridspec(1, 4)
            ax_main = fig.add_subplot(gs[0, :2])
            ax_force = fig.add_subplot(gs[0, 2])
            ax_uv = fig.add_subplot(gs[0, 3])
            main_axes = [ax_main]
        else:
            fig, ax_main = plt.subplots(figsize=(9, 5))
            main_axes = [ax_main]
        ax_main = main_axes[0]

        # base image
        registered = True
        pscale = 1.0                 # projected px -> base-image px divisor
        if modality == "depth":
            # overlays and hand-zoom apply only to color-registered depth
            registered = dscale is not None
            pscale = float(dscale) if registered else 1.0
            ax_main.imshow(np.where(base > 0, base, np.nan), cmap="turbo")
            if not registered:
                overlays = []
                zoom = False
        else:
            ax_main.imshow(base)
        H, W = base.shape[:2]

        # overlays: mesh -> skeleton -> pressure (white glow on top)
        pts_for_zoom = None
        if ann is not None and ann.has_annotation and ann.vertices is not None:
            verts2d = self._project(camera, ann, ann.vertices) / pscale
            joints2d = self._project(camera, ann, ann.joint_position) / pscale
            pts_for_zoom = np.concatenate([verts2d, joints2d], 0)
            if "mesh" in overlays:
                self._draw_mesh(ax_main, camera, ann, verts2d)
            if "skeleton" in overlays:
                for name, chain in FINGERS.items():
                    p = joints2d[chain]
                    ax_main.plot(p[:, 0], p[:, 1], "-",
                                 color=FINGER_COLORS[name], lw=2.5, zorder=4)
                    ax_main.scatter(p[:, 0], p[:, 1], s=20, color=FINGER_COLORS[name],
                                    zorder=5, edgecolors="white",
                                    linewidths=0.5)
        if "pressure" in overlays and (modality != "depth" or registered):
            cal_w, cal_h = self.seq.calibration(camera).image_size
            kpa = warp_force_to_image(
                self.seq.load_force(frame_index),
                self.seq.calibration(camera), (cal_h, cal_w),
                extrinsic=self._extrinsic(camera, ann), stride=4)
            if kpa.shape != base.shape[:2]:      # scaled (registered depth) base
                ys = np.linspace(0, kpa.shape[0] - 1, base.shape[0]).astype(int)
                xs = np.linspace(0, kpa.shape[1] - 1, base.shape[1]).astype(int)
                kpa = kpa[ys][:, xs]
            glow = np.zeros((*kpa.shape, 4))
            glow[..., :3] = 1.0
            glow[..., 3] = np.sqrt(
                np.clip(kpa / SENSEL_MAX_VIS_KPA, 0, 1)) * PRESSURE_GLOW_ALPHA
            ax_main.imshow(glow, zorder=6)

        if zoom and pts_for_zoom is not None:
            self._zoom_to(ax_main, pts_for_zoom, W, H)
        else:
            ax_main.set_xlim(0, W)
            ax_main.set_ylim(H, 0)
        tag = ("" if modality != "depth" else
               " registered to color" if registered else " (raw sensor frame)")
        ax_main.set_title(f"cam-{camera} · {modality}{tag} · frame {frame_index}")
        ax_main.axis("off")

        if panels:
            self._force_panel(ax_force, frame_index)
            self._uv_panel(ax_uv, ann)

        p = self.seq.participant
        fig.suptitle(f"{self.seq.name}  ·  subject {p.id} ({p.hand_side} hand)",
                     y=1.02, fontsize=12)
        fig.tight_layout()
        return fig

    # convenience alias
    def show(self, frame_index: int, **kwargs):
        """Alias for :meth:`render` (draws in the active notebook cell)."""
        return self.render(frame_index, **kwargs)

    # ── all-camera montage ─────────────────────────────────────────────────
    def grid(self, frame_index: int, cameras: Seq[str] | None = None,
             modality: str = "rgb",
             overlays: Seq[str] = ("mesh", "skeleton", "pressure"),
             zoom: bool = True):
        """Montage of every camera for one frame, with pose overlays.

        ``modality="depth"`` shows each camera's depth registered to its
        color view — the same geometry as the RGB tiles, so the overlays
        still apply.

        Args:
            frame_index: Frame number.
            cameras: Camera subset (default: all with imagery).
            modality: ``"rgb"`` (default) or ``"depth"`` (registered).
            overlays: Any of ``"mesh"``, ``"skeleton"``, ``"pressure"``.
            zoom: Crop each tile to the hand.
        """
        # matplotlib is an optional extra (viz) — imported lazily on purpose
        import matplotlib.pyplot as plt

        cams = list(cameras) if cameras is not None else self.seq.cameras
        n = len(cams)
        cols = min(4, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, cam in zip(axes, cams):
            self.render(frame_index, camera=cam, modality=modality,
                        overlays=overlays, zoom=zoom, ax=ax)
        for ax in axes[n:]:
            ax.axis("off")
        tag = "" if modality == "rgb" else f" · {modality}"
        fig.suptitle(f"{self.seq.name} · frame {frame_index} · all cameras{tag}",
                     y=1.0)
        fig.tight_layout()
        return fig

    # ── interactive scrubber ───────────────────────────────────────────────
    def interact(self, camera: str = EGO_CAMERA):
        """Notebook slider over frames (ipywidgets if available)."""
        frames = self.seq.frames
        try:
            import ipywidgets as w
            import matplotlib.pyplot as plt
            from IPython.display import display

            cam_dd = w.Dropdown(options=self.seq.cameras, value=camera,
                                description="camera")
            has_depth = self.depth_provider is not None or any(
                self.seq.has_depth(c) for c in self.seq.cameras)
            mod_dd = w.Dropdown(
                options=["rgb"] + (["depth"] if has_depth else []),
                value="rgb", description="modality")
            ov = w.SelectMultiple(options=OVERLAYS, value=("skeleton", "mesh"),
                                  description="overlays")
            sl = w.IntSlider(min=frames[0], max=frames[-1],
                             value=frames[len(frames) // 2],
                             description="frame", continuous_update=False)

            def _draw(camera, modality, overlays, frame_index):
                self.render(frame_index, camera=camera, modality=modality,
                            overlays=list(overlays))
                plt.show()

            out = w.interactive_output(_draw, {"camera": cam_dd, "modality": mod_dd,
                                               "overlays": ov, "frame_index": sl})
            display(w.VBox([w.HBox([cam_dd, mod_dd, ov]), sl, out]))
            return
        except ImportError:
            # static fallback: render the middle frame
            return self.render(frames[len(frames) // 2], camera=camera)

    # ── internals ──────────────────────────────────────────────────────────
    def _depth_image(self, frame_index: int,
                     camera: str) -> "tuple[np.ndarray, int | None]":
        """``(depth_mm, scale)`` — registered to the color view when possible
        (``scale`` = downscale vs full color resolution, so pose overlays can
        be drawn on top); ``scale`` is ``None`` for the raw sensor frame."""
        cal_w, cal_h = self.seq.calibration(camera).image_size
        if self.depth_provider is not None:
            base = np.asarray(self.depth_provider(frame_index, camera),
                              dtype=np.float64)
            if base.shape[:2] == (cal_h, cal_w):
                return base, 1
            return base, None
        try:
            scale = 2
            return (register_depth_to_color(
                self.seq, frame_index, camera,
                scale=scale).astype(np.float64), scale)
        except FileNotFoundError:
            # k4a configs absent: fall back to the raw sensor frame
            return np.asarray(self.seq.load_depth(frame_index, camera),
                              dtype=np.float64), None

    def _draw_mesh(self, ax, camera: str, ann, verts2d: np.ndarray) -> None:
        """Shaded neutral-gray MANO surface (falls back to vertex dots
        without the user-provided MANO files)."""
        mano = _mano_faces(ann.hand_side or "right")
        if mano is None:
            ax.scatter(verts2d[:, 0], verts2d[:, 1], s=2, c="#00e5ff",
                       alpha=0.30, linewidths=0, zorder=3)
            return
        # matplotlib is an optional extra (viz) — imported lazily on purpose
        from matplotlib.collections import PolyCollection

        cal = self.seq.calibration(camera)
        z = cal.world_to_cam(ann.vertices,
                             extrinsic=self._extrinsic(camera, ann))[:, 2]
        faces = mano["faces_v"]
        tri_uv = verts2d[faces]
        ok = ~np.isnan(tri_uv).any(axis=(1, 2))
        tri_uv, tri_z = tri_uv[ok], z[faces[ok]].mean(axis=1)
        order = np.argsort(-tri_z)                  # back to front
        zmin, zmax = tri_z.min(), tri_z.max() + 1e-9
        shade = 1.0 - MESH_SHADE_RANGE * (tri_z[order] - zmin) / (zmax - zmin)
        cols = np.empty((len(order), 4))
        for i, c in enumerate(MESH_RGB):
            cols[:, i] = c / 255.0 * shade
        cols[:, 3] = MESH_ALPHA
        ax.add_collection(PolyCollection(
            tri_uv[order], facecolors=cols, edgecolors="none", zorder=3))

    def _zoom_to(self, ax, pts, W, H):
        pts = pts[~np.isnan(pts).any(1)]
        if not len(pts):
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            return
        (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
        half = (max(x1 - x0, y1 - y0)) * 0.8 + 40
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        ax.set_xlim(max(cx - half, 0), min(cx + half, W))
        ax.set_ylim(min(cy + half, H), max(cy - half, 0))

    def _force_panel(self, ax, frame_index):
        try:
            force = self.seq.load_force(frame_index)
        except (ModalityNotDownloaded, FileNotFoundError):
            ax.set_title("raw force (not downloaded)")
            ax.axis("off")
            return
        kpa = counts_to_kpa(force)
        # aspect="equal": Sensel cells are square (1.25 mm pitch)
        im = ax.imshow(kpa, cmap="inferno", aspect="equal",
                       vmin=0, vmax=max(float(kpa.max()), 1e-6))
        ax.set_title(f"Sensel force grid [kPa] · peak {kpa.max():.1f}")
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.axis("off")

    def _uv_panel(self, ax, ann):
        # standard toolkit UV design: white background, gray hand wireframe,
        # inferno-blended pressure (shared with the video renderer)
        ax.imshow(uv_pressure_panel(ann, 448, 448))
        ax.set_title("Hand UV pressure map")
        ax.axis("off")
