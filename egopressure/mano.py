"""Optional MANO assets: discovery + guidance.

Some features (exact UV-pressure rendering on the hand surface, mesh faces for
surface rasterisation) require the MANO model files, which are distributed by
MPI under their own license and **cannot be bundled** with this package.

Download once, drop into a ``mano_models/`` folder (working directory, package
root, or ``$EGOPRESSURE_MANO_PATH``), and the toolkit picks them up automatically::

    mano_models/
        MANO_LEFT.pkl
        MANO_RIGHT.pkl
        MANO_UV_left.obj
        MANO_UV_right.obj

Get the files from https://mano.is.tue.mpg.de/ (free account required):
*Downloads -> Models & Code* for the ``.pkl`` files and the UV ``.obj`` files.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np

REQUIRED = ("MANO_LEFT.pkl", "MANO_RIGHT.pkl",
            "MANO_UV_left.obj", "MANO_UV_right.obj")

_SEARCH = (
    lambda: os.environ.get("EGOPRESSURE_MANO_PATH"),
    lambda: "mano_models",
    lambda: Path(__file__).resolve().parent.parent / "mano_models",
)


def find_mano_models(require: bool = True,
                     files: tuple[str, ...] = REQUIRED) -> Path | None:
    """Locate a ``mano_models/`` folder containing the given MANO files.

    Search order: ``$EGOPRESSURE_MANO_PATH``, ``./mano_models``, then a
    ``mano_models`` folder next to the installed package.

    Args:
        require: If ``True``, raise a :class:`FileNotFoundError` with download
            instructions when the files are missing; otherwise return ``None``.
        files: The specific filenames needed (default: all four). Features
            needing only the ``.pkl`` models or only the UV ``.obj`` files
            pass a subset, so users install only what their use case needs.
    """
    for get in _SEARCH:
        cand = get()
        if not cand:
            continue
        p = Path(cand)
        if p.is_dir() and all((p / f).exists() for f in files):
            return p
    if require:
        raise FileNotFoundError(
            "MANO model files not found. Some features (UV pressure rendering, "
            "mesh faces) need them. Download from https://mano.is.tue.mpg.de/ "
            f"(free account) and place {', '.join(REQUIRED)} into a "
            "'mano_models/' folder (or set $EGOPRESSURE_MANO_PATH). "
            "See mano_models/README.md for step-by-step instructions."
        )
    return None


def mano_available() -> bool:
    """True if the optional MANO files are installed."""
    return find_mano_models(require=False) is not None


def load_mano_uv(hand_side: str = "right") -> dict:
    """Parse the MANO UV ``.obj`` (user-provided) for faces and UV coordinates.

    Returns a dict with:
      - ``faces_v``  (F, 3) int — triangle indices into the 778 MANO vertices
      - ``faces_vt`` (F, 3) int — triangle indices into the UV coordinates
      - ``uv``       (T, 2) float — UV coordinates in [0, 1] (v origin bottom)

    Raises :class:`FileNotFoundError` with download instructions when the MANO
    files are absent (see :func:`find_mano_models`).
    """
    root = find_mano_models(require=True,
                            files=(f"MANO_UV_{hand_side}.obj",))
    path = root / f"MANO_UV_{hand_side}.obj"
    faces_v, faces_vt, uv = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("vt "):
                _, u, v = line.split()[:3]
                uv.append((float(u), float(v)))
            elif line.startswith("f "):
                tri_v, tri_vt = [], []
                for tok in line.split()[1:4]:
                    parts = tok.split("/")
                    tri_v.append(int(parts[0]) - 1)
                    tri_vt.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1]
                                  else int(parts[0]) - 1)
                faces_v.append(tri_v)
                faces_vt.append(tri_vt)
    return {"faces_v": np.asarray(faces_v, dtype=int),
            "faces_vt": np.asarray(faces_vt, dtype=int),
            "uv": np.asarray(uv, dtype=float)}


# ── MANO forward pass (chumpy-free) ─────────────────────────────────────────

def load_mano_model(hand_side: str = "right") -> dict:
    """Load ``MANO_{SIDE}.pkl`` as plain numpy arrays — no chumpy/scipy needed.

    The official pickles embed chumpy arrays and scipy sparse matrices; this
    loader unpickles them with lightweight stubs and converts everything to
    dense numpy. Returns a dict with ``v_template (778,3)``,
    ``shapedirs (778,3,S)``, ``posedirs (778,3,135)``, ``J_regressor (16,778)``,
    ``weights (778,16)``, ``kintree_table (2,16)``, ``hands_mean (45,)`` and
    triangle ``f (1538,3)``.
    """
    class _Stub:
        def __init__(self, *args, **kw):
            pass

        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})

    class _CSC(_Stub):
        pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("chumpy"):
                return _Stub
            if module.startswith("scipy.sparse"):
                return _CSC
            return super().find_class(module, name)

    def to_np(x):
        if isinstance(x, _CSC):                      # dense-ify csc state
            d = x.__dict__
            out = np.zeros(d.get("_shape") or d.get("shape"))
            indptr, indices, data = d["indptr"], d["indices"], d["data"]
            for col in range(out.shape[1]):
                for k in range(indptr[col], indptr[col + 1]):
                    out[indices[k], col] = data[k]
            return out
        if isinstance(x, _Stub):
            for key in ("x", "a", "v"):
                if key in x.__dict__:
                    return to_np(x.__dict__[key])
            raise ValueError("unsupported stub in MANO pickle")
        return np.asarray(x)

    root = find_mano_models(require=True,
                            files=(f"MANO_{hand_side.upper()}.pkl",))
    with open(root / f"MANO_{hand_side.upper()}.pkl", "rb") as f:
        raw = _Unpickler(f, encoding="latin1").load()
    return {k: to_np(raw[k])
            for k in ("v_template", "shapedirs", "posedirs", "J_regressor",
                      "weights", "kintree_table", "hands_mean", "f")}


def mano_vertices(model: dict, betas, full_pose, transl=None,
                  displacement=None, normals=None):
    """EgoPressure's exact vertex reconstruction from MANO parameters.

    ``vertices = LBS(betas, full_pose) + transl + displacement * normals``

    - ``full_pose`` (48,) is 16 x 3 axis-angle with the global orientation
      first, in the **flat-hand** convention (``hands_mean`` is *not* added).
    - ``displacement * normals`` is the per-vertex refinement the annotation
      pipeline optimised on top of nominal MANO; with it the released
      ``vertices`` are reproduced to float32 precision (<0.1 mm), without it
      nominal MANO is within ~1 mm mean.

    Args:
        model: From :func:`load_mano_model` (matching the annotation's hand).
        betas: ``(10,)`` or ``(1, 10)`` shape parameters.
        full_pose: ``(48,)`` axis-angle pose.
        transl: ``(3,)`` global translation (world frame), or ``None``.
        displacement: ``(778, 1)`` per-vertex offsets, or ``None``.
        normals: ``(778, 3)`` per-vertex directions for ``displacement``.

    Returns:
        ``(778, 3)`` float64 vertices (world frame, metres, when ``transl``
        and the refinement are applied).
    """
    def rodrigues(r):
        theta = np.linalg.norm(r)
        if theta < 1e-12:
            return np.eye(3)
        k = r / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    pose = np.asarray(full_pose, dtype=np.float64).reshape(16, 3)
    v_shaped = model["v_template"] + np.einsum(
        "vij,j->vi", model["shapedirs"][..., :betas.size], betas)
    J = model["J_regressor"] @ v_shaped
    Rs = np.stack([rodrigues(pose[i]) for i in range(16)])
    pose_feat = (Rs[1:] - np.eye(3)).reshape(-1)
    v_posed = v_shaped + np.einsum("vij,j->vi", model["posedirs"], pose_feat)

    parent = model["kintree_table"][0].astype(int)
    G = np.zeros((16, 4, 4))
    G[0] = np.eye(4)
    G[0][:3, :3] = Rs[0]
    G[0][:3, 3] = J[0]
    for i in range(1, 16):
        L = np.eye(4)
        L[:3, :3] = Rs[i]
        L[:3, 3] = J[i] - J[parent[i]]
        G[i] = G[parent[i]] @ L
    for i in range(16):
        G[i][:3, 3] -= G[i][:3, :3] @ J[i]
    T = np.einsum("vk,kij->vij", model["weights"], G)
    vh = np.concatenate([v_posed, np.ones((778, 1))], axis=1)
    v = np.einsum("vij,vj->vi", T, vh)[:, :3]

    if transl is not None:
        v = v + np.asarray(transl, dtype=np.float64).reshape(3)
    if displacement is not None and normals is not None:
        v = v + (np.asarray(displacement, dtype=np.float64).reshape(-1, 1)
                 * np.asarray(normals, dtype=np.float64))
    return v
