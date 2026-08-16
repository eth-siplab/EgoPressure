"""Data-free smoke tests: imports, layout, geometry math, unit conversions.

Run:  python tests/test_smoke.py   (or pytest)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

failures = 0


def check(label: str, ok: bool) -> None:
    global failures
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failures += (not ok)
    assert ok, label          # pytest-visible; script mode also stops on red


def test_imports():
    import egopressure as ep

    check("package imports", True)
    check("__all__ resolvable",
          all(hasattr(ep, n) for n in ep.__all__))


def test_layout():
    from egopressure import layout

    pats = layout.allow_patterns(participants=["p_001"], cameras=["d"],
                                 modalities=["rgb", "depth", "pressure"])
    check("allow_patterns selects ego color",
          "data/p_001/*/cam-d.color.parquet" in pats)
    check("aliases normalise", layout.canonical_modality("pose") == "annotation")
    check("shard names", layout.shard_filename("depth", "4") == "cam-4.depth.parquet")


def test_calibration_geometry():
    from egopressure.calibration import CameraCalibration

    entry = {"fx": 900.0, "fy": 900.0, "cx": 960.0, "cy": 540.0,
             "k1": 0, "k2": 0, "k3": 0, "k4": 0, "k5": 0, "k6": 0,
             "p1": 0, "p2": 0, "DepthToColor": np.eye(4).tolist(),
             "ModelViewMatrix": np.eye(4).tolist(),
             "ImageSizeX": 1920, "ImageSizeY": 1080}
    cal = CameraCalibration.from_config("1", entry)
    # identity extrinsic (mm-space): a point 1m ahead projects to the centre
    uv = cal.project_world(np.array([[0.0, 0.0, 1.0]]))
    check("project_world centre", np.allclose(uv, [[960.0, 540.0]]))
    # unproject a synthetic flat depth map and reproject: round-trip
    depth = np.full((1080, 1920), 1000, dtype=np.uint16)   # 1m everywhere
    pts = cal.unproject_depth(depth, stride=97)
    uv2 = cal.project_world(pts)
    v, u = np.mgrid[0:1080:97, 0:1920:97]
    expect = np.stack([u.ravel(), v.ravel()], 1).astype(float)
    check("unproject/project round-trip",
          np.allclose(uv2, expect, atol=1e-6))


def test_senselpad_units():
    from egopressure import senselpad as sp

    kpa = sp.counts_to_kpa(np.array([1736.0]))
    check("1736 counts == 1 N == 640 kPa/cell",
          np.isclose(kpa[0], 1.0 / (0.00125 ** 2) / 1000.0))
    check("pad corners map", sp.CORNERS_2D.shape == (4, 2)
          and sp.CORNERS_3D.shape == (4, 3))


def test_mano_placeholder():
    from egopressure.mano import find_mano_models, mano_available

    check("mano optional-absent is graceful",
          mano_available() in (True, False))
    if not mano_available():
        try:
            find_mano_models(require=True)
            check("missing MANO raises with instructions", False)
        except FileNotFoundError as e:
            check("missing MANO raises with instructions",
                  "mano.is.tue.mpg.de" in str(e))


if __name__ == "__main__":
    for fn in [test_imports, test_layout, test_calibration_geometry,
               test_senselpad_units, test_mano_placeholder]:
        print(fn.__name__)
        fn()
    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
