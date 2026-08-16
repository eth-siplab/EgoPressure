"""Every EgoPressure modality and config, end to end.

Downloads one sequence (all cameras, all modalities) and walks through each
piece of data with its units and conventions. Run top to bottom:

    python examples/all_modalities.py
"""

import numpy as np

from egopressure import ARRAY_SHAPES, EgoPressure
from egopressure.registration import register_depth_to_color, world_points_from_depth
from egopressure.senselpad import counts_to_kpa, pad_outline
from egopressure.video import save_video

PID, SEQ = "p_001", "p_001_press_palm_low_x5_right"

# ── 1. Selective download: everything for one sequence ─────────────────────
ep = EgoPressure.from_hub(participants=[PID], sequences=[SEQ],
                          out_dir="egopressure_data")   # all cams, all modalities
seq = ep.sequence(PID, SEQ)
print(seq, "| cameras:", seq.cameras)

# ── 2. Participant metadata & camera configs ───────────────────────────────
p = seq.participant
print(f"participant {p.id}: {p.gender}, {p.age}y, {p.height}cm, "
      f"{p.weight}kg, hand={p.hand_side}")

cal_ego = seq.calibration("d")          # egocentric: pose comes per frame
cal_s4 = seq.calibration("4")           # static: fixed ModelViewMatrix (mm)
print("ego intrinsics fx,fy,cx,cy:", cal_ego.fx, cal_ego.fy, cal_ego.cx, cal_ego.cy)
print("static cam-4 world extrinsic (mm):\n", cal_s4.model_view_matrix)

# ── 3. A fully-annotated frame, all modalities ─────────────────────────────
fi = seq.annotated_frames()[len(seq.annotated_frames()) // 2]
frame = seq.load_frame(fi, mask=True, depth=True)

rgb_ego = frame.rgb["d"]                # (1080, 1920, 3) uint8, undistorted
rgb_s1 = frame.rgb["1"]                 # (1440, 2560, 3) static camera
mask = frame.mask["d"]                  # hand segmentation
depth = frame.depth["d"]                # (512, 512) uint16 millimetres
force = frame.force                     # (105, 185) float32 Sensel counts
ann = frame.annotation
print(f"frame {fi}: rgb {rgb_ego.shape}, mask {mask.shape}, "
      f"depth {depth.shape} [{depth.min()}-{depth.max()}mm], "
      f"force peak {force.max():.0f} counts = {counts_to_kpa(force).max():.1f} kPa")

# ── 4. MANO annotation (shapes in ARRAY_SHAPES) ────────────────────────────
print("MANO vertices (m):", ann.vertices.shape,
      "| joints:", ann.joint_position.shape,
      "| betas:", ann.betas.shape, "| pose:", ann.full_pose.shape)
print("visible vertices per static cam:",
      {c: int(v.sum()) for c, v in ann.visible_vertices.items()})
print("UV pressure map:", ann.pressure_map.shape,
      "| physical peak:", f"{ann.denormalized_pressure_map().max():.0f} counts")
print("(all flat-array shapes:", list(ARRAY_SHAPES)[:5], "…)")

# ── 5. Geometry: world <-> pixels for every camera ─────────────────────────
uv_ego = cal_ego.project_world(ann.vertices, extrinsic=ann.ego_extrinsic())
uv_s4 = cal_s4.project_world(ann.vertices)          # static: no extrinsic arg
print(f"mesh in ego view: u∈[{np.nanmin(uv_ego[:, 0]):.0f},"
      f"{np.nanmax(uv_ego[:, 0]):.0f}]")
print("touchpad outline in cam-4 (px):", pad_outline(cal_s4)[:2].round(0))

# raw depth is in the depth-sensor frame; this registers + unprojects it
cloud = world_points_from_depth(seq, fi, "d", stride=4)
print(f"ego depth -> world cloud: {cloud.shape[0]} points, "
      f"z-range [{cloud[:, 2].min():.2f},{cloud[:, 2].max():.2f}] m "
      "(pad plane at z=0, +z pointing down through the pad)")

# ── 6. Visualization ───────────────────────────────────────────────────────
frame.show(camera="d", overlays=["mesh", "skeleton", "pressure"])   # figure
save_video(seq, "sequence.mp4")                                     # RGB video
save_video(seq, "sequence_depth.mp4", modality="depth")   # registered to color
print("wrote sequence.mp4 + sequence_depth.mp4")

# ── 7. Depth <-> RGB: register raw sensor-frame depth into the color view ──
reg = register_depth_to_color(seq, fi, camera="4", scale=4)   # (360,640) mm
print(f"registered depth cam-4: {reg.shape}, "
      f"coverage {(reg > 0).mean() * 100:.0f}% (0 = no sample)")
