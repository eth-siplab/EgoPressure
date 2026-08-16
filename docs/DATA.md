# EgoPressure — Data Reference

Everything in the dataset, field by field, with units. The toolkit handles these
conventions automatically; this reference is for anyone working with the shards
directly or building on the geometry.

## Units at a glance

| Quantity | Unit | Notes |
|---|---|---|
| MANO vertices / joints / `transl` | **metres** | world (rig) frame |
| Egocentric camera pose (`ego_R`, `ego_T`) | **metres** | per frame, world → camera: `P_cam = R·P_world + T` |
| Static camera `ModelViewMatrix` | **millimetres** | fixed, world → color camera; scale world points ×1000 before applying |
| Depth maps | **millimetres** | uint16, `0` = no measurement; sensor frame, coverage = color view (+32 px) |
| Sensel force grid | **counts** | `counts / 1736 → newtons`; `N / (0.00125 m)² / 1000 → kPa` |
| Pressure UV map | normalised | multiply by `pressure_map_range[1]` for counts |
| Image coordinates | pixels | all released images are **undistorted** → pinhole projection |
| Timestamps (raw sidecar data, not in shards) | µs epoch | PTP-synchronised across devices |

## The world frame

The origin lies on the **touchpad surface**: the pad is the x–y plane at
`z = 0`, centred at the origin, spanning `x ∈ ±0.120 m`, `y ∈ ±0.06875 m`
(the calibrated 240 × 137.5 mm pad rectangle; cell pitch 1.25 mm). Sensor grid
corners map to the pad as `(col,row) = (185,0)→(+x,−y) … (0,0)→(−x,−y)`.

**+z points downward through the pad** (right-handed with x–y in the pad
plane): the hand hovers at *negative* z, and all cameras sit at z < 0. If you
assume the usual z-up convention the scene will appear flipped.

## Repository layout

```
configs/<participant>/<sequence>.json       calibration + participant metadata
configs/<participant>/<sequence>_k4a/       per-camera k4a factory calibrations
data/<participant>/<sequence>/
    cam-{d,1..7}.color.parquet              RGB (original JPEG bytes)
    cam-{d,1..7}.depth.parquet              512×512 uint16 depth (PNG bytes)
    cam-{d,1..7}.mask.parquet               hand masks (PNG bytes)
    pressure.parquet                        force grid + UV pressure per frame
    annotation.parquet                      MANO + ego pose per frame
```

Each Parquet shard has one row per frame (`frame` int column). A frame a
camera dropped at capture time is simply **absent from that camera's shard** —
check row presence, not just frame ranges.

## Cameras

| Token | Role | Color resolution | Extrinsics |
|---|---|---|---|
| `d` | egocentric (head-mounted) | 1920×1080 | per frame in `annotation` (`ego_R` 3×3, `ego_T` 3, metres) |
| `1` … `7` | static Azure Kinect | 2560×1440 | fixed `ModelViewMatrix` (4×4, **mm**) in the sequence config |

The sequence config's `camera_calibrations` is keyed `"0"`(=ego), `"1"`…`"7"`,
each with pinhole intrinsics `fx, fy, cx, cy`, the original lens distortion
coefficients `k1–k6, p1, p2` (plus unused `p3, p4` placeholders in some
entries; informational — images ship undistorted),
`DepthToColor` (4×4 intra-device transform), and the sensor `SerialNo`.

### Depth vs RGB

RGB and depth come from **different sensors** on each Azure Kinect. The RGB
images ship undistorted (plain pinhole — do *not* re-apply the distortion
coefficients). Depth is stored in the **depth-sensor frame** (512×512 WFOV
2×2-binned), *distorted*, and **not** registered to color — pixel `(u,v)` in
a depth map does **not** correspond to `(u,v)` in the RGB image.

Depth **coverage matches the color view**: each map contains the pixels whose
3-D point falls inside that camera's color image (+32 px guard band); all
other pixels are `0` (no measurement, same semantics as any unmeasured
pixel). Within the covered region, values are the sensor's raw measurements —
and the guard band is wide enough that `register_depth_to_color()` output is
identical to what full-sensor maps would produce.

To combine depth with RGB, register on the fly:

```python
from egopressure.registration import register_depth_to_color
reg = register_depth_to_color(seq, frame_index, camera="4", scale=4)
# (H/scale, W/scale) float32 mm in the color camera's pinhole frame, 0 = hole
```

Under the hood (all inputs ship with the dataset):
- `configs/<p>/<seq>_k4a/cam-<c>.k4a_calibration.json` is the k4a factory
  calibration. Depth intrinsics are the `...D0` camera's normalised
  Brown–Conrady `ModelParameters` — multiply the first four (cx, cy, fx, fy)
  by the delivered image width/height (512) for pixel units.
- Each camera entry's `Rt` maps points *from the depth (D0, reference) frame
  into that camera's frame*, translation in **metres** — so the `...PV0`
  (color) entry's `Rt` is the depth→color transform (≈32 mm baseline).
- Unproject each valid depth pixel (inverting the distortion), transform with
  `Rt`, project with the released color intrinsics, z-buffer.

`egopressure video --modality depth` renders depth **registered to color** by
default (`--depth-view sensor` shows the raw maps as stored). For world-space
point clouds from ego depth, `CameraCalibration.unproject_depth()` +
`project_world()` handle the full chain.

## Sequences & gestures

Each participant recorded **32 gestures with each hand** (64 sequences),
named `<participant>_<gesture>_<hand>`:

| Family | Gestures | Description |
|---|---|---|
| Calibration | `calibration_routine` | the hand slowly turns with fingers spread, visible from all cameras |
| Palm presses | `press_palm_{high,low,no-contact}_x5`, `press_palm-and-fingers_{high,low,no-contact}_x5` | whole palm (or palm plus all fingers) pressed flat onto the pad |
| Finger presses | `press_fingers_{high,low,no-contact}_5x`, `press_flat_onebyone_{high,low}_3x`, `press_cupped_onebyone_{high,low}_3x` | all fingers pressed together; or one finger at a time with the hand held flat (`flat_onebyone`) vs. arched so only the fingertips contact (`cupped_onebyone`) |
| Index interactions | `index_press_{high,low,no-contact}_x5`, `index_press_{pull,push,rotate-left,rotate-right}_x5` | single index-finger press; the second group drags while pressed — toward the user (`pull`), away (`push`), or twisting (`rotate-*`) |
| Pinches | `pinch_thumb-down_{high,low,no-contact}_5x`, `pinch-zoom_5x` | pinch with the thumb pressing the pad; `pinch-zoom` is the two-finger touchscreen spread/contract |
| Grasps | `grasp-edge_{curled,uncurled}_thumb-down_5x`, `grasp-edge_curled_thumb-up_5x` | grasping the pad edge, fingers curled or extended, thumb on top (`thumb-up`) or underneath (`thumb-down`) |
| Surface motion | `pull-towards_5x`, `push-away_5x`, `draw_word_5x`, `type_ipad_5x` | flat-hand drags toward/away from the body; writing a word with the index finger; tapping as on a tablet keyboard |

`high` / `low` / `no-contact` are pressure-intensity variants of the same
motion (`no-contact` hovers just above the pad). Single-touch gestures were
repeated 5 times, sequential ones (one-by-one presses, drawing) 3 times —
the repetition count is encoded in each sequence name.

Across the release: **1,344 sequences**, ≈630K frames per camera stream
(≈5M camera-frames overall).

## Per-frame annotation fields

Shard **column** names below; in Python, `ego_R`/`ego_T` surface as
`Annotation.ego_camera_pose` (dict) and `Annotation.ego_extrinsic()` (4×4),
everything else keeps its column name as an attribute.

| Field | Shape | Meaning |
|---|---|---|
| `has_annotation` | bool | `False` on pre/post-contact frames — MANO/pressure fields are null there; `ego_R`/`ego_T` and `hand_side` are **always** present |
| `vertices` | 778×3 | MANO mesh, world frame, metres |
| `joint_position` | 21×3 | joint order: `0` wrist; `1-4` thumb, `5-8` index, `9-12` middle, `13-16` ring, `17-20` pinky (each base→tip) |
| `betas` | 1×10 | MANO shape parameters, stored per frame — use together with the frame's `full_pose`/`transl` for exact vertex reconstruction |
| `full_pose` | 48 | 16 joints × 3 axis-angle (global orientation first) |
| `transl` | 3 | global translation |
| `normals` / `displacement` | 778×3 / 778×1 | per-vertex normals and personalised offset |
| `visible_vertices` | 7×778 | per-**static**-camera binary visibility (rows = cameras 1…7) |
| `ego_R`, `ego_T` | 3×3, 3 | egocentric camera pose (world→camera, metres) |
| `hand_side` | str | `"left"` / `"right"` |

Array columns are stored as flattened float lists — reshape per the
table (programmatically: `egopressure.ARRAY_SHAPES`; the toolkit returns
float32). Image columns (`image` / `depth` / `mask`) are structs
`{bytes, path}` holding the encoded file bytes. Frame numbering starts at
**1**.

## Reconstructing vertices from MANO parameters

The released ``vertices`` are the annotation pipeline's **optimised** meshes:
nominal MANO plus a per-vertex refinement stored explicitly in
``displacement``/``normals``. The exact recipe:

```
vertices = LBS(betas, full_pose) + transl + displacement * normals
```

- ``full_pose`` (48,) is 16 x 3 axis-angle, **global orientation first**, in
  the **flat-hand convention** — MANO's ``hands_mean`` is *not* added.
- Standard MANO linear blend skinning (shape blendshapes with the first 10
  components, pose-corrective blendshapes, 16-joint kinematic tree).
- ``displacement * normals`` is the optimisation residual on top of nominal
  MANO; without it, nominal MANO agrees to ~1 mm mean (up to ~8 mm at
  individual vertices).

The toolkit ships a chumpy-free reference implementation:

```python
from egopressure.mano import load_mano_model, mano_vertices
model = load_mano_model(ann.hand_side)          # needs the MANO pkl files
v = mano_vertices(model, ann.betas, ann.full_pose, ann.transl,
                  ann.displacement, ann.normals)    # == ann.vertices
```

## Pressure: two representations, one measurement

- `force` (105×185 float32): the raw Sensel grid in counts. All-zero is normal
  before contact.
- `pressure_map` (224×224×1) + `pressure_map_range` (2): the same measurement
  baked onto the MANO hand's UV layout, normalised;
  `map × range[1]` recovers counts. Rendering it on the hand surface requires
  the MANO UV files (see `mano_models/README.md`).

Projecting the raw grid into any camera view needs only the pad geometry above
(`egopressure.senselpad` implements it).

## Known data notes

- **p_002 rotate-right**: this participant has *two* right-hand takes
  (`index_press_rotate-right_x5_right` and `…_x5_2_right`) and **no left-hand
  recording** of this gesture — a capture-session re-take; all other
  participants have the standard left/right pair.
- Per-camera dropped frames (a few per sequence) appear as missing shard rows;
  the egocentric stream is complete for every released frame.
- MANO vertices can dip below `z = 0` **outside the pad footprint** (e.g. the
  wrist hanging over the pad edge, up to a few cm) — physically correct, not
  an annotation error; within the pad area, penetration stays within a few mm.
- Sequence-name repetition suffixes vary in form (`x5` vs `5x` vs `3x`) —
  match both orders when globbing by repetition count.

## Participant metadata

Each sequence config carries a `participant` block: pseudonymous `id`, `age`,
`height` (cm), `weight` (kg), `gender`, `hand_side`, capture `exposure` and
`light_tubes` settings.
