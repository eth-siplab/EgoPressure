# MANO model files (optional, user-provided)

Some toolkit features — exact UV-pressure rendering on the hand surface and mesh
faces for surface rasterisation — need the MANO model files. MPI distributes
them under a separate license, so they **cannot be bundled** here and must not
be committed to this repository.

## How to get them

1. Register (free) at https://mano.is.tue.mpg.de/
2. Go to **Downloads** and fetch:
   - `MANO_LEFT.pkl`, `MANO_RIGHT.pkl` (Models & Code)
   - `MANO_UV_left.obj`, `MANO_UV_right.obj` (UV files)
3. Place all four files in this folder:

```
mano_models/
    MANO_LEFT.pkl
    MANO_RIGHT.pkl
    MANO_UV_left.obj
    MANO_UV_right.obj
```

Alternatively put them anywhere and set `EGOPRESSURE_MANO_PATH=/path/to/folder`.

The toolkit detects them automatically (`egopressure.mano.mano_available()`); all
core features (loading, download, viewer, geometry, pressure overlays) work
**without** these files.
