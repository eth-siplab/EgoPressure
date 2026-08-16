"""``egopressure`` command-line interface.

Shell-first access to the dataset — fetch data or get a quick visual check
without writing any Python::

    egopressure list
    egopressure download --participants p_001 --cameras d --modalities rgb,depth,pose
    egopressure show p_001 p_001_press_palm_low_x5_right 60 --out frame.png
"""

from __future__ import annotations

import argparse
import sys

from .constants import DEFAULT_DATASET_REPO

# Command bodies import their dependencies lazily so `egopressure --help`
# and argument errors stay instant regardless of optional-dependency cost.


def _split(value: str | None) -> list[str] | None:
    return value.split(",") if value else None


def _cmd_list(args) -> int:
    from .hub import list_available

    tree = list_available(repo=args.repo, token=args.token)
    if not tree:
        print("(repository is empty or inaccessible)")
        return 1
    for pid, seqs in sorted(tree.items()):
        print(f"{pid}  ({len(seqs)} sequences)")
        if args.verbose:
            for name, shards in sorted(seqs.items()):
                print(f"  {name}  [{len(shards)} shards]")
    return 0


def _cmd_download(args) -> int:
    from .hub import download

    path = download(
        repo=args.repo,
        token=args.token,
        participants=_split(args.participants),
        sequences=_split(args.sequences),
        cameras=_split(args.cameras),
        modalities=_split(args.modalities),
        out_dir=args.out,
    )
    print(f"downloaded to {path}")
    return 0


def _cmd_video(args) -> int:
    from .dataset import EgoPressureDataset
    from .video import save_video

    ds = EgoPressureDataset(args.root)
    seq = ds.sequence(args.participant, args.sequence)
    out = args.out or f"{args.sequence}.mp4"
    overlays = tuple(o for o in (args.overlays or "").split(",") if o)
    frames = None
    if args.frames:
        if ":" in args.frames:
            a, b = (int(x) for x in args.frames.split(":", 1))
            frames = [f for f in seq.frames if a <= f <= b]
        else:
            frames = [int(x) for x in args.frames.split(",")]
        if not frames:
            raise ValueError(f"--frames {args.frames!r} matches no frames "
                             f"(sequence spans {seq.frames[0]}..{seq.frames[-1]})")
    path = save_video(seq, out, cameras=_split(args.cameras),
                      overlays=overlays, frames=frames,
                      modality=args.modality, depth_view=args.depth_view,
                      mesh_style=args.mesh_style, fps=args.fps, crf=args.crf)
    print(f"wrote {path}")
    return 0


def _cmd_show(args) -> int:
    import matplotlib

    matplotlib.use("Agg" if args.out else matplotlib.get_backend())
    import matplotlib.pyplot as plt

    from .dataset import EgoPressureDataset

    ds = EgoPressureDataset(args.root)
    seq = ds.sequence(args.participant, args.sequence)
    fig = seq.show(args.frame, camera=args.camera,
                   overlays=args.overlays.split(","))
    if args.out:
        fig.savefig(args.out, dpi=110, bbox_inches="tight")
        print(f"wrote {args.out}")
    else:
        plt.show()
    return 0


def _friendly(exc: BaseException) -> str | None:
    """Map common failures to a one-line actionable message (None = re-raise)."""
    name = type(exc).__name__
    if name == "GatedRepoError":
        return ("access to this dataset is gated — request access at "
                "https://huggingface.co/datasets/eth-siplab/EgoPressure, then "
                "authenticate with `hf auth login` (or pass --token)")
    if name in ("RepositoryNotFoundError", "LocalTokenNotFoundError"):
        return f"Hugging Face Hub: {exc}"
    if name in ("RemoteProtocolError", "ConnectError", "ConnectTimeout",
                "ReadTimeout", "ChunkedEncodingError") \
            or isinstance(exc, ConnectionError):
        return ("network error while contacting the Hub — check your "
                "connection and retry (partial downloads resume automatically)")
    if isinstance(exc, ModuleNotFoundError):
        if exc.name == "matplotlib":
            return ('this command needs matplotlib — install with '
                    '`pip install "egopressure[viz]"`')
        if exc.name == "torch":
            return "this command needs PyTorch — `pip install torch`"
        return None
    if isinstance(exc, (FileNotFoundError, ValueError, RuntimeError)):
        return str(exc)
    if isinstance(exc, KeyError):
        return str(exc.args[0]) if exc.args else str(exc)
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="egopressure",
        description="EgoPressure dataset: list, download, and view.")
    hub_args = argparse.ArgumentParser(add_help=False)
    hub_args.add_argument("--repo", default=DEFAULT_DATASET_REPO,
                          help=f"HF dataset repo (default: {DEFAULT_DATASET_REPO})")
    hub_args.add_argument("--token", default=None,
                          help="HF access token (default: `hf auth login` credentials)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", parents=[hub_args],
                       help="list participants/sequences in the repo")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=_cmd_list)

    p = sub.add_parser("download", parents=[hub_args],
                       help="download a selection of the dataset")
    p.add_argument("--participants", help="comma-separated, e.g. p_001,p_002")
    p.add_argument("--sequences", help="comma-separated sequence names")
    p.add_argument("--cameras", help="comma-separated tokens: d,1..7")
    p.add_argument("--modalities",
                   help="comma-separated: rgb,depth,mask,pressure,pose")
    p.add_argument("--out", default="egopressure_data")
    p.set_defaults(fn=_cmd_download)

    p = sub.add_parser("video", help="render a sequence to MP4 (needs ffmpeg)")
    p.add_argument("participant")
    p.add_argument("sequence")
    p.add_argument("--root", default="egopressure_data")
    p.add_argument("--cameras", help="comma-separated subset (default: all)")
    p.add_argument("--modality", default="rgb", choices=["rgb", "depth"])
    p.add_argument("--depth-view", default="registered",
                   choices=["registered", "sensor"],
                   help="depth tiles: registered to color (default) "
                        "or raw sensor frame")
    p.add_argument("--mesh-style", default="surface",
                   choices=["points", "surface"],
                   help="hand overlay: shaded MANO surface (default) or vertex points")
    p.add_argument("--overlays", default="mesh,skeleton,pressure",
                   help="comma-separated: mesh,skeleton,pressure (or empty for none)")
    p.add_argument("--out", default=None, help="output .mp4 (default: <sequence>.mp4)")
    p.add_argument("--frames", default=None,
                   help="frame subset: 'START:END' (inclusive) or comma list")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--crf", type=int, default=28, help="H.264 quality (higher=smaller)")
    p.set_defaults(fn=_cmd_video)

    p = sub.add_parser("show", help="render one frame (needs matplotlib)")
    p.add_argument("participant")
    p.add_argument("sequence")
    p.add_argument("frame", type=int)
    p.add_argument("--root", default="egopressure_data")
    p.add_argument("--camera", default="d")
    p.add_argument("--overlays", default="skeleton,mesh,pressure")
    p.add_argument("--out", default=None, help="save PNG instead of opening a window")
    p.set_defaults(fn=_cmd_show)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:                     # CLI boundary
        msg = _friendly(exc)
        if msg is None:
            raise
        print(f"error: {msg}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
