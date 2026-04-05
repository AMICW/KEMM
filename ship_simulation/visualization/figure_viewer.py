"""打开已导出的 matplotlib figure bundle，并可调整 3D 视角后另存。"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt


def load_figure_bundle(path: str | Path):
    bundle_path = Path(path)
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)


def apply_3d_view(fig, elev: float | None = None, azim: float | None = None) -> bool:
    changed = False
    for ax in fig.axes:
        if not hasattr(ax, "view_init"):
            continue
        current_elev = getattr(ax, "elev", None)
        current_azim = getattr(ax, "azim", None)
        ax.view_init(
            elev=current_elev if elev is None else elev,
            azim=current_azim if azim is None else azim,
        )
        changed = True
    return changed


def open_figure_bundle(
    path: str | Path,
    *,
    elev: float | None = None,
    azim: float | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    fig = load_figure_bundle(path)
    apply_3d_view(fig, elev=elev, azim=azim)
    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return Path(save_path) if save_path is not None else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open an exported matplotlib figure bundle.")
    parser.add_argument("path", help="Path to the .fig.pickle bundle.")
    parser.add_argument("--elev", type=float, default=None, help="Optional 3D elevation angle.")
    parser.add_argument("--azim", type=float, default=None, help="Optional 3D azimuth angle.")
    parser.add_argument("--save-path", default="", help="Optional path to save the current view as a static image.")
    parser.add_argument("--no-show", action="store_true", help="Apply view / save without opening an interactive window.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    open_figure_bundle(
        args.path,
        elev=args.elev,
        azim=args.azim,
        save_path=args.save_path or None,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
