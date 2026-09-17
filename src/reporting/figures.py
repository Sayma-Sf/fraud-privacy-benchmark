"""Static README figures, rendered once for a light and once for a dark background."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": "#3987e5",
    },
}
DPI = 200


def _system_sans() -> str:
    """First installed UI sans, so matplotlib doesn't warn once per text element."""
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["Segoe UI", "Helvetica Neue", "Helvetica", "Arial"]
    return next((name for name in candidates if name in installed), "DejaVu Sans")


FONT = _system_sans()


def _frame(ax, theme: dict, *, grid_axis: str) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.tick_params(colors=theme["muted"], labelcolor=theme["secondary"], length=0, labelsize=9)
    ax.grid(axis=grid_axis, color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def _titles(fig, theme: dict, title: str, subtitle: str) -> None:
    """Title block pinned to the top edge in inches, so it holds at any figure height."""
    height = fig.get_figheight()
    fig.text(0.02, 1 - 0.18 / height, title, ha="left", va="top", fontsize=13,
             weight="semibold", color=theme["ink"])
    fig.text(0.02, 1 - 0.47 / height, subtitle, ha="left", va="top", fontsize=9.5,
             color=theme["secondary"])


def privacy_utility(results: dict, path: Path, mode: str) -> None:
    """Mean test AUPRC of DP logistic regression against epsilon, with reference lines."""
    theme = THEMES[mode]
    finite = [r for r in results["dp_sweep"] if r["epsilon"] is not None]
    no_noise = next(r for r in results["dp_sweep"] if r["epsilon"] is None)
    lightgbm = results["baseline"]["lightgbm"]["auprc"]
    eps = [r["epsilon"] for r in finite]

    plt.rcParams["font.family"] = FONT
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=DPI, facecolor=theme["surface"])
    fig.subplots_adjust(left=0.09, right=0.97, top=1 - 1.0 / 4.6, bottom=0.14)
    _frame(ax, theme, grid_axis="y")
    ax.tick_params(which="minor", length=0)

    ax.fill_between(eps, [r["auprc_p10"] for r in finite], [r["auprc_p90"] for r in finite],
                    color=theme["series"], alpha=0.12, linewidth=0)
    ax.plot(eps, [r["auprc_mean"] for r in finite], color=theme["series"], linewidth=2,
            solid_capstyle="round", solid_joinstyle="round", marker="o", markersize=6.5,
            markeredgecolor=theme["surface"], markeredgewidth=1.5, zorder=3)

    # The two references sit close together, so one label goes above its line and one below.
    # They sit at the left edge, where the private curve is far below them.
    references = [
        (lightgbm, f"LightGBM, no privacy: {lightgbm:.2f}", "bottom", 4),
        (no_noise["auprc_mean"],
         f"Same model with no noise (ε = ∞): {no_noise['auprc_mean']:.2f}", "top", -4),
    ]
    for value, label, va, offset in references:
        ax.axhline(value, color=theme["muted"], linewidth=1, zorder=2)
        ax.annotate(label, xy=(0.0, value), xycoords=("axes fraction", "data"),
                    xytext=(6, offset), textcoords="offset points", va=va, fontsize=9,
                    color=theme["secondary"])

    ax.set_xscale("log")
    ax.set_xlim(min(eps) / 1.3, max(eps) * 1.3)
    ticks = [t for t in (0.01, 0.1, 1, 10, 100) if min(eps) <= t <= max(eps)]
    ax.set_xticks(ticks, labels=[f"{t:g}" for t in ticks])
    ax.set_ylim(0, 1)
    ax.set_xlabel("Privacy budget ε (log scale). Smaller = more private", fontsize=9.5,
                  color=theme["secondary"])
    ax.set_ylabel("AUPRC on held-out test set", fontsize=9.5, color=theme["secondary"])

    data = results["dataset"]
    runs = results["config"]["dp_repeats"]
    _titles(fig, theme, "What privacy costs a fraud model",
            f"Differentially private logistic regression: mean of {runs} runs per ε, band = "
            f"10th–90th percentile. Test set: {data['test_frauds']} frauds in "
            f"{data['test_rows']:,} transactions.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def _rounded_hbar(ax, y: float, width: float, height: float, color: str, radius_pt: float = 3):
    """Horizontal bar with a rounded data end and a square end at the baseline.

    Drawn as a single path so there is no seam, with the corner radius converted
    from points into each axis's data units so the corners stay circular.
    """
    fig = ax.figure
    origin = ax.transData.transform((0, 0))
    px_per_x = ax.transData.transform((1, 0))[0] - origin[0]
    px_per_y = abs(ax.transData.transform((0, 1))[1] - origin[1])
    radius_px = radius_pt / 72 * fig.dpi
    rx = min(radius_px / px_per_x, width / 2)
    ry = min(radius_px / px_per_y, height / 2)
    top, bottom = y - height / 2, y + height / 2
    vertices = [(0, top), (width - rx, top), (width, top), (width, top + ry),
                (width, bottom - ry), (width, bottom), (width - rx, bottom), (0, bottom),
                (0, top)]
    codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
             MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(vertices, codes), facecolor=color, edgecolor="none"))


def synthetic_utility(results: dict, path: Path, mode: str) -> None:
    """Test AUPRC of LightGBM trained on real data versus on each synthetic dataset."""
    theme = THEMES[mode]
    names = {"gaussian_copula": "Gaussian copula synthetic", "ctgan": "CTGAN synthetic"}
    rows = [("Real training data", results["baseline"]["lightgbm"]["auprc"])]
    rows += [(names.get(k, k), v["tstr"]["auprc"]) for k, v in results["synthetic"].items()]

    plt.rcParams["font.family"] = FONT
    height = 1.7 + 0.5 * len(rows)
    fig, ax = plt.subplots(figsize=(8, height), dpi=DPI, facecolor=theme["surface"])
    fig.subplots_adjust(left=0.27, right=0.95, top=1 - 0.95 / height, bottom=0.6 / height)
    _frame(ax, theme, grid_axis="x")
    ax.set_xlim(0, 1)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), labels=[r[0] for r in rows])
    ax.tick_params(axis="y", labelsize=9.5, labelcolor=theme["ink"])

    bar_height = 0.42
    for i, (_, value) in enumerate(rows):
        _rounded_hbar(ax, i, value, bar_height, theme["series"])
        ax.text(value + 0.012, i, f"{value:.2f}", va="center", fontsize=9.5, color=theme["ink"])

    ax.set_xlabel("AUPRC on the real held-out test set", fontsize=9.5, color=theme["secondary"])
    _titles(fig, theme, "Train on synthetic, test on real",
            "LightGBM trained only on generated transactions, scored on real ones it never saw.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=theme["surface"])
    plt.close(fig)


def render_all(results: dict, figures_dir: Path) -> list[Path]:
    written = []
    for mode in THEMES:
        for name, draw in (("privacy_utility", privacy_utility),
                           ("synthetic_utility", synthetic_utility)):
            if name == "synthetic_utility" and not results.get("synthetic"):
                continue
            path = figures_dir / f"{name}_{mode}.png"
            draw(results, path, mode)
            written.append(path)
    return written
