"""
Shared print-final style for Figures 1-3.

Every dimension / font size is the literal printed size: the figures go into
LaTeX at width=\\textwidth = 372 TeX pt = 5.147 in with no rescaling.

Import this at the top of each build_figN.py:
    from figstyle import *
    setup_style()
"""
import re
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- data -----------------------------------------------------------------
# Figure inputs are the .npy/.npz intermediates written by the analysis
# notebooks, and the schematic images. They sit next to these scripts so
# Figures 1-3 can be rebuilt without the raw data, which is under embargo
# until publication.
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE / "assets"


def load(name):
    """Load a figure input array from figures/data/."""
    return np.load(DATA / name)


def asset(name):
    """Path to a schematic image in figures/assets/."""
    return str(ASSETS / name)


# ---- canvas ---------------------------------------------------------------
PT_PER_IN = 72.27
FIG_W_IN  = 372.0 / PT_PER_IN          # = 5.1474 in  (column width)

# ---- typography (points) --------------------------------------------------
FS      = 7        # everything: ticks, labels, colorbar, legend, annotations
FS_LET  = 8        # panel letters (bold)
LW_DATA = 0.8      # data curves
LW_AXIS = 0.6      # spines + ticks

# ---- shared text / colors -------------------------------------------------
CBAR_LABEL = "Norm. counts (arb. u.)"          # identical on every figure
MAGMA      = "magma"

# white vertical guide lines over heatmaps
GUIDE_KW = dict(color="white", alpha=0.55, linewidth=0.7,
                linestyle=(0, (3, 3)), zorder=5)

FONT_USED = None


def setup_style():
    """Set global rcParams. Returns the font family actually used."""
    global FONT_USED
    avail = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for f in ("Arial", "Helvetica", "Helvetica Neue", "DejaVu Sans"):
        if f in avail:
            FONT_USED = f
            break
    mpl.rcParams.update({
        "font.family"       : "sans-serif",
        "font.sans-serif"   : [FONT_USED],
        "font.size"         : FS,
        "axes.titlesize"    : FS,
        "axes.labelsize"    : FS,
        "xtick.labelsize"   : FS,
        "ytick.labelsize"   : FS,
        "legend.fontsize"   : FS,
        "axes.linewidth"    : LW_AXIS,
        "lines.linewidth"   : LW_DATA,
        "xtick.direction"   : "in",
        "ytick.direction"   : "in",
        "xtick.major.width" : LW_AXIS,
        "ytick.major.width" : LW_AXIS,
        "xtick.minor.width" : LW_AXIS,
        "ytick.minor.width" : LW_AXIS,
        "xtick.major.size"  : 2.4,
        "ytick.major.size"  : 2.4,
        "xtick.minor.size"  : 1.3,
        "ytick.minor.size"  : 1.3,
        "pdf.fonttype"      : 42,     # embed TrueType, editable
        "ps.fonttype"       : 42,
        "svg.fonttype"      : "none",
    })
    return FONT_USED


def panel_letter(fig, x_in, y_in, letter):
    """Bold 8 pt lowercase panel letter at (x_in, y_in) inches from bottom-left."""
    fig_w, fig_h = fig.get_size_inches()
    fig.text(x_in / fig_w, y_in / fig_h, letter,
             fontsize=FS_LET, fontweight="bold", ha="left", va="bottom")


def guide_lines(ax, positions, ymin=None, ymax=None):
    """Draw white dashed vertical guide lines at the given x positions."""
    for x in positions:
        ax.axvline(x, **GUIDE_KW)


def verify_pdf_width(path, target_in=5.147, tol=0.01):
    """Read the PDF /MediaBox and assert page width == target within tol."""
    blob = open(path, "rb").read()
    m = re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]",
                  blob)
    x0, y0, x1, y1 = (float(v) for v in m.groups())
    w_in = (x1 - x0) / 72.0
    ok = abs(w_in - target_in) < tol
    print(f"  PDF width = {w_in:.4f} in  (target {target_in}, |dx|={abs(w_in-target_in):.4f})"
          f"  {'OK' if ok else 'FAIL'}")
    assert ok, f"PDF width {w_in} off target {target_in}"
    return w_in


def save_and_report(fig, stem, height_in):
    """Save <stem>.pdf (vector) + <stem>.png (600 dpi) and verify width."""
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=600)
    print(f"[{stem}] font={FONT_USED}  size={FIG_W_IN:.4f} x {height_in:.4f} in"
          f"  ({FIG_W_IN*PT_PER_IN:.1f} x {height_in*PT_PER_IN:.1f} pt)")
    verify_pdf_width(f"{stem}.pdf")
