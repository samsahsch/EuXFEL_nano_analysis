#!/usr/bin/env python
"""
Figure 2 (NEW / trial version -- kept separate from build_fig2.py so the current
figure is untouched until we decide to merge).

Same two stacked heatmaps as build_fig2.py, but with ONE addition:

  * A single horizontal DASHED line at Hit intensity = WDM_Y marks the
    warm-dense-matter (1 MJ/kg) threshold.  No band -- a band would imply a
    marginal intensity window, but the result is window-invariant, so one line
    says exactly what is needed.
  * The identical line is drawn in BOTH panels (they share the intensity axis,
    so it reads as one level).  It is labelled "WDM transition" in panel b;
    in panel a the existing "1 MJ/kg" arrow already explains it, so no second
    label is added there (the arrow explains the line instead of competing).

The published figure draws the WDM line in white. Cyan and green were trialled
because white can wash out over the bright Si+ ridge; pass --variants to render
those alongside it for comparison.
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import gaussian_filter1d
from figstyle import *

setup_style()

# ============================ PARAMETERS ===================================
FIG_H_IN = 4.70

# layout (inches)
M_LEFT, M_RIGHT, M_TOP, M_BOT = 0.52, 0.05, 0.18, 0.36
GAP, CBAR_GAP, CBAR_W = 0.7, 0.06, 0.10

XLIM_A, XLIM_B = (1.04, 1.39), (120, 490)
YLIM_B = (501, 3900)
VMAX = 1.0

# --- contour sources -------------------------------------------------------
OUTLINES_A = "fig2a_smoothed_outlines.npz"   # per-species {x_left,x_right,y} bands
MASKS_A = ["mask_1", "mask_2", "mask_3"]     # Si+, H2O+, OH+
CONTOUR_NPZ_B = "fig2b_drawn_contours.npz"   # hand-drawn strokes b0..b5 (3 pairs)
CURVE_SMOOTH = 1.0                           # extra light along-curve smoothing (px)

# species name labels (text, x, y) in data coords -- freely movable
LABELS_A = [("Si$^+$", 1.17, 3500), ("H$_2$O$^+$", 1.185, 1500), ("OH$^+$", 1.08, 1500)]
LABELS_B = [("H$^+$", 180, 3500), ("H$_2^+$", 325, 3500), ("H$_3^+$", 445, 3500)]

KE_TICKPOS, KE_TICKLAB = [209.2, 227.0], ["150", "0"]
KE_LABEL_XY = (0.265, 1.140)   # axes-frac: centred over the 150/0 ticks, above them

# panel-a annotation: a thick white arrow pointing at the kink where the Si+ band
# bends from vertical to diagonal.  All in data coords -- freely movable.
KINK_XY      = (1.142, 2600)   # arrow head: the kink point on the Si+ band
KINK_TEXT_XY = (1.1, 3280)     # tail / label anchor (nudged up so it clears the
                               # WDM line at 3000; the arrow shaft now crosses the
                               # line, so "1 MJ/kg" reads as the line's caption)
KINK_LABEL   = "1 MJ/kg"

# --- WDM transition line ---------------------------------------------------
WDM_Y        = 2500.0                 # hit-intensity level of the 1 MJ/kg threshold
WDM_LABEL    = "WDM transition"
# label anchors (data coords) -- above the line and over the DARK background so
# the text never sits on the bright species features underneath.
WDM_LABEL_XY_A = (1.050, 2560)        # panel a: dark strip left of the Si+ ridge
WDM_LABEL_XY_B = (126,   2560)        # panel b: dark strip left of the H+ ridge
WDM_LW       = 0.9                     # thinner -> a bit less prominent
WDM_ALPHA    = 0.9                     # slightly translucent -> less prominent
WDM_DASH     = (0, (4, 3))            # dashed
# colour variants to render (name -> line/label colour over the magma map)
WDM_VARIANTS = {
    "white": "#FFFFFF",
    "cyan":  "#00E5EE",
    "green": "#00FF7F",
}
# light dark halo (subtle) so the line stays legible where it crosses a ridge
WDM_STROKE = [pe.Stroke(linewidth=WDM_LW + 0.9, foreground="black", alpha=0.4),
              pe.Normal()]

# half-transparent white dotted guide curves with a faint dark halo
ENV_KW = dict(color="white", lw=0.9, ls=(0, (1, 1.4)), alpha=0.9, zorder=5)
STROKE = [pe.withStroke(linewidth=1.4, foreground="black")]
ENV_STROKE = [pe.Stroke(linewidth=1.7, foreground="black", alpha=0.4), pe.Normal()]
# ===========================================================================

Xa = load("fig2a_X_us.npy");  Ya = load("fig2a_Y_h2o.npy")
Ca = load("fig2a_hists_norm_h2o.npy")
Xb = load("fig2b_X_ns.npy");  Yb = load("fig2b_Y_h.npy")
Cb = load("fig2b_hists_norm_h.npy")

# ---- panel a contours: each mask -> a (left, right) pair of polylines ------
_OA = load(OUTLINES_A)
CURVES_A = []
for m in MASKS_A:
    y = _OA[f"{m}_y"]
    CURVES_A.append(np.column_stack([_OA[f"{m}_x_left"],  y]))
    CURVES_A.append(np.column_stack([_OA[f"{m}_x_right"], y]))

# ---- panel b contours: the 6 hand-drawn strokes ---------------------------
CLIP_B_YMIN = {4: 603}
_CB = load(CONTOUR_NPZ_B)
CURVES_B = []
for i in range(int(_CB["nb"])):
    c = _CB[f"b{i}"]
    if i in CLIP_B_YMIN:
        c = c[c[:, 1] >= CLIP_B_YMIN[i]]
    CURVES_B.append(c)


def smooth_curve(c):
    """Light non-periodic smoothing of an open hand-drawn stroke."""
    if len(c) < 5:
        return c
    x = gaussian_filter1d(c[:, 0], CURVE_SMOOTH, mode="nearest")
    y = gaussian_filter1d(c[:, 1], CURVE_SMOOTH, mode="nearest")
    return np.column_stack([x, y])


def frac(x, y, w, h):
    return [x / FIG_W_IN, y / FIG_H_IN, w / FIG_W_IN, h / FIG_H_IN]


def draw_heatmap(fig, ax, cax, X, Y, C, xlim, ylim, xlabel, curves, labels):
    pcm = ax.pcolormesh(X, Y, C, cmap=MAGMA, shading="auto", vmin=0, vmax=VMAX)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel); ax.set_ylabel("Hit intensity (arb. u.)")
    ax.tick_params(which="both", top=False, right=False, direction="out")
    cb = fig.colorbar(pcm, cax=cax); cb.set_label(CBAR_LABEL)
    cb.ax.tick_params(direction="out")
    for c in curves:                                   # the guide lines
        c = smooth_curve(c)
        ax.plot(c[:, 0], c[:, 1], path_effects=ENV_STROKE, **ENV_KW)
    for name, lx, ly in labels:
        ax.text(lx, ly, name, color="white", fontsize=FS, ha="center",
                va="center", zorder=6, path_effects=STROKE)
    return pcm


def build(wdm_color, stem):
    """Build the whole figure with the WDM line drawn in `wdm_color`; save `stem`."""
    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))

    plot_w = FIG_W_IN - M_LEFT - M_RIGHT - CBAR_GAP - CBAR_W - 0.34
    panel_h = (FIG_H_IN - M_TOP - M_BOT - GAP) / 2.0
    y_b, y_a = M_BOT, M_BOT + panel_h + GAP

    ax_a  = fig.add_axes(frac(M_LEFT, y_a, plot_w, panel_h))
    cax_a = fig.add_axes(frac(M_LEFT + plot_w + CBAR_GAP, y_a, CBAR_W, panel_h))
    ax_b  = fig.add_axes(frac(M_LEFT, y_b, plot_w, panel_h))
    cax_b = fig.add_axes(frac(M_LEFT + plot_w + CBAR_GAP, y_b, CBAR_W, panel_h))

    draw_heatmap(fig, ax_a, cax_a, Xa, Ya, Ca, XLIM_A, (Ya.min(), Ya.max()),
                 "Time of flight (µs)", CURVES_A, LABELS_A)
    draw_heatmap(fig, ax_b, cax_b, Xb, Yb, Cb, XLIM_B, YLIM_B,
                 "Time of flight (ns)", CURVES_B, LABELS_B)

    # ---- WDM transition line: identical level in both panels ---------------
    for ax in (ax_a, ax_b):
        ax.axhline(WDM_Y, color=wdm_color, lw=WDM_LW, ls=WDM_DASH,
                   alpha=WDM_ALPHA, zorder=6, path_effects=WDM_STROKE)
    # "WDM transition" label above the line in BOTH panels, over dark background
    for ax, xy in ((ax_a, WDM_LABEL_XY_A), (ax_b, WDM_LABEL_XY_B)):
        ax.text(*xy, WDM_LABEL, color=wdm_color, fontsize=FS, ha="left",
                va="bottom", zorder=7,
                path_effects=[pe.withStroke(linewidth=1.6, foreground="black")])

    # ---- panel a : "1 MJ/kg" arrow pointing at the Si+ band kink -----------
    ax_a.annotate(KINK_LABEL, xy=KINK_XY, xytext=KINK_TEXT_XY,
                  color="white", fontsize=FS, ha="center", va="center",
                  zorder=7, path_effects=STROKE,
                  arrowprops=dict(arrowstyle="-|>", color="white", lw=1.6,
                                  mutation_scale=12, shrinkA=4, shrinkB=2,
                                  path_effects=[pe.withStroke(linewidth=2.6,
                                                              foreground="black")]))

    # ---- panel b red KE top axis ------------------------------------------
    ax2 = ax_b.twiny()
    ax2.set_xlim(ax_b.get_xlim())
    ax2.set_xticks(KE_TICKPOS); ax2.set_xticklabels(KE_TICKLAB)
    ax2.tick_params(axis="x", which="both", colors="red", direction="out",
                    top=True, labelsize=FS)
    ax2.minorticks_off()
    ax2.spines["top"].set_color("red"); ax2.spines["top"].set_linewidth(LW_AXIS)
    ax2.spines["top"].set_bounds(min(KE_TICKPOS), max(KE_TICKPOS))
    ax_b.spines["top"].set_visible(True); ax_b.spines["top"].set_linewidth(LW_AXIS)
    for s in ("bottom", "left", "right"):
        ax2.spines[s].set_visible(False)
    ax2.set_xlabel("H$^+$ kinetic energy (eV)", color="red")
    ax2.xaxis.set_label_coords(*KE_LABEL_XY)

    # ---- panel letters ----------------------------------------------------
    panel_letter(fig, M_LEFT - 0.46, y_a + panel_h + 0.02, "a")
    panel_letter(fig, M_LEFT - 0.46, y_b + panel_h + 0.02, "b")

    save_and_report(fig, stem, FIG_H_IN)
    plt.close(fig)


PUBLISHED_VARIANT = "white"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--variants", action="store_true",
                    help="also render the cyan and green WDM-line trials")
    args = ap.parse_args()

    build(WDM_VARIANTS[PUBLISHED_VARIANT], "fig2")
    if args.variants:
        for _name, _col in WDM_VARIANTS.items():
            if _name != PUBLISHED_VARIANT:
                build(_col, f"fig2_wdm_{_name}")
