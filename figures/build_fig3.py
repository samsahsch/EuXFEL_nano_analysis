#!/usr/bin/env python
"""
Figure 3 : pinwheel 2x2  (a,d wide TOF siblings ; b yields ; c CDI patterns).
Ticks point OUT, no top/right ticks. Field "colorbar" is a log-scaled binned bar
(true bin edges, real 6->10 gap). CDI images fill the full row height; their
shared colorbar spans the true photon-count range.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from figstyle import *

setup_style()

# ============================ PARAMETERS ===================================
M_LEFT, M_RIGHT, M_TOP, M_BOT = 0.42, 0.06, 0.16, 0.34
ROW_GAP, ROW_H, COL_GAP = 0.5, 1.30, 0.44   # row gap clears a's x-label vs c's x-label
GAP_C = 0.08                      # between the two stacked CDI images (room for q_y=0 labels)
CDI_CBAR_GAP, CDI_CBAR_W, CDI_CBAR_LAB = 0.03, 0.06, 0.34

BANDS = [(180, 225, "#1f77b4"), (265, 325, "#d62728"), (347, 393, "#2ca02c")]
BAND_LABELS = ["H$^+$", "H$_2^+$", "H$_3^+$"]
# label y just above each curve peak (H+~5.9, H2+~2.95, H3+~1.48); ylim top gives
# headroom so even the tall H+ label stays inside the axes.
BAND_LABEL_Y = [5.7, 3.30, 1.85]
# label x (data coords); None -> centre of the band.  H+ nudged left off centre.
BAND_LABEL_X = [194, None, None]
TOF_XLIM, TOF_YLIM = (120, 490), (0, 6.7)

# CDI (panel c) q-axes: linear, identical on both images.  q in 1/nm
# (= the reference's 1/A x10: 0.02 A^-1 = 0.2 nm^-1; gives clean 0-0.2 axes).
QX_LIM = (-0.223, 0.221)
QY_LIM = (0.0, 0.215)
QX_TICKS = [-0.2, 0.0, 0.2]
QY_TICKS = [0.0, 0.2]
QX_LABEL = r"$q_x$ (nm$^{-1}$)"
QY_LABEL = r"$q_y$ (nm$^{-1}$)"
QY_LABEL_DX = -0.155                  # ylabel x in axes-frac: lines up with panel a's
CDI_VMIN_LIN = 0.0                    # linear scale; vmax chosen so the rings are as
CDI_VMAX = 6.0e3                      # visible as fig3 copy.png (saved data max ~28k,
CDI_TICKS = [2e3, 4e3, 6e3]           # so honest labels top at 6x10^3, not 5x10^4)

BINS = ["0.2-0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", "3-6", "10-30"]
# true field intervals for the log bar (note the real 6->10 gap)
FIELD_BINS = [(0.2, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 6), (10, 30)]
FIELD_TICKS = [0.2, 0.5, 1, 2, 3, 6, 10, 30]

STROKE = [pe.withStroke(linewidth=1.4, foreground="black")]
# ===========================================================================

FIG_H_IN = M_TOP + 2 * ROW_H + ROW_GAP + M_BOT
fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))


def frac(x, y, w, h):
    return [x / FIG_W_IN, y / FIG_H_IN, w / FIG_W_IN, h / FIG_H_IN]


def tidy(ax):
    ax.tick_params(which="both", top=False, right=False, direction="out")


# ---- geometry (c sets the wide-panel width so images fill the row height) --
Wtot = FIG_W_IN - M_LEFT - M_RIGHT
img_h = (ROW_H - GAP_C) / 2.0
img_w = 2.0 * img_h                                   # CDI arrays are 2:1
c_total = img_w + CDI_CBAR_GAP + CDI_CBAR_W + CDI_CBAR_LAB
wide_w = Wtot - c_total - COL_GAP                     # a & d width
b_w = Wtot - wide_w - COL_GAP                         # = c_total
y_top, y_bot = M_BOT + ROW_H + ROW_GAP, M_BOT

ax_a = fig.add_axes(frac(M_LEFT, y_top, wide_w, ROW_H))
bx0 = FIG_W_IN - M_RIGHT - b_w
ax_b = fig.add_axes(frac(bx0, y_top, b_w, ROW_H))
dx0 = FIG_W_IN - M_RIGHT - wide_w
ax_d = fig.add_axes(frac(dx0, y_bot, wide_w, ROW_H))


def draw_bands(ax):
    for (lo, hi, c), lab, ly, lx in zip(BANDS, BAND_LABELS, BAND_LABEL_Y, BAND_LABEL_X):
        ax.axvspan(lo, hi, color=c, alpha=0.10, lw=0)
        tx = (lo + hi) / 2 if lx is None else lx
        ax.text(tx, ly, lab, color="0.25",
                fontsize=FS, ha="center", va="bottom")


# ------------------------------------------------------------------ panel a
colors = cm.viridis(np.linspace(0.05, 0.95, 7))
for i, R in enumerate(BINS):
    ax_a.plot(load(f"fig3a_x_{R}.npy"), load(f"fig3a_y_{R}.npy"),
              color=colors[i], lw=LW_DATA)
draw_bands(ax_a)
ax_a.set_xlim(*TOF_XLIM); ax_a.set_ylim(*TOF_YLIM)
ax_a.set_xlabel("Time of flight (ns)")
ax_a.set_ylabel("Norm. counts (arb. u.)")
tidy(ax_a)

# binned field bar: 7 EQUAL-width colour segments with a blank slot for the
# missing 6->10 V/nm bin; boundary ticks carry the true edge values.
cax_a = ax_a.inset_axes([0.30, 0.92, 0.66, 0.045])
slot_colors = list(colors[:6]) + [None, colors[6]]      # slot 6 = gap (no data)
for i, col in enumerate(slot_colors):
    if col is not None:
        cax_a.axvspan(i, i + 1, color=col, lw=0)
cax_a.set_xlim(0, 8); cax_a.set_ylim(0, 1); cax_a.set_yticks([])
cax_a.set_xticks(range(9))
cax_a.set_xticklabels(["0.2", "0.5", "1", "1.5", "2", "3", "6", "10", "30"])
cax_a.tick_params(labelsize=FS, length=2, width=LW_AXIS, direction="out", pad=1.5)
for s in cax_a.spines.values():
    s.set_linewidth(LW_AXIS)
cax_a.set_xlabel("Surface electric field (V/nm)", fontsize=FS, labelpad=1.5)

# ------------------------------------------------------------------ panel b
# Yield normalised to the lowest-field bin (raw_yield / raw_yield[0]); err_low is
# the bootstrap error on that ratio (first bin = 0 by construction).  Fit is done
# on the raw yield with its err_raw, then divided by the same reference.
field = load("fig3b_field_centers.npy")
f_sat = lambda E, A, E0, C: A * (1 - np.exp(-E / E0)) + C
f_dec = lambda E, A, E0, C: A * np.exp(-E / E0) + C
Egrid = np.linspace(field.min(), field.max(), 300)
series = [
    ("H$^+$",   "#1f77b4", "o", (0, (4, 2)),          f_sat),
    ("H$_2^+$", "#d62728", "s", (0, (1, 1.5)),        f_dec),
    ("H$_3^+$", "#2ca02c", "^", (0, (4, 1.5, 1, 1.5)), f_sat),
]
for name, c, mk, ls, fn in series:
    raw = load(f"fig3b_raw_yield_{name}.npy")
    err_raw = load(f"fig3b_err_raw_{name}.npy")
    err_low = load(f"fig3b_err_low_{name}.npy")
    ref = raw[0]
    y, e = raw / ref, err_low
    p, _ = curve_fit(fn, field, raw, p0=[raw.max() - raw.min(), 1.5, raw.min()],
                     sigma=np.clip(err_raw, 1e-6, None), absolute_sigma=True,
                     maxfev=20000)
    ax_b.errorbar(field, y, yerr=e, fmt=mk, color=c, ms=4, ls="none",
                  capsize=1.5, elinewidth=0.6, capthick=0.6, label=name, zorder=3)
    ax_b.plot(Egrid, fn(Egrid, *p) / ref, color=c, lw=1.0, ls=ls, zorder=2)
ax_b.set_xlabel("Surface electric field (V/nm)")
ax_b.set_ylabel("Norm. relative yield (arb. u.)")
ax_b.set_ylim(0.7, 1.9)
tidy(ax_b)
ax_b.legend(frameon=False, handlelength=1.4, handletextpad=0.3, columnspacing=0.9,
            loc="upper left", borderaxespad=0.3, ncols=3)

# ------------------------------------------------------------------ panel c
# LINEAR colour scale 0..CDI_VMAX: most pixels are low so the diffraction rings
# (a few hundred counts) show clearly; extend='max' flags the bright core.
# Linear q-axes (nm^-1), identical on both images; the two share GAP_C so the
# q_y=0 / q_y=0.02 labels never collide across the stack.
mono = load("fig3c_monomer.npy"); dim = load("fig3c_dimer.npy")
ax_c_top = fig.add_axes(frac(M_LEFT, y_bot + img_h + GAP_C, img_w, img_h))
ax_c_bot = fig.add_axes(frac(M_LEFT, y_bot, img_w, img_h))
cb_h = 2 * img_h + GAP_C
cax_c = fig.add_axes(frac(M_LEFT + img_w + CDI_CBAR_GAP, y_bot, CDI_CBAR_W, cb_h))
absfmt = plt.matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):g}")  # bare 0.02
for ax, data, lab, bottom in [(ax_c_top, mono, "Monomer", False),
                              (ax_c_bot, dim,  "Dimer",   True)]:
    im = ax.imshow(np.clip(data, 0, None), cmap=MAGMA, vmin=CDI_VMIN_LIN,
                   vmax=CDI_VMAX, aspect="auto",
                   extent=[QX_LIM[0], QX_LIM[1], QY_LIM[0], QY_LIM[1]])
    ax.set_xlim(*QX_LIM); ax.set_ylim(*QY_LIM)
    ax.set_xticks(QX_TICKS); ax.set_yticks(QY_TICKS)
    ax.xaxis.set_major_formatter(absfmt); ax.yaxis.set_major_formatter(absfmt)
    ax.tick_params(which="both", top=False, right=False, direction="out")
    ax.set_ylabel(QY_LABEL)
    # pin the y-label x so it lines up with panel a's y-label column (not pushed
    # out by the wider tick labels)
    ax.yaxis.set_label_coords(QY_LABEL_DX, 0.5)
    if bottom:
        ax.set_xlabel(QX_LABEL, labelpad=2)
    else:
        ax.set_xticklabels([])
    ax.text(0.05, 0.92, lab, transform=ax.transAxes, color="white", fontsize=FS,
            ha="left", va="top", path_effects=STROKE)
# linear colorbar; vmax tuned for ring visibility (see CDI_VMAX note)
cb_c = fig.colorbar(im, cax=cax_c, extend="max")
cb_c.set_label("Photon counts", fontsize=FS, labelpad=2)
cb_c.set_ticks(CDI_TICKS)
cb_c.ax.tick_params(direction="out", labelsize=FS)

# ------------------------------------------------------------------ panel d
xd = load("fig3d_x.npy")
ax_d.plot(xd, load("fig3d_y_mono.npy"),  color="#5a4a3a", lw=1.0, label="Monomer")
ax_d.plot(xd, load("fig3d_y_dimer.npy"), color="#c9a635", lw=1.0, label="Dimer")
draw_bands(ax_d)
ax_d.set_xlim(*TOF_XLIM); ax_d.set_ylim(*TOF_YLIM)
ax_d.set_xlabel("Time of flight (ns)")
ax_d.set_ylabel("Norm. counts (arb. u.)")
tidy(ax_d)
ax_d.legend(frameon=False, handlelength=1.4, handletextpad=0.5, labelspacing=0.3,
            loc="upper right", bbox_to_anchor=(1.015, 1.0))

# ------------------------------------------------------------------ letters
yT, yB = y_top + ROW_H + 0.02, y_bot + ROW_H + 0.02
panel_letter(fig, M_LEFT - 0.36, yT, "a")
panel_letter(fig, bx0 - 0.40, yT, "b")
panel_letter(fig, M_LEFT - 0.36, yB, "c")
panel_letter(fig, dx0 - 0.40, yB, "d")

save_and_report(fig, "fig3", FIG_H_IN)
print(f"  CDI_VMAX = {CDI_VMAX:.0f}   wide_w={wide_w:.3f}  b_w={b_w:.3f}  img_w={img_w:.3f}")
