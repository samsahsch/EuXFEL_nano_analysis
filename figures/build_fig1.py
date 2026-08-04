#!/usr/bin/env python
"""
Figure 1 : three stacked panels (all the SAME total horizontal extent).
  a  scheme (label-free image + matplotlib 7 pt labels)
  b  setup  (baked labels covered + matplotlib 7 pt labels)
  c  m/q panel: 2D momentum heatmap ON TOP, 1D mass spectrum BELOW (flush),
     colorbar pulled INSIDE the panel width so c is no wider than a/b.
All text is 7 pt (Arial); only the panel letters are 8 pt bold.
"""
import copy
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse, Rectangle
from figstyle import *

setup_style()

# ============================ PARAMETERS ===================================
X0 = 0.52                    # common left edge
M_RIGHT = 0.10               # right margin -> panels fill the column width
W  = FIG_W_IN - X0 - M_RIGHT # TOTAL horizontal extent of every panel
CBAR_GAP, CBAR_W, CBAR_LAB = 0.07, 0.10, 0.36   # colorbar lives inside W
HM_W = W - CBAR_GAP - CBAR_W - CBAR_LAB          # heatmap/spectrum width

M_TOP, GAP_AB, GAP_BC, M_BOT = 0.16, 0.16, 0.24, 0.42
HM_H = SP_H = 1.02
GAP_C = 0.06
LETTER_X = 0.06
# panel a's scheme image is nearly square; if stretched to the full column W it
# towers over b/c.  Cap its HEIGHT here and let its width follow the aspect
# ratio (centred in the column) so all three panels are ~equal vertical thirds.
A_H = 2.0

SCHEME_IMG = asset("scheme_without.png")     # label-free
LOG_VMIN, LOG_VMAX = 3e-6, 2e-4
LINESPACING = 1.4                     # gap between the two lines of \n labels

# scheme labels (text, x, y, ha, va, weight) in panel-a axes fraction (y up)
SCHEME_LABELS = [
    (r"$\mathbf{E}$",        0.35, 0.9, "center", "center", "bold"),
    ("Water\nsplitting",     0.35, 0.83,  "center", "top",    "normal"),
    ("H$_3^+$\nformation",   0.63,  0.83,  "center", "top",    "normal"),
    ("1.88 keV",             0.105, 0.415, "center", "center", "normal"),
]
# setup_without.png is label-free -> all panel-b labels drawn here at 7 pt
SETUP_LABELS = [
    ("Nanoparticle\ninjector", 0.12,  0.78, "center", "center"),
    ("Ion detector",      0.43,  0.95, "center", "center"),
    ("X-ray camera",           0.75, 1.03, "center", "center"),
    ("Electron detector",           0.43,  -0.05, "center", "center"),
    ("1.88 keV",               0.13,  0.23, "center", "bottom"),
]
# scale bar (the user's 500 nm bar): length as a fraction of panel width, plus
# its centre (axes frac).  BAR_LEN is FIXED -- it encodes the 500 nm resolution,
# measured directly off setup_microscope_label.png (273 px of 2256 px content
# width = 0.121; use the label-image's own content width 0.118).  Do NOT retune.
# Placed in the clean white space directly UNDER the field-of-view box (which
# spans x0.56-0.88, y0.28-0.46) so it reads as that box's scale.
BAR_LEN = 0.0456
BAR_CX, BAR_CY = 0.944, 0.31
BAR_LABEL = "500 nm"
# scale-bar styling (all freely editable):
BAR_LW       = 0.9     # thickness of the bar + its end ticks (pt)
BAR_TICK_H   = 0.015   # half-height of the end ticks, in axes-frac (the I-beam caps)
BAR_LABEL_GAP = 0.012  # vertical gap between the bar and the "500 nm" label (axes-frac)

# m/q-panel ovals -- FULLY user-controllable:
#   (name, cx, cy, width, height, lx, ly)
#   cx,cy  = ellipse centre in DATA coords (x=m/q ; y=heatmap row 0..NROWS, NROWS=191,
#            so cy=95 is the vertical middle / p_perp = 0)
#   width  = m/q span      height = row span (191 ~= the full vertical of the heatmap)
#   lx,ly  = label anchor (DATA coords), horizontal text, independently movable.
# Defaults: H2O+ directly above its oval; Si+ slightly above + a touch right; the
# silica fragments at the oval top but pushed RIGHT so they clear the plot top.
SPECIES_C = [("H$_2$O$^+$",  18.5, 100,  3.5,  110, 16, 160),
             ("Si$^+$",      29.5, 95,  10.0, 175, 26, 177),
             ("SiO$^+$",     43.5, 95,  13.0, 183, 38, 177),
             ("Si$_2^+$",    57, 95,  14.0, 183, 64, 177),
             ("Si$_2$O$^+$", 72.5, 95,  17.0, 183, 82, 177)]
# Argon labels: (name, m/q peak, text_x, text_y[axes frac]).  Placed in the empty
# upper band, clear of the tall peaks at m/q~0-2 and ~13, and off the left edge.
# Leader lines drop from each label to its peak.
# NB: write the charge as ^{n{+}} -- braces round the + stop mathtext from adding
# binary-operator space after the digit (otherwise it renders as "Ar^{4 +}").
AR_LINES = [("Ar$^{6{+}}$", 40/6,  4.5, 0.3),
            ("Ar$^{5{+}}$", 8.2,  7.2, 0.55),
            ("Ar$^{4{+}}$", 10.3, 10.5, 0.75),
            ("Ar$^{3{+}}$", 14, 18, 0.90),
            ("Ar$^{2{+}}$", 21, 24.5, 0.65)]

STROKE = [pe.withStroke(linewidth=1.4, foreground="black")]
# ===========================================================================


def comp_crop(path, pad=12):
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    c = Image.alpha_composite(bg, im).convert("RGB")
    a = np.asarray(c).astype(int)
    nw = a.sum(2) < 720
    rs = np.where(nw.any(1))[0]; cs = np.where(nw.any(0))[0]
    r0, r1 = max(0, rs.min() - pad), min(a.shape[0], rs.max() + pad)
    c0, c1 = max(0, cs.min() - pad), min(a.shape[1], cs.max() + pad)
    return np.asarray(c.crop((c0, r0, c1, r1)))


scheme = comp_crop(SCHEME_IMG)
setup = comp_crop(asset("setup_without.png"))
# Panels a and b are label-free images, so centre them in the FULL figure width
# (not in the c-panel band, whose left edge X0 is pushed right to make room for
# c's y-axis labels -- that band offset is what left a/b with more space on the
# left than the right).  Panel a: fixed height A_H, width follows image aspect.
a_h = A_H
a_w = a_h * (scheme.shape[1] / scheme.shape[0])
a_x = (FIG_W_IN - a_w) / 2
b_h = W / (setup.shape[1] / setup.shape[0])
b_x = (FIG_W_IN - W) / 2

y_sp = M_BOT
y_hm = M_BOT + SP_H + GAP_C
c_top = y_hm + HM_H
y_b = c_top + GAP_BC
y_a = y_b + b_h + GAP_AB
FIG_H_IN = y_a + a_h + M_TOP

fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))


def frac(x, y, w, h):
    return [x / FIG_W_IN, y / FIG_H_IN, w / FIG_W_IN, h / FIG_H_IN]


# ---- panel a : scheme (label-free image + matplotlib labels) ---------------
ax_a = fig.add_axes(frac(a_x, y_a, a_w, a_h))
ax_a.imshow(scheme, extent=[0, 1, 0, 1], aspect="auto")     # fills 0..1, upright
ax_a.set_xticks([]); ax_a.set_yticks([]); ax_a.set_xlim(0, 1); ax_a.set_ylim(0, 1)
for s in ax_a.spines.values():
    s.set_visible(False)
for txt, x, y, ha, va, wt in SCHEME_LABELS:
    ax_a.text(x, y, txt, transform=ax_a.transAxes, fontsize=FS, color="black",
              ha=ha, va=va, fontweight=wt, linespacing=LINESPACING)

# ---- panel b : setup (cover baked labels, re-add at 7 pt) ------------------
ax_b = fig.add_axes(frac(b_x, y_b, W, b_h))
ax_b.imshow(setup, extent=[0, 1, 0, 1], aspect="auto")
ax_b.set_xticks([]); ax_b.set_yticks([]); ax_b.set_xlim(0, 1); ax_b.set_ylim(0, 1)
for s in ax_b.spines.values():
    s.set_visible(False)
for txt, x, y, ha, va in SETUP_LABELS:
    ax_b.text(x, y, txt, transform=ax_b.transAxes, fontsize=FS, color="black",
              ha=ha, va=va, zorder=4, linespacing=LINESPACING)
# scale bar drawn as a thin I-beam (end ticks on either side) so it reads as a
# resolution bar; label sits just below.  All geometry from the params above.
xl, xr = BAR_CX - BAR_LEN / 2, BAR_CX + BAR_LEN / 2
ax_b.plot([xl, xr], [BAR_CY, BAR_CY], transform=ax_b.transAxes, color="black",
          lw=BAR_LW, solid_capstyle="butt", zorder=4)
for xe in (xl, xr):                                   # end ticks (I-beam caps)
    ax_b.plot([xe, xe], [BAR_CY - BAR_TICK_H, BAR_CY + BAR_TICK_H],
              transform=ax_b.transAxes, color="black", lw=BAR_LW,
              solid_capstyle="butt", zorder=4)
ax_b.text(BAR_CX, BAR_CY - BAR_TICK_H - BAR_LABEL_GAP, BAR_LABEL,
          transform=ax_b.transAxes, fontsize=FS, color="black", ha="center",
          va="top", zorder=4)

# ---- panel c : heatmap (top) + spectrum (bottom) ---------------------------
x = load("fig1c_x.npy")
hist = load("fig1c_histselected_.npy").astype(float)
hm = load("fig1c_heatmap_data.npy").astype(float)
nrows = hm.shape[0]
hmn = np.ma.masked_invalid(np.where(hm > 0, hm / hm.sum(), np.nan))
cmap = copy.copy(plt.cm.magma); cmap.set_bad("black"); cmap.set_under("black")

ax_hm = fig.add_axes(frac(X0, y_hm, HM_W, HM_H))
cax = fig.add_axes(frac(X0 + HM_W + CBAR_GAP, y_hm, CBAR_W, HM_H))
ax_sp = fig.add_axes(frac(X0, y_sp, HM_W, SP_H))

im = ax_hm.imshow(hmn, cmap=cmap, aspect="auto",
                  norm=LogNorm(vmin=LOG_VMIN, vmax=LOG_VMAX),
                  extent=[x.min(), x.max(), 0, nrows])
ax_hm.set_yticks([22, 62, 102, 142, 182]); ax_hm.set_yticklabels([-80, -40, 0, 40, 80])
ax_hm.set_ylabel(r"$\propto\,p_\perp$")
ax_hm.set_xlim(x.min(), x.max()); ax_hm.set_xticklabels([])
ax_hm.tick_params(direction="out")
cb = fig.colorbar(im, cax=cax); cb.set_label(CBAR_LABEL); cb.ax.tick_params(which='both',direction="out"); cb.ax.minorticks_off()

ax_sp.plot(x, hist / hist.max(), color="black", lw=LW_DATA)
ax_sp.set_xlim(x.min(), x.max()); ax_sp.set_ylim(0, 1.05)
ax_sp.set_xlabel("m/q"); ax_sp.set_ylabel("Norm. counts (arb. u.)")
ax_sp.tick_params(which="both", direction="out")

for name, cx, cy, wmq, hrow, lx, ly in SPECIES_C:
    ax_hm.add_patch(Ellipse((cx, cy), width=wmq, height=hrow, facecolor="none",
                            edgecolor="white", lw=0.7, alpha=0.6,
                            linestyle=(0, (1, 1.4)), zorder=5))
    ax_hm.text(lx, ly, name, color="white", fontsize=FS,
               ha="center", va="center", path_effects=STROKE, zorder=6)

spec = hist / hist.max()
NO_LINE = {"Ar$^{3{+}}$", "Ar$^{2{+}}$"}        # these two: label only, no leader line
for name, mq, tx, ty in AR_LINES:
    if name in NO_LINE:
        ax_sp.text(tx, ty, name, fontsize=FS, color="0.15", ha="center", va="bottom")
        continue
    j = int(np.argmin(np.abs(x - mq)))
    h = spec[max(0, j - 3):j + 4].max()
    # shrinkA=0 -> touches the peak; shrinkB small -> reaches right up to the label
    ax_sp.annotate(name, xy=(mq, h + 0.005), xytext=(tx, ty), fontsize=FS,
                   color="0.15", ha="center", va="bottom",
                   arrowprops=dict(arrowstyle="-", lw=0.5, color="0.35", shrinkA=0, shrinkB=1))

panel_letter(fig, LETTER_X, y_a + a_h + 0.02, "a")
panel_letter(fig, LETTER_X, y_b + b_h + 0.02, "b")
panel_letter(fig, LETTER_X, c_top + 0.02, "c")

save_and_report(fig, "fig1", FIG_H_IN)
