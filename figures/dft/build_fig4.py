#!/usr/bin/env python
"""
Figure 4 of the H3 manuscript.

Panel a : atomistic structure (struct.png), native aspect preserved.
Panel b : 2x2 grid of PDOS(E-E_f) for four electric fields (0, 2.5, 5, 10 V/nm),
          three curves each: Water O, Bulk O, Si.

Print-final: every dimension / font size below is the literal printed size
(included in LaTeX at width=\\textwidth, 372 pt, no rescaling).

Edit the PARAMETERS block to tweak; then `python build_fig4.py`.
"""
import os
import numpy as np
import ase.io
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage as ndi

# ----------------------------------------------------------------------------
# PARAMETERS  (edit me)
# ----------------------------------------------------------------------------
PT_PER_IN   = 72.27                     # TeX points per inch
FIG_W_PT    = 372.0                     # target column width in TeX pt
FIG_W_IN    = FIG_W_PT / PT_PER_IN      # = 5.147 in
FIG_H_IN    = 2.62                      # total height (reported at the end)

# layout, all in INCHES -------------------------------------------------------
M_LEFT   = 0.05     # left margin (panel a sits flush-ish left)
M_RIGHT  = 0.05     # right margin
M_TOP    = 0.16     # top margin (room for panel letters)
M_BOT    = 0.42     # bottom margin (x tick labels + x axis label)
GAP_AB   = 0.34     # horizontal gap between panel a and panel b grid
A_BOT    = 0.10     # bottom of panel a == bottom of panel b's x-axis LABEL (~0.10 in)
WSPACE   = 0.09     # fractional spacing between b subpanels (of subpanel w)
HSPACE   = 0.09     # fractional spacing between b subpanel rows (of subpanel h)

# typography ------------------------------------------------------------------
FS       = 7        # pt: ticks, labels, legend, annotations
FS_LET   = 8        # pt: panel letters (bold)
LW_DATA  = 0.8      # pt: data curves
LW_AXIS  = 0.6      # pt: spines + ticks

# data ------------------------------------------------------------------------
XLIM        = (-4.2, 4.2)
FIELDS      = [0.0, 2.5, 5.0, 10.0]                 # V/nm, row-major
DIRNAMES    = ['pdos-0', 'pdos-0.25', 'pdos-0.5', 'pdos-1.00']
FERMI       = [-4.68576, -5.00508, -6.36201, -10.44245]
FIELD_LABEL = ['0 V/nm', '2.5 V/nm', '5 V/nm', '10 V/nm']

COL_WATER = "#CC0000"
COL_BULK  = '#0033CC'
COL_SI    = '#CC9966'
# Colours match panel a's atom colours.  Red (Surface O) and tan (Si) collapse to
# the same olive under red-green colour-blindness, so each curve also gets a
# distinct LINE STYLE -- a redundant channel that survives every CVD type and
# grayscale printing while keeping the colour-to-panel-a match.
# Surface O is the one curve in BOTH ambiguous pairs: it collides with Si under
# red-green CVD (both olive) and with Bulk O in grayscale (both dark: L* 42 vs 31).
# Giving ONLY Surface O a dashed style separates it from both, so the other two
# stay solid -- one non-solid line fixes both colour-blind and B&W printing.
LS_WATER = (0, (3.5, 1.5))      # Surface O: dashed (shared member of both collisions)
LS_BULK  = '-'                  # Bulk O: solid
LS_SI    = '-'                  # Si: solid (lightest in gray, distinct by hue)

# panel-a atom legend -- colours + relative sizes SAMPLED from struct.png so the
# legend balls match the real render exactly (H < O < Si; surface O red, bulk O
# blue, Si tan).  Colours are the measured median ball colours; the ball SHADING
# is lifted from a pristine bulk-O ball (see _atom_ball) so each legend sphere
# reproduces the render's own lighting.  Radii are the measured ball radii in
# image pixels (H:O:Si ~ 1:1.9:3.0); surface & bulk O are both oxygen -> same.
ATOM_LEGEND = [
    ("H",         "#E9E8E8", 11.0),   # (label, exact colour, radius in image px)
    ("Surface O", "#D0140F", 20.0),
    ("Bulk O",    "#194AC9", 20.0),
    ("Si",        "#D3B08D", 32.0),
]
LEG_Y_PX    = 60                       # vertical centre of the legend row (px from top)
LEG_TXT_GAP = 8                        # px gap between a ball edge and its OWN label
LEG_GAP     = 48                        # px gap between a label and the NEXT ball
                                       # (constant -> uniform spacing; positions are
                                       # solved from measured text widths, centred)

STRUCT_IMG = 'struct.png'
OUT_PDF    = 'fig4.pdf'
OUT_PNG    = 'fig4.png'

# ----------------------------------------------------------------------------
# GLOBAL STYLE
# ----------------------------------------------------------------------------
# Font: Arial -> Helvetica -> generic sans
_avail = {f.name for f in mpl.font_manager.fontManager.ttflist}
for _f in ('Arial', 'Helvetica', 'Helvetica Neue', 'DejaVu Sans'):
    if _f in _avail:
        FONT_USED = _f
        break
mpl.rcParams.update({
    'font.family'      : 'sans-serif',
    'font.sans-serif'  : [FONT_USED],
    'font.size'        : FS,
    'axes.titlesize'   : FS,
    'axes.labelsize'   : FS,
    'xtick.labelsize'  : FS,
    'ytick.labelsize'  : FS,
    'legend.fontsize'  : FS,
    'axes.linewidth'   : LW_AXIS,
    'lines.linewidth'  : LW_DATA,
    'xtick.direction'  : 'in',
    'ytick.direction'  : 'in',
    'xtick.major.width': LW_AXIS,
    'ytick.major.width': LW_AXIS,
    'xtick.minor.width': LW_AXIS,
    'ytick.minor.width': LW_AXIS,
    'xtick.major.size' : 2.4,
    'ytick.major.size' : 2.4,
    'xtick.minor.size' : 1.3,
    'ytick.minor.size' : 1.3,
    'pdf.fonttype'     : 42,   # embed TrueType (editable, fonts embedded)
    'ps.fonttype'      : 42,
    'svg.fonttype'     : 'none',
})

# ----------------------------------------------------------------------------
# LOAD PDOS DATA  (replicates the notebook: sum of p-orbital PDOS per group)
# ----------------------------------------------------------------------------
atoms = ase.io.read('combo.0001.xyz')
r     = atoms.get_positions()
types = np.asarray(atoms.get_chemical_symbols())

idx_water = np.where((r[:, 2] > 26.3) & (types == 'O'))[0]   # Water O
idx_bulkO = np.where((r[:, 2] < 26.3) & (types == 'O'))[0]   # Bulk (slab) O
idx_si    = np.where((r[:, 2] < 26.3) & (types == 'Si'))[0]  # Si


def load_group(dirname, idx):
    """Mean p-PDOS over a group of atoms; returns (energy_raw, pdos)."""
    acc = None
    eng = None
    for i in idx:
        d = np.loadtxt(os.path.join(dirname, 'atom-%s-L-p.txt' % str(i).zfill(3)))
        eng = d[:, 0]
        acc = d[:, 1] if acc is None else acc + d[:, 1]
    return eng, acc / len(idx)


energies, water, bulk, si = [], [], [], []
for d, ef in zip(DIRNAMES, FERMI):
    e, w = load_group(d, idx_water)
    _, b = load_group(d, idx_bulkO)
    _, s = load_group(d, idx_si)
    energies.append(e - ef)        # shift to E - E_f
    water.append(w); bulk.append(b); si.append(s)

# --- sanity checks ----------------------------------------------------------
for k, e in enumerate(energies):
    assert e.min() <= XLIM[0] and e.max() >= XLIM[1], \
        'energy grid for %s does not cover %s' % (DIRNAMES[k], XLIM)

# global y-max within the x window, rounded up to a clean value
ymax_raw = 0.0
for k in range(len(FIELDS)):
    m = (energies[k] >= XLIM[0]) & (energies[k] <= XLIM[1])
    ymax_raw = max(ymax_raw, water[k][m].max(), bulk[k][m].max(), si[k][m].max())
# round up to 1 significant-ish clean step
step  = 10 ** np.floor(np.log10(ymax_raw))
YMAX  = np.ceil(ymax_raw / (step / 2)) * (step / 2)
print('raw global y-max = %.4g  ->  YMAX = %.4g' % (ymax_raw, YMAX))

# ----------------------------------------------------------------------------
# FIGURE GEOMETRY  (everything computed in inches -> figure fractions)
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN))


def frac(x_in, y_in, w_in, h_in):
    return [x_in / FIG_W_IN, y_in / FIG_H_IN, w_in / FIG_W_IN, h_in / FIG_H_IN]


grid_top_in = FIG_H_IN - M_TOP
grid_h_in   = FIG_H_IN - M_TOP - M_BOT          # height of the 2x2 grid block

# panel a : square (native aspect of struct.png). Spans the FULL visual height
# of panel b -- from the bottom of panel b's x-axis label (A_BOT) up to the top
# of the upper subpanels (grid_top_in). Aspect is locked, so a taller panel a is
# proportionally wider.
img = mpimg.imread(STRUCT_IMG)
img_aspect = img.shape[1] / img.shape[0]        # width / height (=1.0 here)
a_h_in = grid_top_in - A_BOT
a_w_in = a_h_in * img_aspect
ax_a = fig.add_axes(frac(M_LEFT, A_BOT, a_w_in, a_h_in))
ax_a.imshow(img, aspect='auto')                 # box already matches native aspect
ax_a.set_xticks([]); ax_a.set_yticks([])
for sp in ax_a.spines.values():
    sp.set_visible(False)


# pristine bulk-O ball in struct.png -> shared lighting reference for every legend
# ball (the render lights all atoms identically, so re-tinting this one clean ball
# with each species' exact colour reproduces the real balls without bond stubs).
SRC_BALL = (696, 249, 22)   # (row, col, radius) in struct.png pixels


def _atom_ball(hex_color, px=240):
    """Shaded sphere sprite re-tinted from the pristine bulk-O ball, so its colour
    is EXACTLY hex_color carried by the render's own shading + highlight."""
    scy, scx, sR = SRC_BALL
    sub = img[scy - sR:scy + sR, scx - sR:scx + sR, :3]
    lum = 0.2126 * sub[..., 0] + 0.7152 * sub[..., 1] + 0.0722 * sub[..., 2]
    yy, xx = np.mgrid[0:2 * sR, 0:2 * sR]
    src_disk = np.sqrt((yy - sR) ** 2 + (xx - sR) ** 2) <= sR - 0.5
    lum = np.where(src_disk, lum, np.median(lum[src_disk]))   # kill background ring
    shade = ndi.zoom(lum, px / (2 * sR), order=1)
    shade = shade / np.median(shade)                          # base tone -> 1.0
    T = np.array(mpl.colors.to_rgb(hex_color))
    tint = np.clip(T[None, None, :] * shade[..., None], 0, 1)
    spec = np.clip((shade - 1.06) / (shade.max() - 1.06), 0, 1)   # highlight -> white
    rgb = np.clip(tint * (1 - 0.9 * spec[..., None]) + 0.9 * spec[..., None], 0, 1)
    gy, gx = np.mgrid[0:px, 0:px]                              # analytic smooth edge
    rr = np.sqrt((gy - (px - 1) / 2.0) ** 2 + (gx - (px - 1) / 2.0) ** 2)
    alpha = np.clip(((px / 2.0 - 1.5) - rr) / 2.0, 0, 1)
    return np.dstack([rgb, alpha])


# ---- panel-a atom legend: shaded balls (matching struct.png) + species labels
_W_img, _H_img = img.shape[1], img.shape[0]
# lock the data limits first so transData is final while we measure text widths
ax_a.set_xlim(-0.5, _W_img - 0.5); ax_a.set_ylim(_H_img - 0.5, -0.5)
fig.canvas.draw()                                  # ensure a renderer exists
_rend = fig.canvas.get_renderer()
_inv  = ax_a.transData.inverted()


def _label_w_px(s):
    """Width of label `s` in image-pixel (data) units at the legend font size."""
    t = ax_a.text(0, LEG_Y_PX, s, fontsize=FS)
    w_disp = t.get_window_extent(_rend).width
    t.remove()
    (x0, _), (x1, _) = _inv.transform([(0, 0), (w_disp, 0)])
    return abs(x1 - x0)


_labs = [e[0] for e in ATOM_LEGEND]
_cols = [e[1] for e in ATOM_LEGEND]
_rpxs = [e[2] for e in ATOM_LEGEND]
_tws  = [_label_w_px(l) for l in _labs]
# entry width = ball(2r) + gap-to-own-text + text; between entries a CONSTANT
# LEG_GAP separates each label from the next ball -> uniform visual rhythm.
_entry_w = [2 * r + LEG_TXT_GAP + tw for r, tw in zip(_rpxs, _tws)]
_total   = sum(_entry_w) + LEG_GAP * (len(_labs) - 1)
_x = (_W_img - _total) / 2.0                        # centre the whole row
for _lab, _col, _rpx, _ew in zip(_labs, _cols, _rpxs, _entry_w):
    _cx = _x + _rpx
    ax_a.imshow(_atom_ball(_col), aspect='auto', zorder=10, clip_on=False,
                interpolation='antialiased',
                extent=(_cx - _rpx, _cx + _rpx, LEG_Y_PX + _rpx, LEG_Y_PX - _rpx))
    _rim = tuple(0.5 * c for c in mpl.colors.to_rgb(_col))   # crisp edge (helps H)
    ax_a.add_patch(plt.Circle((_cx, LEG_Y_PX), _rpx, fill=False, lw=0.4,
                              ec=_rim, zorder=11, clip_on=False))
    ax_a.text(_cx + _rpx + LEG_TXT_GAP, LEG_Y_PX, _lab, fontsize=FS,
              ha='left', va='center', color='black', zorder=11, clip_on=False)
    _x += _ew + LEG_GAP
# overlay imshows reset the data limits; restore the base image extent
ax_a.set_xlim(-0.5, _W_img - 0.5); ax_a.set_ylim(_H_img - 0.5, -0.5)

# panel b : 2x2 grid in the remaining width
b_left_in = M_LEFT + a_w_in + GAP_AB
b_w_in    = FIG_W_IN - M_RIGHT - b_left_in
sub_w     = b_w_in   / (2 + WSPACE)             # subpanel width  (with 1 gap)
sub_h     = grid_h_in / (2 + HSPACE)            # subpanel height (with 1 gap)
gap_w     = sub_w * WSPACE
gap_h     = sub_h * HSPACE

axes_b = []
for row in range(2):
    for col in range(2):
        x = b_left_in + col * (sub_w + gap_w)
        y = grid_top_in - sub_h - row * (sub_h + gap_h)
        axes_b.append(fig.add_axes(frac(x, y, sub_w, sub_h)))

# ----------------------------------------------------------------------------
# PLOT PANEL b
# ----------------------------------------------------------------------------
for k, ax in enumerate(axes_b):
    ax.plot(energies[k], water[k], color=COL_WATER, ls=LS_WATER, label='Surface O')
    ax.plot(energies[k], bulk[k],  color=COL_BULK,  ls=LS_BULK,  label='Bulk O')
    ax.plot(energies[k], si[k],    color=COL_SI,    ls=LS_SI,    label='Si')

    ax.set_xlim(*XLIM)
    ax.set_ylim(0, YMAX)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.tick_params(which='major', direction='out')
    ax.ticklabel_format(axis='both', style='plain')

    row, col = divmod(k, 2)
    if row != 1:                       # x tick labels only on bottom row
        ax.set_xticklabels([])
    else:                              # bottom row: blank the two INTERIOR edge
        # labels (left panel's '4' and right panel's '-4') so they don't collide
        # across the column gap and read as "4-4".  Outer ±4 extents kept.
        if col == 0:
            ax.set_xticklabels(['-4', '-2', '0', '2', '4'])
        else:
            ax.set_xticklabels(['-4', '-2', '0', '2', '4'])
    if col != 0:                       # y tick labels only on left column
        ax.set_yticklabels([])

    # field annotation, top-right inside
    ax.text(0.96, 0.94, FIELD_LABEL[k], transform=ax.transAxes,
            ha='right', va='top', fontsize=FS)

# shared axis labels
axes_b[2].set_xlabel(r'$E-E_\mathrm{f}$ (eV)')   # bottom-left
axes_b[3].set_xlabel(r'$E-E_\mathrm{f}$ (eV)')   # bottom-right
axes_b[0].set_ylabel('PDOS (a.u.)')
axes_b[2].set_ylabel('PDOS (a.u.)')

# single legend, inside 0 V/nm panel (top-left), below its field annotation
axes_b[0].legend(loc='upper right', bbox_to_anchor=(0.98, 0.84),
                 frameon=False, handlelength=1.1, handletextpad=0.5,
                 labelspacing=0.3, borderaxespad=0.0)

# ----------------------------------------------------------------------------
# PANEL LETTERS  (bold, 8 pt, top-left just outside each panel, common baseline)
# ----------------------------------------------------------------------------
letter_y = grid_top_in + 0.02          # common baseline, just above grid top
fig.text((M_LEFT - 0.02) / FIG_W_IN, letter_y / FIG_H_IN, 'a',
         fontsize=FS_LET, fontweight='bold', ha='left', va='bottom')
fig.text((b_left_in - 0.30) / FIG_W_IN, letter_y / FIG_H_IN, 'b',
         fontsize=FS_LET, fontweight='bold', ha='left', va='bottom')

# ----------------------------------------------------------------------------
# SAVE + VERIFY
# ----------------------------------------------------------------------------
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=600)

print('Font used        : %s' % FONT_USED)
print('Figure size (in) : %.4f x %.4f' % (FIG_W_IN, FIG_H_IN))
print('Figure size (pt) : %.2f x %.2f' % (FIG_W_IN * PT_PER_IN, FIG_H_IN * PT_PER_IN))

# verify PDF page width by reading the /MediaBox directly (no extra deps)
import re
with open(OUT_PDF, 'rb') as fh:
    blob = fh.read()
m = re.search(rb'/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]', blob)
x0, y0, x1, y1 = (float(v) for v in m.groups())
pdf_w_in = (x1 - x0) / 72.0            # PDF points are 1/72 in
print('PDF page width   : %.4f in (target 5.147, |dx|=%.4f)'
      % (pdf_w_in, abs(pdf_w_in - 5.147)))
assert abs(pdf_w_in - 5.147) < 0.01, 'PDF width off!'
print('OK')
