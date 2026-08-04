"""
Export full-bleed magma heatmaps for fig2 panels a & b so you can DRAW the
species enclosing contours directly on them.

How to use:
  1. Open fig2a_clean.png / fig2b_clean.png in any editor.
  2. Draw each contour as a thin solid line in a UNIQUE pure colour, e.g.
        Si+  red (255,0,0)      H2O+ lime (0,255,0)     OH+  cyan (0,255,255)
        H+   red                H2+  lime               H3+  cyan
     (just outline / enclose each yield; one colour per species per panel).
  3. Save as fig2a_drawn.png / fig2b_drawn.png (same pixel size).
  4. Tell me, and I'll extract the curves and map them to data coordinates.

The image pixels map LINEARLY to data:  x_data = XLIM[0] + px/Wpx*(XLIM[1]-XLIM[0]),
y_data = YLIM[1] - py/Hpx*(YLIM[1]-YLIM[0])   (py measured from the top).
"""
import numpy as np
import matplotlib.pyplot as plt

PANELS = [
    ("fig2a", "fig2a_X_us.npy", "fig2a_Y_h2o.npy", "fig2a_hists_norm_h2o.npy",
     (1.04, 1.39), None),
    ("fig2b", "fig2b_X_ns.npy", "fig2b_Y_h.npy", "fig2b_hists_norm_h.npy",
     (120, 490), (501, 3900)),
]

for name, fx, fy, fc, xlim, ylim in PANELS:
    X = np.load(fx); Y = np.load(fy); C = np.load(fc)
    if ylim is None:
        ylim = (Y.min(), Y.max())
    fig = plt.figure(figsize=(10.0, 4.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(X, Y, C, cmap="magma", vmin=0, vmax=1, shading="auto")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    fig.savefig(f"{name}_clean.png", dpi=200)
    plt.close(fig)
    print(f"{name}_clean.png  XLIM={xlim}  YLIM=({ylim[0]:.0f},{ylim[1]:.0f})")
