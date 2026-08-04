#!/usr/bin/env python
"""
Extract the user's hand-drawn white guide curves for Figure 2 PANEL B from the
full-bleed image fig2b_drawn.png (drawn directly on the clean magma heatmap, so
the pixel->data map is the simple linear one used by export_clean_heatmaps.py).

6 curves expected (2 each H+, H2+, H3+).  The H2+-right and H3+-left strokes
converge into a single "V" at the bottom, so they label as ONE connected
component; this splits every row into its left/right runs to recover both sides.

Output: fig2b_drawn_contours.npz  with b0..b5 (open polylines) + nb.
Re-run ONLY if fig2b_drawn.png changes.  build_fig2.py just plots the result.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

XL, YL = (120, 490), (501, 3900)          # data limits of the clean heatmap
RUN_GAP = 5                                # px gap that separates two runs in a row
SPLIT_DATA = 15                            # if the two sides differ by >this (data x),
                                           # treat the component as TWO curves
SMOOTH = 4                                 # along-curve gaussian (px)


def trace_runs(mask):
    """Per row, return (left_run_centre, right_run_centre) polylines."""
    ys, xs = np.where(mask)
    L, R = [], []
    for r in np.unique(ys):
        rx = np.sort(xs[ys == r])
        runs = np.split(rx, np.where(np.diff(rx) > RUN_GAP)[0] + 1)
        cs = [run.mean() for run in runs]
        L.append((r, cs[0])); R.append((r, cs[-1]))
    return np.array(L), np.array(R)


def px2data(px, py, W, H):
    return XL[0] + px / W * (XL[1] - XL[0]), YL[1] - py / H * (YL[1] - YL[0])


def extract():
    a = np.asarray(Image.open("fig2b_drawn.png").convert("RGB")).astype(int)
    H, W, _ = a.shape
    white = (a[..., 0] > 225) & (a[..., 1] > 225) & (a[..., 2] > 225)
    lab, n = ndimage.label(white, structure=np.ones((3, 3)))
    sizes = ndimage.sum(white, lab, range(1, n + 1))
    curves = []
    for cid in [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] > 100]:
        m = (lab == cid)
        L, R = trace_runs(m)
        sep = np.median(np.abs(R[:, 1] - L[:, 1])) / W * (XL[1] - XL[0])
        sides = (L, R) if sep > SPLIT_DATA else (L,)     # one comp may hold two curves
        for arr in sides:
            xd, yd = px2data(arr[:, 1], arr[:, 0], W, H)
            curves.append(np.column_stack([gaussian_filter1d(xd, SMOOTH),
                                           gaussian_filter1d(yd, SMOOTH)]))
    curves.sort(key=lambda c: c[:, 0].mean())
    return curves


if __name__ == "__main__":
    cb = extract()
    np.savez("fig2b_drawn_contours.npz",
             **{f"b{i}": c for i, c in enumerate(cb)}, nb=len(cb))
    print(f"panel b: {len(cb)} curves -> fig2b_drawn_contours.npz")
    for c in cb:
        print(f"  x[{c[:,0].min():.0f},{c[:,0].max():.0f}] mean={c[:,0].mean():.0f}")
