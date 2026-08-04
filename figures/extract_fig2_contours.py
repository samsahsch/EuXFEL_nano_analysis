#!/usr/bin/env python
"""
Extract the user's hand-drawn white guide curves for Figure 2 into data coords.
  panel a : 6 curves (2 each Si+/H2O+/OH+) from fig2a_clean.png  (axes-off, full bleed)
  panel b : 6 curves (2 each H+/H2+/H3+)   from fig2idea.png     (heatmap w/ white border)
Output: fig2_drawn_contours.npz  with a0..a5, b0..b5 (open polylines), na, nb.
Re-run ONLY if the drawings change.  build_fig2.py just plots the result.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d


def extract_panel_a():
    # fig2a_clean.png was saved axes-off, ax filling [0,1] with these data limits
    XL, YL = (1.04, 1.39), (501, 3940)
    a = np.asarray(Image.open("fig2a_clean.png").convert("RGB")).astype(int)
    H, W, _ = a.shape
    white = (a[..., 0] > 225) & (a[..., 1] > 225) & (a[..., 2] > 225)
    lab, n = ndimage.label(ndimage.binary_dilation(white, iterations=2))
    sizes = ndimage.sum(white, lab, range(1, n + 1))
    cur = []
    for cid in [i + 1 for i in np.argsort(sizes)[::-1] if sizes[i] > 80]:
        ys, xs = np.where((lab == cid) & white)
        rows = np.unique(ys)
        px = np.array([xs[ys == r].mean() for r in rows]); py = rows.astype(float)
        xd = XL[0] + px / W * (XL[1] - XL[0]); yd = YL[1] - py / H * (YL[1] - YL[0])
        cur.append(np.column_stack([gaussian_filter1d(xd, 2), gaussian_filter1d(yd, 2)]))
    cur.sort(key=lambda c: c[:, 0].mean())
    return cur


def extract_panel_b():
    # fig2idea.png: lower heatmap inside a white border box; x-calibrated by
    # matching the two strongest bright ridges to the data ridges (H+, H2+).
    a = np.asarray(Image.open("fig2idea.png").convert("RGB")).astype(int)
    r0, r1, c0, c1 = 1370, 2398, 234, 2386          # inside the white border
    box = a[r0:r1, c0:c1]; Hh, Ww, _ = box.shape
    R, G, B = box[..., 0], box[..., 1], box[..., 2]
    white = (R > 230) & (G > 230) & (B > 230)        # B>230 excludes bright magma
    white[:, int(0.95 * Ww):] = False                # drop the colorbar arrow
    white[white.sum(1) > 0.35 * Ww, :] = False        # drop the horizontal streak rows
    lab, n = ndimage.label(white, structure=np.ones((3, 3)))
    SL, IC = 401.9, 116.2                            # data_x = IC + SL*frac
    cur = []
    for cid in range(1, n + 1):
        ys, xs = np.where(lab == cid)
        if len(ys) < 150 or (ys.max() - ys.min()) < 0.5 * Hh:
            continue                                 # keep tall bracketing curves
        if (xs.max() - xs.min()) < 10:
            continue                                 # drop the thin vertical border lines
        rows = np.unique(ys)
        px = np.array([xs[ys == r].mean() for r in rows]); py = rows.astype(float)
        xd = IC + SL * (px / Ww); yd = 3940 - py / Hh * (3940 - 501)
        cur.append(np.column_stack([gaussian_filter1d(xd, 4), gaussian_filter1d(yd, 4)]))
    cur.sort(key=lambda c: c[:, 0].mean())
    return cur


if __name__ == "__main__":
    ca, cb = extract_panel_a(), extract_panel_b()
    np.savez("fig2_drawn_contours.npz",
             **{f"a{i}": c for i, c in enumerate(ca)},
             **{f"b{i}": c for i, c in enumerate(cb)},
             na=len(ca), nb=len(cb))
    print(f"panel a: {len(ca)} curves   panel b: {len(cb)} curves -> fig2_drawn_contours.npz")
