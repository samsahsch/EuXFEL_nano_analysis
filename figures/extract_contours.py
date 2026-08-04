"""Extract the hand-drawn contours (diff of *_drawn vs *_clean) -> data coords."""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

PANELS = {
    "a": dict(xlim=(1.04, 1.39), ylim=(501, 3940)),
    "b": dict(xlim=(120, 490),   ylim=(501, 3900)),
}


def extract(panel):
    cl = np.asarray(Image.open(f"fig2{panel}_clean.png").convert("RGB")).astype(int)
    dr = np.asarray(Image.open(f"fig2{panel}_drawn.png").convert("RGB")).astype(int)
    H, W, _ = cl.shape
    xlim = PANELS[panel]["xlim"]; ylim = PANELS[panel]["ylim"]
    mask = np.abs(cl - dr).sum(2) > 60
    # thicken slightly then label connected curves
    mask = ndimage.binary_dilation(mask, iterations=1)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    curves = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if len(xs) < 120:                       # ignore specks
            continue
        # order along the dominant axis; these curves are ~single-valued in y
        if ys.max() - ys.min() >= xs.max() - xs.min():
            rows = np.unique(ys)
            px = np.array([xs[ys == r].mean() for r in rows]); py = rows.astype(float)
        else:
            cols = np.unique(xs)
            py = np.array([ys[xs == c].mean() for c in cols]); px = cols.astype(float)
            order = np.argsort(px); px, py = px[order], py[order]
        # map pixels -> data
        xd = xlim[0] + px / W * (xlim[1] - xlim[0])
        yd = ylim[1] - py / H * (ylim[1] - ylim[0])
        # light smoothing
        if len(xd) > 5:
            xd = gaussian_filter1d(xd, 2.0); yd = gaussian_filter1d(yd, 2.0)
        curves.append(np.column_stack([xd, yd]))
    # sort curves left->right by mean x
    curves.sort(key=lambda c: c[:, 0].mean())
    print(f"panel {panel}: {len(curves)} curves, lengths {[len(c) for c in curves]}")
    return curves


allc = {p: extract(p) for p in PANELS}
np.savez("fig2_contours.npz",
         **{f"{p}_{i}": c for p in allc for i, c in enumerate(allc[p])},
         counts=np.array([len(allc["a"]), len(allc["b"])]))
print("saved fig2_contours.npz")
