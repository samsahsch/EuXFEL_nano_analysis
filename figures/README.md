# Figure generation

Each manuscript figure is produced by one script here. The scripts read only
the arrays in `data/` (and the schematics in `assets/`), so **Figures 1-3 can be
rebuilt without the raw experimental data**. Fig. 4 is built from the DFT
outputs on Zenodo (https://doi.org/10.5281/zenodo.23747182).

```
cd figures
python build_fig1.py     # -> fig1.pdf, fig1.png
python build_fig2.py     # -> fig2.pdf, fig2.png
python build_fig3.py     # -> fig3.pdf, fig3.png
# Fig. 4: run build_fig4.py inside the unpacked Zenodo record
```

All of Figures 1-3 share `figstyle.py`, which fixes the printed geometry: the
canvas is exactly 372 TeX pt = 5.147 in wide, all text is 7 pt (8 pt bold for
panel letters), and fonts are embedded as editable TrueType. `save_and_report`
re-reads the written PDF's `/MediaBox` and asserts the page width, so a figure
cannot silently drift off the column width.

## Figure to script to data

| Figure | Script | Inputs (`data/`) |
|---|---|---|
| 1a, 1b | `build_fig1.py` | `assets/scheme_without.png`, `assets/setup_without.png` |
| 1c | `build_fig1.py` | `fig1c_x.npy`, `fig1c_histselected_.npy`, `fig1c_heatmap_data.npy` |

Note that Fig. 1c is on a different spectrometer setting and a different run
list from every other panel (7594 V, voltage setting 1), and its x axis is m/q
rather than flight time: `fig1c_x.npy` spans 0-90 in m/q, and
`fig1c_heatmap_data.npy` is the raw counts pivot table, which `build_fig1.py`
normalizes itself.
| 2a | `build_fig2.py` | `fig2a_X_us.npy`, `fig2a_Y_h2o.npy`, `fig2a_hists_norm_h2o.npy`, `fig2a_smoothed_outlines.npz`, `fig2a_white_curves.npz` |
| 2b | `build_fig2.py` | `fig2b_X_ns.npy`, `fig2b_Y_h.npy`, `fig2b_hists_norm_h.npy`, `fig2b_drawn_contours.npz`, `fig2b_white_curves.npz` |
| 3a | `build_fig3.py` | `fig3a_x_<field range>.npy`, `fig3a_y_<field range>.npy` (7 field bins) |
| 3b | `build_fig3.py` | `fig3b_field_centers.npy`, `fig3b_raw_yield_<species>.npy`, `fig3b_err_raw_<species>.npy`, `fig3b_err_low_<species>.npy` |
| 3c | `build_fig3.py` | `fig3c_monomer.npy`, `fig3c_dimer.npy` |
| 3d | `build_fig3.py` | `fig3d_x.npy`, `fig3d_y_mono.npy`, `fig3d_y_dimer.npy` |
| 4a, 4b | `dft/build_fig4.py` | `pdos-*/`, `struct.png`, `combo.0001.xyz` (Zenodo record) |

Helper scripts, run before the build scripts when the inputs are regenerated
from raw data: `extract_contours.py`, `extract_fig2_contours.py`,
`extract_fig2b_drawn.py` trace the guide contours drawn over the Fig. 2
heatmaps; `export_clean_heatmaps.py` writes the heatmap arrays.

`build_fig2.py` draws the warm-dense-matter transition line in white, which is
the published version. Cyan and green were trialled because white can wash out
over the bright Si+ ridge; `python build_fig2.py --variants` renders those
alongside it as `fig2_wdm_cyan` and `fig2_wdm_green`.

## Where the inputs come from

`data/*.npy` is written by the analysis notebooks in `../notebooks/`, which read
the preprocessed per-run HDF5 files. The full chain is:

```
raw EuXFEL data  --scripts/preprocess.py-->  datarun<ID>.h5
                 --notebooks/-------------->  figures/data/*.npy
                 --figures/build_figN.py--->  figN.pdf
```

Panel 3b is verifiable offline: fitting `data/fig3b_raw_yield_*.npy` against
`data/fig3b_err_raw_*.npy` with `euxfel_nano.yields.significance_from_arrays`
returns the amplitude significances quoted in the Supplementary Information
(28.9 sigma for H+, 10.2 for H2+, 8.7 for H3+). See the snippet in the top-level
README.

## `dft/`

`field.py` runs the GPAW calculations of the silica-water interface under an
applied field of 0, 0.25, 0.5, 0.75 and 1.0 V/A; `data/Sio2_Ez_+*.txt` are their
output logs and `data/combo.0001.xyz` the structure. The atom-projected densities
of states that `build_fig4.py` renders as Fig. 4b are on Zenodo
(https://doi.org/10.5281/zenodo.23747182). `dft-md.py` is a GPAW molecular-dynamics driver that no
figure uses.
