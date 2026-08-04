# X-ray Driven Trihydrogen Formation on Silica Nanosurfaces

Analysis, figure-generation and simulation code for the single-particle X-ray
study of trihydrogen (H₃⁺) formation on hydrated silica nanoparticles.

Experiment: European XFEL proposal 3408, SQS instrument, NQS endstation, June 2023.

- **Paper**: *X-ray Driven Trihydrogen Formation on Silica Nanosurfaces*
- **Raw data**: <https://doi.org/10.22003/XFEL.EU-DATA-003408-00>
  (under embargo until publication)

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Rebuild Figures 1-3 — no experimental data needed.
cd figures && python build_fig3.py
```

The figure scripts read only the arrays committed under `figures/data/`, so
the main figures are reproducible today, while the raw data is embargoed.

To rerun the analysis itself you need the preprocessed per-run files; point
`EUXFEL_NANO_DATA` at the directory holding them:

```bash
export EUXFEL_NANO_DATA=/path/to/preprocess
jupyter lab notebooks/figure3.ipynb
```

## Layout

```
src/euxfel_nano/     Analysis library (importable, single definition of each routine)
config/              Run IDs and every analysis parameter, as YAML
notebooks/           One notebook per manuscript figure
figures/             Figure-generation scripts + the arrays they consume
scripts/preprocess.py  Raw European XFEL run -> per-run HDF5
models/              Standalone electrostatic and ionization models
```

## Pipeline

```
raw EuXFEL run
  │  scripts/preprocess.py --run <ID> --outdir <dir>
  ▼
datarun<ID>.h5            per-event and per-pulse tables, eTOF and pnCCD arrays
  │  notebooks/figureN.ipynb   (uses src/euxfel_nano)
  ▼
figures/data/*.npy        figure inputs
  │  figures/build_figN.py
  ▼
figN.pdf
```

`scripts/preprocess.py` needs roughly 300 GB of memory per run and the European
XFEL `extra_data` library, and will not work for runs where the Timepix3 camera
dropped two or more consecutive trains at the start or end.

## Figures

| Figure | Notebook | Build script |
|---|---|---|
| 1 | `notebooks/figure1.ipynb` | `figures/build_fig1.py` |
| 2 | `notebooks/figure2.ipynb` | `figures/build_fig2.py` |
| 3 | `notebooks/figure3.ipynb` | `figures/build_fig3.py` |
| 4 | — (GPAW / QXMD) | `figures/dft/build_fig4.py` |
| S10, S11 | `notebooks/supplementary.ipynb` | — |
| S13 | `notebooks/classifier.ipynb` | — |

See `figures/README.md` for the panel-by-panel input map.

## Analysis parameters

Every number a figure depends on is in `config/params.yaml` (calibration
constants, species windows, field bins, ion-count windows, bootstrap settings,
classifier hyperparameters) or `config/runs.yaml` (run IDs and measured
particle diameters), rather than being repeated in notebook cells.

```python
from euxfel_nano import config
config.run_ids("300")            # run IDs for the 300 nm sample
config.field_bins()              # surface field bins, V/nm
config.measured_diameters_m()    # TEM-measured diameters
```

### Particle diameters

The nominal sizes (500, 300, 100 nm SiO₂ and 150 nm Au@SiO₂) are labels only.
Every electric field is computed from the TEM-measured diameters — 504, 312,
126 and 147 nm — which is what `config.measured_diameters_m()` returns and what
`euxfel_nano.fields.build_charge_maps` uses by default.

### TOF calibration

Flight time and mass-over-charge are related by SI Eq. 1,
`tof = t₀ + k·√(m/q)`, with the exponent held at its physical value of 2. Two
voltage settings were used, both tabulated in `config/params.yaml`:

| Setting | Extraction | t₀ | k | Used for |
|---|---|---|---|---|
| 1 | 7594 V | 7.451 ns | 2.0648 × 10⁻⁷ s | Fig. 1c only |
| 2 | 5400 V | 6.82 ns | 2.450 × 10⁻⁷ s | all remaining figures |

Fig. 1c is the only panel on setting 1, and it uses its own run list
(`sio2_500nm.runs.setting1_7594V` in `config/runs.yaml`). It is also the only
panel whose x axis is m/q rather than flight time.

The analysis also fits an unconstrained power law `m/q = a·tof^b`, which
absorbs the nonzero t₀ into a free exponent and returns b = 2.018 (setting 1)
and b = 2.013 (setting 2) — the values the SI quotes. Within a given setting
the two forms agree to about a nanosecond across the argon series, so the
choice of form is immaterial; the choice of *setting* is not, as the two differ
by ~19 % in flight time at fixed m/q. Both parameterisations are recorded in
`config/params.yaml` against the setting each reproduces, and
`notebooks/figure1.ipynb` prints the agreement.

## Uncertainties

The independent statistical unit is the X-ray shot, not the individual ion:
ions from one shot share that shot's field and particle type, and the four
particle types are pooled within each field bin. Errors therefore come from
resampling shots with replacement (4000 resamples, seed 0), not from an
ion-level binomial, which underestimates the spread by 1.4-2.4×.

`euxfel_nano.yields.verify_errors` runs the validation reported in the SI:
label-shuffling recovers the binomial limit, a delete-one-shot jackknife agrees
with the bootstrap, the normalized errors match analytic ratio propagation, and
the result is converged in the number of resamples.

The headline significances are reproducible offline from the committed arrays,
with no embargoed data:

```python
import numpy as np
from euxfel_nano.yields import significance_from_arrays

field = np.load("figures/data/fig3b_field_centers.npy")
for sp in ("H$^+$", "H$_2^+$", "H$_3^+$"):
    point = np.load(f"figures/data/fig3b_raw_yield_{sp}.npy")
    err   = np.load(f"figures/data/fig3b_err_raw_{sp}.npy")
    print(sp, significance_from_arrays(point, err, field)[2])
# H+ 28.9σ, H2+ 10.2σ, H3+ 8.7σ
```

## Reproducibility

Bootstrap uncertainties are fully deterministic (seeded generator, default
seed 0). The monomer/dimer CNN is not bit-reproducible from a retrain: the
split is seeded, but GPU kernel scheduling makes fitted weights vary between
runs. Call `euxfel_nano.seeds.set_all_seeds()` before training to remove every
controllable source of randomness; the classification reported in the paper
comes from the deposited weights, which are the authoritative artifact.

## Simulations

`figures/dft/` drives the first-principles work: `field.py` and `dft-md.py` run
the GPAW density-functional and QXMD nonadiabatic quantum molecular dynamics
calculations of the silica-water interface under an applied field, and
`build_fig4.py` renders the resulting partial densities of states as Fig. 4b.

`models/` holds standalone analytic models used as cross-checks:
electrostatics of charged and paired nanoparticles, and estimates of X-ray
ionization yields for monomer and dimer configurations.

## Citation

See `CITATION.cff`. Please cite the paper.

## License

BSD 3-Clause — see `LICENSE`.
