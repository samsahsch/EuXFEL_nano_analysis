"""Field-binned hydrogen ion yields, shot-level bootstrap errors and fits.

This module produces the relative H+, H2+ and H3+ yields of Fig. 3b and their
uncertainties, following the "Yield Uncertainties and Significance" section of
the Supplementary Information.

The statistically independent unit is the X-ray shot, not the individual ion:
ions recorded in one shot share that shot's field and particle type, and the
four particle types are pooled within each field bin. An ion-level binomial
error would treat correlated ions as independent and underestimate the spread,
so errors come from resampling *shots* with replacement.

The notebook version of this analysis kept its inputs in module-level globals.
Here they are collected into :class:`FieldBinnedData` and passed explicitly;
the arithmetic is unchanged.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2 as _chi2

# Default number of bootstrap resamples. The published errors used 4000, which
# check [5] of verify_errors shows to be converged.
N_BOOT = 4000

# Default RNG seed, so the published error bars are exactly reproducible.
SEED = 0


def sat_func(E, A, E0, C):
    """Saturating field dependence, used for H+ and H3+."""
    return A * (1 - np.exp(-E / E0)) + C


def dec_func(E, A, E0, C):
    """Decaying field dependence, used for H2+."""
    return A * np.exp(-E / E0) + C


@dataclass
class FieldBinnedData:
    """Per-particle-type event tables together with their per-shot fields.

    Parameters
    ----------
    event_dfs
        ``{particle_key: dfevent}``, each with ``pulseId`` and ``tof`` columns.
    charge_maps
        ``{particle_key: pandas.Series}`` mapping ``pulseId`` to the per-shot
        surface electric field in V/nm (see :mod:`euxfel_nano.fields`).
    charge_windows
        Field bin edges as ``[(low, high), ...]`` in V/nm.
    tof_range
        ``(low, high)`` flight-time window in seconds defining the denominator
        of the pooled fraction, i.e. the full analysis window.
    keys
        Particle types to pool, in a fixed order.
    """

    event_dfs: dict
    charge_maps: dict
    charge_windows: list
    tof_range: tuple
    keys: tuple = ("500", "300", "100", "cs")

    @property
    def field_centers(self):
        """Midpoint of each field bin, the x axis of Fig. 3b."""
        return np.array([0.5 * (low + high) for (low, high) in self.charge_windows])


def get_pershot_counts(data, species_range):
    """For each field bin, return per-shot ``(in_peak, total)`` count arrays.

    The four particle types are pooled by concatenating their shots.
    ``species_range`` is the species flight-time window in nanoseconds.
    """
    t0, t1 = species_range[0] * 1e-9, species_range[1] * 1e-9
    per_bin = []
    for (low, high) in data.charge_windows:
        inpk_list, tot_list = [], []
        for key in data.keys:
            cmap = data.charge_maps[key]
            sel = cmap.index[(cmap.values >= low) & (cmap.values < high)].values
            ev = data.event_dfs[key][data.event_dfs[key]["pulseId"].isin(sel)]
            ev = ev[(ev.tof >= data.tof_range[0]) & (ev.tof <= data.tof_range[1])]
            if len(ev) == 0:
                continue
            in_peak = ((ev["tof"].values >= t0) & (ev["tof"].values <= t1)).astype(int)
            g = ev.assign(_ip=in_peak).groupby("pulseId")
            tot_list.append(g.size().to_numpy())         # same group order, so
            inpk_list.append(g["_ip"].sum().to_numpy())  # in_peak and total stay aligned
        inpk = np.concatenate(inpk_list).astype(float) if inpk_list else np.array([])
        tot = np.concatenate(tot_list).astype(float) if tot_list else np.array([])
        per_bin.append((inpk, tot))
    return per_bin


def _boot_dist(inpk, tot, n_boot, rng):
    """Bootstrap distribution of the pooled fraction, resampling SHOTS with replacement."""
    n = len(tot)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        d = tot[idx].sum()
        out[b] = inpk[idx].sum() / d if d > 0 else np.nan
    return out


def get_yields(data, species_range, n_boot=N_BOOT, seed=SEED):
    """Central raw fractions plus shot-bootstrap errors.

    Returns ``(point, err_raw, err_low)`` where ``point`` is the raw pooled
    fraction per field bin, ``err_raw`` its 1-sigma bootstrap error, and
    ``err_low`` the error on the yield normalized to the lowest-field bin. The
    reference bin is resampled in the same procedure so its uncertainty
    propagates into every normalized point; the reference itself is unity by
    construction and carries no error bar.
    """
    per_bin = get_pershot_counts(data, species_range)
    rng = np.random.default_rng(seed)
    boots = [_boot_dist(ip, t, n_boot, rng) for (ip, t) in per_bin]
    point = np.array([ip.sum() / t.sum() if t.sum() > 0 else np.nan for (ip, t) in per_bin])
    err_raw = np.array([np.nanstd(b) for b in boots])
    ref = boots[0]                                   # lowest-field bin bootstrap
    err_low = np.array([0.0] + [np.nanstd(b / ref) for b in boots[1:]])
    return point, err_raw, err_low


def fit_species(data, species_range, decaying, n_boot=N_BOOT, seed=SEED):
    """Fit the saturating or decaying form to one species' field dependence.

    Returns ``(popt, pcov, point, err_raw, err_low)``.
    """
    point, err_raw, err_low = get_yields(data, species_range, n_boot, seed)
    model = dec_func if decaying else sat_func
    popt, pcov = curve_fit(
        model, data.field_centers, point,
        sigma=np.clip(err_raw, 1e-6, None), absolute_sigma=True,
        p0=[point.max() - point.min(), 1.5, point.min()],
        maxfev=10000,
    )
    return popt, pcov, point, err_raw, err_low


def significance_from_arrays(point, err_raw, field_centers):
    """Amplitude significance and likelihood-ratio test from precomputed arrays.

    This is the offline-verifiable path: it takes the raw pooled fractions and
    their bootstrap errors directly (e.g. the committed ``fig3b_raw_yield_*`` /
    ``fig3b_err_raw_*`` arrays) and does no resampling, so it needs none of the
    embargoed per-event data. :func:`significance` is the same fit fed from a
    :class:`FieldBinnedData`.

    Returns ``(amplitude, sigma_amplitude, n_sigma, delta_chi2, p_value)`` -- the
    28.9 / 10.2 / 8.7 sigma figures quoted in the SI.
    """
    point = np.asarray(point, float)
    err_raw = np.asarray(err_raw, float)
    field_centers = np.asarray(field_centers, float)
    fmodel = dec_func if point[-1] < point[0] else sat_func
    popt, pcov = curve_fit(
        fmodel, field_centers, point,
        sigma=np.clip(err_raw, 1e-9, None), absolute_sigma=True,
        p0=[abs(point.max() - point.min()), 1.5, point.min()],
        bounds=([-2, 0.05, -1], [2, 1e3, 2]), maxfev=10000,
    )
    A, sA = popt[0], np.sqrt(pcov[0, 0])
    chi2_fit = np.sum(((point - fmodel(field_centers, *popt)) / err_raw) ** 2)
    cmean = np.average(point, weights=1 / err_raw ** 2)
    dchi2 = np.sum(((point - cmean) / err_raw) ** 2) - chi2_fit
    return A, sA, abs(A / sA), dchi2, _chi2.sf(dchi2, 2)


def significance(data, species_range, n_boot=N_BOOT, seed=SEED):
    """Amplitude significance and the likelihood-ratio test against a constant.

    Bootstraps the yields from ``data`` (needs the per-event tables), then fits.
    For the offline check on the committed arrays, use
    :func:`significance_from_arrays`.

    Returns ``(amplitude, sigma_amplitude, n_sigma, delta_chi2, p_value)``.
    These are the 28.9 / 10.2 / 8.7 sigma figures quoted in the SI.
    """
    point, err_raw, _ = get_yields(data, species_range, n_boot, seed)
    return significance_from_arrays(point, err_raw, data.field_centers)


def verify_errors(data, species_range, n_boot=N_BOOT, seed=SEED):
    """Validate the error model on the data itself, with no outside assumptions.

    Runs the four checks reported in the SI, plus convergence and significance:

    [1] Shuffling in-peak labels across ions removes all shot structure, so the
        bootstrap must reproduce the analytic binomial error (ratio ~ 1).
    [2] Without shuffling, the bootstrap exceeds the binomial by 1.4-2.4,
        quantifying the overdispersion from shot clustering and type pooling.
    [3] A delete-one-shot jackknife must agree with the bootstrap.
    [4] For normalized yields the bootstrap must match first-order ratio
        propagation.
    [5] The error is converged in the number of resamples.
    [6] The weighted fit amplitude differs from zero, and a sloped fit beats a
        constant.
    """
    per_bin = get_pershot_counts(data, species_range)
    rng = np.random.default_rng(seed)
    line = "=" * 72
    print(line + "\nERROR VALIDATION on your data\n" + line)

    print("\n[1] Calibration: shuffle in-peak labels across ions to remove all shot")
    print("    structure (the independence limit). The bootstrap MUST then reproduce")
    print("    the analytic binomial sqrt(p(1-p)/N). PASS if ratio ~ 1.")
    for i, (ip, t) in enumerate(per_bin):
        N = int(t.sum()); K = int(ip.sum()); p = K / N
        labels = np.zeros(N); labels[:K] = 1.0; rng.shuffle(labels)
        sizes = t.astype(int)
        ip_null = np.array([c.sum() for c in np.split(labels, np.cumsum(sizes)[:-1])])
        se_null = np.nanstd(_boot_dist(ip_null, t, n_boot, rng))
        se_binom = np.sqrt(max(p * (1 - p), 1e-12) / N)
        print(f"    bin {i}: boot(shuffled)={se_null:.5f}  binomial={se_binom:.5f}"
              f"  ratio={se_null / se_binom:.2f}")

    print("\n[2] Real-data overdispersion: same comparison WITHOUT shuffling.")
    print("    ratio ~1 -> binomial bars would be fine; ratio >1 -> they are too small.")
    for i, (ip, t) in enumerate(per_bin):
        N = int(t.sum()); p = ip.sum() / N
        se_boot = np.nanstd(_boot_dist(ip, t, n_boot, rng))
        se_binom = np.sqrt(max(p * (1 - p), 1e-12) / N)
        print(f"    bin {i}: bootstrap={se_boot:.5f}  binomial={se_binom:.5f}"
              f"  ratio={se_boot / se_binom:.2f}  (N_shots={len(t)})")

    print("\n[3] Independent method: delete-one-shot JACKKNIFE must agree with bootstrap.")
    for i, (ip, t) in enumerate(per_bin):
        n = len(t); ip_all = ip.sum(); tot_all = t.sum()
        jk = np.array([(ip_all - ip[k]) / (tot_all - t[k]) for k in range(n)])
        se_jack = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))
        se_boot = np.nanstd(_boot_dist(ip, t, n_boot, rng))
        print(f"    bin {i}: jackknife={se_jack:.5f}  bootstrap={se_boot:.5f}"
              f"  ratio={se_jack / se_boot:.2f}")

    print("\n[4] Normalized error: bootstrap ratio vs analytic ratio propagation.")
    print("    PASS if ratio ~1 (bootstrap matches R*sqrt((si/pi)^2+(s0/p0)^2)).")
    point, err_raw, err_low = get_yields(data, species_range, n_boot, seed)
    p0, s0 = point[0], err_raw[0]
    for i in range(1, len(point)):
        R = point[i] / p0
        analytic = R * np.sqrt((err_raw[i] / point[i]) ** 2 + (s0 / p0) ** 2)
        print(f"    bin {i}: bootstrap={err_low[i]:.5f}  analytic={analytic:.5f}"
              f"  ratio={err_low[i] / analytic:.2f}")

    print("\n[5] Convergence in n_boot (largest bin).")
    i = int(np.argmax([len(t) for (_, t) in per_bin]))
    ip, t = per_bin[i]
    for nb in (500, 2000, 8000, 20000):
        print(f"    n_boot={nb:6d}: SE={np.nanstd(_boot_dist(ip, t, nb, np.random.default_rng(seed))):.6f}")

    print("\n[6] Field-dependence significance from the weighted fit (all bins):")
    A, sA, nsig, dchi2, pval = significance(data, species_range, n_boot, seed)
    print(f"    amplitude {A:.4f} +/- {sA:.4f} -> {nsig:.1f} sigma; "
          f"flat-vs-sloped dchi2={dchi2:.1f}, p={pval:.4f}")
    print(line)


def hist_for_selection(df, pulse_ids, bins, tof_range):
    """Flight-time histogram of the events belonging to ``pulse_ids``."""
    df2 = df[df["pulseId"].isin(pulse_ids)]
    df2 = df2[df2["tof"].between(tof_range[0], tof_range[1])]
    h, _ = np.histogram(df2["tof"].values, bins=bins)
    return h
