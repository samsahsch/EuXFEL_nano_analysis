import numpy as np
from scipy.constants import e, epsilon_0, pi
from dataclasses import dataclass

# -----------------------------
# Configuration dataclass
# -----------------------------
@dataclass
class NPConfig:
    d: float            # nanoparticle diameter (m)
    Nslices: int        # grid size per dimension (preferably odd)
    dr: float           # electron mean free path to surface (m)
    e_per_photon: float # electrons per absorbed photon (if used)
    total_electrons: float  # total electrons emitted (normalization)
    # For dimers/two-monomers, total_electrons is the total for the configuration.
    # Use factors (e.g., 1.75x) on monomer counts outside.

# -----------------------------
# Grid and geometry utilities
# -----------------------------
def build_sphere_indices(Nslices):
    radius = (Nslices - 1) // 2
    # Return list of (x,y,z) inside sphere, and a ring on the equatorial plane y=y0
    pts = []
    ring = []
    x0 = y0 = z0 = Nslices // 2
    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            for z in range(z0 - radius, z0 + radius + 1):
                rsq = (x0 - x)**2 + (y0 - y)**2 + (z0 - z)**2
                if rsq <= radius**2:
                    pts.append((x, y, z))
                    # Surface ring: near surface AND central equator y==y0
                    if y == y0:
                        # near-surface band: within one voxel of surface
                        if radius - np.sqrt(rsq) < 1.0:
                            ring.append((x, y, z))
    return pts, ring, radius, (x0, y0, z0)

def voxel_centers(pts, cellSize, d, center):
    # Convert voxel indices to physical coordinates with zero at sphere center
    x0, y0, z0 = center
    coords = []
    for (x, y, z) in pts:
        # Shift indices so center voxel ~0; scale by cell size; center at (-d/2, d/2] range
        # Use center at 0 by: (idx - center_index)*cellSize
        X = (x - x0) * cellSize
        Y = (y - y0) * cellSize
        Z = (z - z0) * cellSize
        coords.append((X, Y, Z))
    return np.array(coords)

# -----------------------------
# Electron contribution model
# -----------------------------
def electron_contribution_array(Nslices, d, dr, NphotonsAbs_or_total_e, e_per_photon=1.0, mode='direct'):
    """
    Return a 3D array of electrons contributing per cell to the surface signal,
    proportional to exp(-depth/dr). Two modes:
    - 'direct': NphotonsAbs_or_total_e is total electrons to distribute uniformly in volume, weighted by exp(-depth/dr)
    - 'photon': NphotonsAbs_or_total_e is absorbed photons per second; multiply by e_per_photon first
    """
    radius = (Nslices - 1) // 2
    cellSize = d / Nslices

    # Decide total electrons
    if mode == 'photon':
        total_electrons = NphotonsAbs_or_total_e * e_per_photon
    else:
        total_electrons = NphotonsAbs_or_total_e

    sphere = np.zeros((Nslices, Nslices, Nslices), dtype=np.float64)
    x0 = y0 = z0 = Nslices // 2

    # Compute unnormalized weights first: w(x) = exp(-depth/dr) for interior voxels
    weights = []
    indices = []
    for x in range(x0 - radius, x0 + radius + 1):
        for y in range(y0 - radius, y0 + radius + 1):
            for z in range(z0 - radius, z0 + radius + 1):
                rsq = (x0 - x)**2 + (y0 - y)**2 + (z0 - z)**2
                if rsq <= radius**2:
                    # depth to surface in meters
                    depth_vox = (radius - np.sqrt(rsq)) * cellSize
                    w = np.exp(-max(depth_vox, 0.0)/dr)
                    weights.append(w)
                    indices.append((x, y, z))
    weights = np.array(weights, dtype=np.float64)
    Wsum = weights.sum()
    if Wsum <= 0:
        raise ValueError("Weight sum is zero; check Nslices, d, dr.")
    # Normalize to total_electrons
    scale = total_electrons / Wsum
    for (idx, w) in zip(indices, weights):
        x, y, z = idx
        sphere[x, y, z] = w * scale

    return sphere  # units: electrons per cell (already normalized to total_electrons)

# -----------------------------
# Field evaluation
# -----------------------------
def field_on_surface_ring(sphere_e, d, Nslices, sample_ring, center):
    """
    Compute local electrostatic field (magnitude, V/m) at each ring sample point,
    from all electrons in 'sphere_e'. We use Coulomb's law; each cell is a point charge q = e * Ne_cell.
    For consistency with your previous scalar accumulation, we sum magnitudes from Coulomb contributions
    along the line-of-centers (i.e., contributions treated as scalar E = k*q/r^2), then report magnitudes.
    """
    cellSize = d / Nslices
    x0, y0, z0 = center

    # Pre-build source list (coords in meters, q in Coulombs)
    src_idx = np.argwhere(sphere_e > 0)
    if src_idx.size == 0:
        raise ValueError("No source electrons in sphere.")
    q_elec = []
    src_pos = []
    for (xi, yi, zi) in src_idx:
        q = sphere_e[xi, yi, zi] * e  # Coulombs
        X = (xi - x0) * cellSize
        Y = (yi - y0) * cellSize
        Z = (zi - z0) * cellSize
        q_elec.append(q)
        src_pos.append((X, Y, Z))
    q_elec = np.array(q_elec, dtype=np.float64)
    src_pos = np.array(src_pos, dtype=np.float64)

    # Evaluate at each ring point
    Evals = []
    k = 1.0 / (4.0 * pi * epsilon_0)

    for (i, j, kidx) in sample_ring:
        Xp = (i - x0) * cellSize
        Yp = (j - y0) * cellSize
        Zp = (kidx - z0) * cellSize
        rp = np.array([Xp, Yp, Zp])
        R = src_pos - rp  # vectors from field point to sources
        r2 = np.einsum('ij,ij->i', R, R)
        # Avoid singularity: add one cell in radius as you did
        r = np.sqrt(r2) + cellSize
        # Scalar Coulomb magnitude contribution per source:
        # E_mag_i = k * q_i / r_i^2
        Emagi = (k * q_elec) / (r * r)
        # Sum magnitudes (to align with your energy scalar mapping approach)
        Etotal = Emagi.sum()
        Evals.append(Etotal)

    return np.array(Evals, dtype=np.float64)

# -----------------------------
# Multi-sphere (dimer, two monomers) assembly
# -----------------------------
def translate_sphere_e(sphere_e, shift_vox, Nslices):
    """
    Translate a sphere electron distribution by integer voxel shifts (dx, dy, dz).
    Areas outside are dropped (zero-padded). Useful to place two spheres (dimer/two-monomers).
    """
    from numpy import roll
    dx, dy, dz = shift_vox
    # Use roll then zero out wrapped regions:
    shifted = np.roll(sphere_e, shift=dx, axis=0)
    shifted = np.roll(shifted, shift=dy, axis=1)
    shifted = np.roll(shifted, shift=dz, axis=2)
    # Zero wrap regions explicitly
    x0 = 0 if dx >= 0 else Nslices + dx
    x1 = dx if dx >= 0 else 0
    if dx != 0:
        shifted[x0:x1 if x1 != 0 else Nslices, :, :] = 0.0
    y0 = 0 if dy >= 0 else Nslices + dy
    y1 = dy if dy >= 0 else 0
    if dy != 0:
        shifted[:, y0:y1 if y1 != 0 else Nslices, :] = 0.0
    z0 = 0 if dz >= 0 else Nslices + dz
    z1 = dz if dz >= 0 else 0
    if dz != 0:
        shifted[:, :, z0:z1 if z1 != 0 else Nslices] = 0.0
    return shifted

def build_dimer_distribution(mono_e, Nslices, gap_voxels=0):
    """
    Build 'touching' dimer by placing two identical spheres along x-axis.
    Center separation ~ 2R (in voxels) + gap_voxels.
    """
    radius = (Nslices - 1) // 2
    center_sep = 2 * radius + gap_voxels
    dx = center_sep // 2
    # Shift left/right around center index:
    left = translate_sphere_e(mono_e, (-dx, 0, 0), Nslices)
    right = translate_sphere_e(mono_e, (dx, 0, 0), Nslices)
    return left + right

def build_two_monomers_distribution(mono_e, Nslices, big_sep_voxels):
    """
    Place two monomers far apart (no overlap, negligible cross-coupling in your picture).
    Superposition still applies for fields, but arrangement is similar to dimer with larger separation.
    """
    dx = big_sep_voxels // 2
    left = translate_sphere_e(mono_e, (-dx, 0, 0), Nslices)
    right = translate_sphere_e(mono_e, (dx, 0, 0), Nslices)
    return left + right

# -----------------------------
# Main driver to compute fields
# -----------------------------
def compute_local_fields(config_mono: NPConfig,
                         dimer_factor=1.75,
                         two_monomers_factor=2.0,
                         dimer_gap_vox=0,
                         two_mono_sep_vox=120,
                         mode='direct'):
    """
    Returns a dictionary with field arrays (V/m) on the equatorial surface ring:
    - 'monomer': E(theta_i)
    - 'dimer': E(theta_i) for touching agglomerate
    - 'two_monomers': E(theta_i) for well-separated
    Also returns ring angles (theta from voxel coordinates) in radians for plotting.
    """
    d = config_mono.d
    Nslices = config_mono.Nslices
    dr = config_mono.dr
    e_per_phot = config_mono.e_per_photon

    cellSize = d / Nslices

    # Build sphere indices and ring sampling
    pts, ring, radius, center = build_sphere_indices(Nslices)

    # Compute monomer electrons array normalized to total electrons
    mono_e = electron_contribution_array(
        Nslices=Nslices,
        d=d,
        dr=dr,
        NphotonsAbs_or_total_e=config_mono.total_electrons,
        e_per_photon=e_per_phot,
        mode=mode
    )

    # Dimer and two-monomers electron arrays with scaled totals
    # Scale monomer distribution proportionally:
    mono_e_unit = mono_e / mono_e.sum()  # shape normalized
    dimer_e = build_dimer_distribution(mono_e_unit * (config_mono.total_electrons * dimer_factor), Nslices, gap_voxels=dimer_gap_vox)
    two_e = build_two_monomers_distribution(mono_e_unit * (config_mono.total_electrons * two_monomers_factor), Nslices, big_sep_voxels=two_mono_sep_vox)

    # Surface ring physical coordinates & angles (r, theta) for reporting
    x0, y0, z0 = center
    thetas = []
    for (i, j, kidx) in ring:
        # equatorial plane -> r,theta in x-z
        xi = (i - x0)
        zi = (kidx - z0)
        theta = np.arctan2(zi, xi)
        thetas.append(theta)
    thetas = np.array(thetas)

    # Compute E on ring for each configuration
    E_mono = field_on_surface_ring(mono_e, d, Nslices, ring, center)
    E_dimer = field_on_surface_ring(dimer_e, d, Nslices, ring, center)
    E_two = field_on_surface_ring(two_e, d, Nslices, ring, center)

    out = {
        'theta_rad': thetas,          # angle on equatorial ring
        'E_monomer_Vpm': E_mono,      # local field magnitudes (V/m)
        'E_dimer_Vpm': E_dimer,
        'E_two_monomers_Vpm': E_two,
        'stats': {
            'monomer': {'mean': float(np.mean(E_mono)), 'min': float(np.min(E_mono)), 'max': float(np.max(E_mono))},
            'dimer': {'mean': float(np.mean(E_dimer)), 'min': float(np.min(E_dimer)), 'max': float(np.max(E_dimer))},
            'two_monomers': {'mean': float(np.mean(E_two)), 'min': float(np.min(E_two)), 'max': float(np.max(E_two))},
        }
    }
    return out

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Example: match your 60 nm case (dc_estimate) for local field magnitude
    cfg = NPConfig(
        d=500e-9,            # 60 nm diameter
        Nslices=101,        # fine grid (odd)
        dr=3e-9,            # 3 nm mean free path
        e_per_photon=2.0,   # if using photon mode
        total_electrons=5.4381689462e10  # example from your dc_estimate printout
    )

    # Compute fields (set factors for dimer/two-monomers as desired)
    # dimer_factor within 1.6–1.9 for touching agglomerate as you indicated
    results = compute_local_fields(
        config_mono=cfg,
        dimer_factor=1.75,
        two_monomers_factor=2.0,
        dimer_gap_vox=0,         # touching
        two_mono_sep_vox=200,    # large separation in voxels
        mode='photon'            # we pass total_electrons directly
    )

    # Print quick stats
    print("Angles (rad) sampled on equator:", results['theta_rad'].shape[0])
    for k, v in results['stats'].items():
        print(f"{k}  mean={v['mean']:.3e} V/m   min={v['min']:.3e}   max={v['max']:.3e}")

    # If needed, sort by theta for plotting
    import numpy as np
    order = np.argsort(results['theta_rad'])
    theta_sorted = results['theta_rad'][order]
    Em = results['E_monomer_Vpm'][order]
    Ed = results['E_dimer_Vpm'][order]
    Et = results['E_two_monomers_Vpm'][order]

    # Quick plot (optional)
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,5))
        plt.plot(theta_sorted, Em, '-', label='Monomer')
        plt.plot(theta_sorted, Ed, '-', label='Dimer (touching)')
        plt.plot(theta_sorted, Et, '-', label='Two monomers (separated)')
        plt.xlabel('Surface angle θ (rad) on equator')
        plt.ylabel('Local field |E| (V/m)')
        plt.title('Local surface field on equatorial ring')
        plt.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass
