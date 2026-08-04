#!/usr/bin/env python

"""
Run GPAW ground-state DFT for an xyz geometry under
a series of electric fields along z, with no PBC along z.

Strategy:
  1) Do a single SCF calc at Ez = 0 (no external field), save .gpw
  2) For each Ez != 0, restart from the zero-field .gpw and add ConstantElectricField
"""

import sys
import numpy as np

from ase.io import read
from gpaw import GPAW, PoissonSolver, FermiDirac
from gpaw.external import ConstantElectricField

# ----------------- user-adjustable parameters ----------------- #

# Electric field strengths along +z in V/Å:
field_strengths = np.linspace(0.0, 1.0, 5)  # includes 0.0


# Real-space grid spacing (Å)
h = 0.20

# Exchange-correlation functional
xc = 'PBE'

# Smearing (eV)
smearing = 0.01

# -------------------------------------------------------------- #

#if len(sys.argv) != 2:
#    print("Usage: python dft_Efield_z_restart.py geometry.xyz")
#    sys.exit(1)

xyz_filename='combo.0001.xyz'
# Read structure
atoms = read(xyz_filename)

# Center in a box with vacuum in all directions
#atoms.center(vacuum=vacuum)

# Periodic only in x,y; non-periodic in z
atoms.pbc = (True, True, False)
# For a fully isolated cluster you could also do:
# atoms.pbc = (False, False, False)

poisson = PoissonSolver('fd', dipolelayer='xy')

print(f"Running fields (V/Å): {field_strengths}")

# ---- Step 1: zero-field SCF ---- #
Ez0 = 0.0
base='Sio2'
txt0 = f"{base}_Ez_{Ez0:+.3f}.txt"
gpw0 = f"{base}_Ez_{Ez0:+.3f}.gpw"

print(f"\n=== Zero-field run: Ez = {Ez0:+.3f} V/Å ===")

calc0 = GPAW(
    mode='fd',
    h=h,
    xc=xc,
    occupations=FermiDirac(smearing),
    poissonsolver=poisson,
    txt=txt0,
)

atoms.calc = calc0
E0 = atoms.get_potential_energy()
dip0 = calc0.get_dipole_moment()

print(f"Total energy (Ez=0): {E0: .6f} eV")
print(f"Dipole moment (Ez=0): {dip0}")

calc0.write(gpw0, mode='all')

# ---- Step 2: loop over finite fields, restart from Ez=0 ---- #
from gpaw import restart as gpaw_restart  # convenience helper

for Ez in field_strengths:
    if abs(Ez) < 1e-10:
        # already did this one
        continue

    print(f"\n=== Field Ez = {Ez:+.3f} V/Å ===")

    # Restart from zero-field density + wfs
    atoms_field, calc_field = gpaw_restart(gpw0, txt=f"{base}_Ez_{Ez:+.3f}.txt")

    # Attach the constant electric field along z
    ext = ConstantElectricField(strength=Ez, direction=[0.0, 0.0, 1.0])
    calc_field.set(external=ext)

    # Make sure Atoms object has the updated calc
    atoms_field.calc = calc_field

    E = atoms_field.get_potential_energy()
    dip = calc_field.get_dipole_moment()

    print(f"Total energy: {E: .6f} eV")
    print(f"Dipole moment (ex, ey, ez): {dip}")

    calc_field.write(f"{base}_Ez_{Ez:+.3f}.gpw", mode='all')

print("\nDone. Zero-field file is *_Ez_+0.000.gpw; finite fields restart from that.")

