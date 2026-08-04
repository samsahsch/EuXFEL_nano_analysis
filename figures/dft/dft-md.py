import ase
from ase import Atoms
from ase.io import write, Trajectory,read
from ase import units
import copy
import gpaw.mpi as mpi
from gpaw.utilities.ps2ae import PS2AE
import numpy as np
from gpaw.utilities.dos import raw_orbital_LDOS

from math import pi, cos, sin
from gpaw.analyse.hirshfeld import HirshfeldPartitioning
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary,ZeroRotation
from gpaw import GPAW, setup_paths, FermiDirac,PoissonSolver
from ase.md.verlet import VelocityVerlet
from gpaw import Mixer, MixerSum, MixerDif
from gpaw import restart as gpaw_restart 

# START INITAL Structure and DFT Calculator parameters(GPAW) #

# Water Molecule for testing
a = 10.0  # use a large cell for vaccum

d = 0.9575
t = pi / 180 * 104.51
#atoms = Atoms('OH2',
#              [(0, 0, 0),
#               (d, 0, 0),
#               (d * cos(t), d * sin(t), 0)],
#              cell=(a, a, a))
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
txt0='LOG_MD'
xyz_filename='start.xyz'
# Read structure
atoms = read(xyz_filename)
#atoms.pbc = (True, True, False)
#poisson = PoissonSolver('fd', dipolelayer='xy')
### ALL DFT options are specfied by the calculator ###
calc = GPAW(
    mode='fd',
    h=h,
    xc=xc,
    occupations=FermiDirac(smearing),
    txt=txt0,
    charge=-1,
    maxiter=500
)
#calc=gpaw_restart(gpw0, txt=f"{base}_Ez_{Ez:+.3f}.txt")


atoms.calc = calc # Attach calculator to structure
# END INITAL Structure and DFT Calculator parameters(GPAW) #


### START MD and analysis  Parameters########
timestep=0.5 #fs
dump_interval=5 # Dump Analysis interval
save_interval=5 #How often to dump restart file (can be BIG file) 
maxsteps=500 #steps to run

temperature_K=300 #Temp in Kelvin
lscale=True # Do tempertaure Scaling ?
scale_interval=5 # How often
scalemode=0 #  0--> Re-scale velocites, else--> Re-Randomize
scale_max=None # What step to stop scaling. if NONE max steps
lstart=True
restart_step=30
if(lstart) :
    gpw0='RESTART' + str(restart_step).zfill(7) + '.gpw'
    atoms, calc = gpaw_restart(gpw0, txt=txt0+'.RESTART.'+str(restart_step).zfill(7))
    calc.set(mixer=Mixer(0.02, 5, 100))
ldpwave=False #Dump wavefunctions ?
bands_dump=[690,691,692,693,694,695] #Bands to dump
ldpwave_ae=False # Dump pseudo or all electron wavefunctions. For most visulization cases only pseudo wavefuntion needed

ldpbandprojection=True #Dump PAW projections of wfs onto atoms 
ldpbandprojection_spd=False # Decompose  PAW projections by spd ?


ldpcharge=False #Dump Electron Density?
ldpcharge_ae=False # All electron or psuedo?

### END MD and analysis  Parameters########

### MD Driver Class ###

class MD_instance_GPAW() :
    '''
    MD Class to drive dynamics.
    
    I wrote this in a very "FORTRAN" way.  ¯\_(ツ)_/¯· And then wrapped in a class to make more python like 
    
    Only Gamma point (Designed for Supercells,liquids,and isolated molecules)
    Simmialry only Non spin polarized calcualtions have been tested for analysis.

    Spin polarized and multiple k-points can be added  trivially. 
    If you want do to non-collinear spins, possible but probably hard

    Tested with openMPI (GPAW can have MPI problems sometimes depending on vendor. Have fun ;) )
    
    Dynamics Supported :
         NVE MD with option for Tempertaure Scaling
         
         TODO add other thermostats from ASE (should be simple).
              add element velocity wise scaling 
    Analysis Supported :
        Standard stuff (Eigen Values,xyzs)
        Partial Charges (Hirshfield Charges) 
        Dump wavefunctions (Psuedo or All Electron)
        Dump charge density (Psuedo or All Electron) 
        Atomic contributions to Wavefunction through PAW projections (PDOS and Partial charges can be constructed from this with Eigen values and Occupations)
        
        TODO 
        --PDOS 
        --Local Potential
    
    '''
    def __init__(self,atoms,maxsteps,timestep,temperature_K,lscale=False, scale_interval=1, scalemode=0, scale_max=None, dump_interval=None,save_interval=None,ldpwave=False,bands_dump=None, ldpwave_ae=False, ldpbandprojection=False, ldpbandprojection_spd=False,ldpcharge=False,ldpcharge_ae=False) :
        '''
        Intialize all parameters
        TODO Read from input file
        '''
        self.atoms=atoms
        self.natoms=len(self.atoms.get_chemical_symbols())
        self.timestep=timestep #fs
        self.dump_interval=dump_interval # Dump Analysis interval
        self.save_interval=save_interval #How often to dump restart file (can be BIG file) 
        self.maxsteps=maxsteps #steps to run

        self.temperature_K=temperature_K #Temp in Kelvin
        self.lscale=lscale # Do tempertaure Scaling ?
        self.scale_interval=scale_interval # How often
        self.scalemode=scalemode #  0--> Re-scale velocites, else--> Re-Randomize
        self.scale_max=scale_max # What step to stop scaling. if NONE max steps

        self.ldpwave=ldpwave #Dump wavefunctions ?
        self.bands_dump=bands_dump #Bands to dump
        self.ldpwave_ae=ldpwave_ae # Dump pseudo or all electron wavefunctions. For most visulization cases only pseudo wavefuntion needed

        self.ldpbandprojection=ldpbandprojection #Dump PAW projections of wfs onto atoms 
        self.ldpbandprojection_spd=ldpbandprojection_spd # Decompose  PAW projections by spd ?

        self.ldpcharge=ldpcharge #Dump Electron Density?
        self.ldpcharge_ae=ldpcharge_ae # All electron or psuedo?



    def run_MD_NVE(self) :
        '''
        I wrote this in a very "FORTRAN" way.  ¯\_(ツ)_/¯· 

        NVE MD with option for Tempertaure Scaling

        Only Gamma point and non spin polarized calcualtions have been tested for analysis.

        Spin polarized and multiple k-points can be added somewhat trivially. If you want do to non-collinear spins have fun.

        Tested with openMPI (GPAW can have MPI problems sometimes depending on vendor. Have fun ;) )

        Current Analysis supported :
        Standard stuff (Eigen Values,xyzs)
        Partial Charges (Hirshfield Charges) 
        Dump wavefunctions (Psuedo or All Electron)
        Dump charge density (Psuedo or All Electron) 
        Atomic contribution to Wavefunction through PAW projections (Partial charges can also be constructed from this)

        '''
        step=0
        if(lstart) :
            step=restart_step
        if(self.dump_interval is None) :
            self.dump_interval=self.maxsteps
        if(self.save_interval is None) :
            self.dump_interval=self.maxsteps
        if(self.scale_max is None) :
            self.scale_max=self.maxsteps
        # STEP 0
        self.atoms.get_potential_energy()
        self.dump_data(step)
        self.atoms.calc.write('RESTART'+str(step).zfill(7)+'.gpw', mode='all')

        # Integrator for the equations of motion, timestep depends on system
        self.dyn = VelocityVerlet(atoms, timestep * units.fs)

        self.intial_thermalize()
        traj_file="OUTPUT_TRAJECTORY_ASE.traj"

        self.dyn.attach(self._on_step, interval=1)

        # Now run everything in one go
        mpi.world.barrier()
        self.dyn.run(self.maxsteps)

    def _on_step(self):
        """
        This is called every MD step by ASE.
        We use dyn.nsteps to know where we are and decide
        when to scale T, dump analysis, and write restarts.
        """
        step = self.dyn.nsteps  # completed MD steps so far (1-based after first step)

        # Temperature scaling
        if self.lscale:
            if (step % self.scale_interval == 0) and (step < self.scale_max):
                if self.scalemode == 0:
                    self.scale_temperature()
                else:
                    self.intial_thermalize()

        # Analysis / dump
        if step % self.dump_interval == 0:
            self.dump_data(step)

        # Restart writing
        if step % self.save_interval == 0:
            self.atoms.calc.write('RESTART' + str(step).zfill(7) + '.gpw', mode='all')

    def intial_thermalize(self) :
        #Intalize Velocity 
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=self.temperature_K)
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
    def scale_temperature(self) :
        #Re-Scale temperature (Possible TODO add RXMD species wise scaling)
        target_temp=self.temperature_K*units.kB
        #ndof = self.atoms.get_number_of_degrees_of_freedom()
        current_temp = self.atoms.get_kinetic_energy() / self.natoms / 1.5
        scale = target_temp / current_temp
        self.atoms.set_momenta(self.atoms.get_momenta() * np.sqrt(scale))
        Stationary(self.atoms)
        ZeroRotation(self.atoms)
    def dump_data(self,step) :
        # Dump partial charges in xyz file
        hf = HirshfeldPartitioning(self.atoms.calc)
        for atom, charge in zip(self.atoms, hf.get_charges()):
            atom.charge = charge
        write(str(step).zfill(7)+'.xyz',self.atoms)
        
        # Dump Eigenvalues in text file file

        eigen_values=self.atoms.calc.get_eigenvalues()
        occupations=self.atoms.calc.get_occupation_numbers()
        fermi_level=self.atoms.calc.get_fermi_level()
        

        Mout=np.column_stack((eigen_values-fermi_level,occupations))
        np.savetxt('EIG.'+str(step).zfill(7)+'.txt',Mout)
        
        # Dump wavefunctions in cube file

        if (self.ldpwave) :
            t = PS2AE(self.atoms.calc, grid_spacing=0.05)
            for ib in self.bands_dump :
                if(self.ldpwave_ae) :
                    wf=t.get_wave_function(ib,ae=True)
                else :
                    wf = self.atoms.calc.get_pseudo_wave_function(band=ib)
                fname = 'wf.band.'+str(ib).zfill(4)+'.step.'+str(step).zfill(7)+'.cube'
               # mpi.world.barrier()
                write(fname, self.atoms, data=wf *wf* units.Bohr**1.5)
        
        # Dump projections in text file

        if(self.ldpbandprojection) :
            if(self.ldpbandprojection_spd) :
                Atom_wfn_Overlap=np.zeros((self.natoms,4,self.atoms.calc.get_number_of_bands()))
                chems=['s','p','d','f']
                for iatom in range(self.atoms) :
                    for spd in range(4) :
                        energies, weights=raw_orbital_LDOS(self.atoms.calc, a=iatom, spin=0, angular=chems[spd], nbands=None)
                        Atom_wfn_Overlap[iatom,spd,:]=weights 
                for iband in range(self.atoms.calc.get_number_of_bands()) :
                    norm_band=np.sum(Atom_wfn_Overlap[:,:,iband])
                    Atom_wfn_Overlap[:,:,iband]=Atom_wfn_Overlap[:,:,iband]/norm_band
            else:
                Atom_wfn_Overlap=np.zeros((self.natoms,self.atoms.calc.get_number_of_bands()))
                for iatom in range(self.natoms) :
                    energies, weights=raw_orbital_LDOS(self.atoms.calc, a=iatom, spin=0, angular='spdf', nbands=None)
                    Atom_wfn_Overlap[iatom,:]=weights 
                for iband in range(self.atoms.calc.get_number_of_bands()) :
                    norm_band=np.sum(Atom_wfn_Overlap[:,iband])
                    Atom_wfn_Overlap[:,iband]=Atom_wfn_Overlap[:,iband]/norm_band
            np.savetxt('AtomWvnOverlap.'+str(step).zfill(7)+'.txt',Atom_wfn_Overlap)
        
        # Dump chgd in cube file

        if(self.ldpcharge) :
            if(self.ldpcharge_ae) :
                density_dump=self.atoms.calc.get_all_electron_density(gridrefinement=4)
            else :
                density_dump=self.atoms.calc.get_pseudo_density()
            write('chgd.'+str(step).zfill(7)+'.cube', self.atoms, data=density_dump * units.Bohr**3)

    
if __name__ == "__main__" :
    MD=MD_instance_GPAW(atoms=atoms,maxsteps=maxsteps,
                        timestep=timestep,temperature_K=temperature_K,
                        lscale=lscale, 
                        scale_interval=scale_interval, 
                        scalemode=scalemode, 
                        scale_max=scale_max,
                        dump_interval=dump_interval,
                        save_interval=save_interval,
                        ldpwave=ldpwave,bands_dump=bands_dump, ldpwave_ae=ldpwave_ae, 
                        ldpbandprojection=ldpbandprojection, 
                        ldpbandprojection_spd=ldpbandprojection_spd,
                        ldpcharge=ldpcharge,
                        ldpcharge_ae=ldpcharge_ae)
    MD.run_MD_NVE()
