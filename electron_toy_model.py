import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.constants import e, epsilon_0, pi, m_e
from tqdm import tqdm

# Simulation parameters
nanoparticle_radius = 250e-9  # 500 nm diameter
detector_distance = 300e-3    # 300 mm
simulation_time = 500e-9      # 500 ns
dt = 100e-12                  # 1 ps time step
num_intensity_levels = 10
base_electron_count = 500     # Base number of electrons (scales with intensity)

# Constants (in computational units)
k_coulomb = 1.0  # Normalized Coulomb constant

def simulate_electron_dynamics():
    """Simulate electron dynamics for multiple intensity levels."""
    # Define intensity levels
    intensity_levels = np.linspace(.1, 1, num_intensity_levels)
    
    # Arrays to store results
    all_times = []
    all_intensities = []
    
    # Loop through intensity levels
    for intensity in tqdm(intensity_levels, desc="Simulating intensities"):
        # Number of electrons scales with intensity
        n_electrons = int(base_electron_count * intensity)
        n_electrons = max(n_electrons, 50)  # Ensure minimum number of electrons
        
        # Initial energy proportional to X-ray intensity
        base_energy_spread = 10.0 * intensity # eV
        energy_spread = base_energy_spread * (1 + intensity_factor * intensity)
        initial_energy = np.random.normal(10.0 * intensity, energy_spread, n_electrons)

        initial_speed = np.sqrt(2 * initial_energy)
        
        # Initialize electrons on nanoparticle surface
        theta = np.random.uniform(0, 2*pi, n_electrons)
        phi = np.random.uniform(0, pi, n_electrons)
        
        positions = np.zeros((n_electrons, 3))
        positions[:, 0] = nanoparticle_radius * np.sin(phi) * np.cos(theta)
        positions[:, 1] = nanoparticle_radius * np.sin(phi) * np.sin(theta)
        positions[:, 2] = nanoparticle_radius * np.cos(phi)
        
        # Initial velocities (radially outward with speeds proportional to sqrt(intensity))
        velocities = np.zeros_like(positions)
        for i in range(n_electrons):
            direction = positions[i] / np.linalg.norm(positions[i])
            # Speed proportional to sqrt(intensity) with some random variation
            # This represents the initial kinetic energy from X-ray absorption
            speed = 100.0 * np.sqrt(intensity) * (1 + 0.1 * np.random.randn())
            velocities[i] = speed * direction
        
        # Hole charge left behind (proportional to number of electrons that have left)
        hole_charge = n_electrons
        
        # Arrays to track flight times and detected status
        flight_times = np.ones(n_electrons) * simulation_time
        detected = np.zeros(n_electrons, dtype=bool)
        
        # Current state
        current_positions = positions.copy()
        current_velocities = velocities.copy()
        
        # Time evolution
        for t in np.arange(0, simulation_time, dt):
            # Calculate forces
            forces = np.zeros_like(current_positions)
            
            # Electron-electron repulsion
            for i in range(n_electrons):
                if detected[i]:
                    continue
                
                for j in range(i+1, n_electrons):
                    if detected[j]:
                        continue
                    
                    r_vec = current_positions[j] - current_positions[i]
                    r_mag = np.linalg.norm(r_vec)
                    
                    if r_mag > 0.1:  # Avoid singularity
                        # Implement screening in the electron-electron repulsion:
                        screening_length = 1e-6  # Characteristic screening length that varies with density
                        force_mag = k_coulomb * np.exp(-r_mag/screening_length) / r_mag**2
                        force_dir = r_vec / r_mag
                        
                        # Apply Coulomb repulsion
                        forces[i] += -force_dir * force_mag
                        forces[j] += force_dir * force_mag
            
            # Electron-hole attraction (from the positive charge left behind)
            for i in range(n_electrons):
                if detected[i]:
                    continue
                
                r_vec = current_positions[i]
                r_mag = np.linalg.norm(r_vec)
                
                if r_mag > nanoparticle_radius:  # Only apply if electron has left surface
                    # KEY EFFECT: Attraction scales with hole charge AND intensity
                    # This creates stronger attraction at higher intensities
                    force_mag = k_coulomb * hole_charge * intensity / r_mag**2
                    force_dir = -r_vec / r_mag
                    forces[i] += force_dir * force_mag
            
            # Update velocities and positions using simple Euler integration
            current_velocities += forces * dt
            current_positions += current_velocities * dt
            
            # Check for electrons reaching detection distance
            for i in range(n_electrons):
                if not detected[i]:
                    distance = np.linalg.norm(current_positions[i])
                    if distance >= detector_distance:
                        flight_times[i] = t
                        detected[i] = True
            
            # Stop if all electrons are detected
            if np.all(detected):
                break
        
        # Store results
        all_times.extend(flight_times)
        all_intensities.extend([intensity] * n_electrons)
    
    return np.array(all_intensities), np.array(all_times)

def plot_results(intensities, times):
    """Create a 2D histogram similar to the experimental data."""
    plt.figure(figsize=(10, 8))
    
    # Create 2D histogram
    hist, x_edges, y_edges = np.histogram2d(
        times * 1e9,  # Convert to ns
        intensities,
        bins=[200, 50],
        range=[[0, simulation_time * 1e9], [min(intensities), max(intensities)]]
    )
    
    # Plot as an image with logarithmic color scale
    plt.imshow(
        hist.T,
        aspect='auto',
        origin='lower',
        extent=[0, simulation_time * 1e9, min(intensities), max(intensities)],
        interpolation='gaussian',
        norm=LogNorm(),
        cmap='viridis'
    )
    
    plt.colorbar(label='Electron Count')
    plt.ylabel('X-ray Intensity (a.u.)')
    plt.xlabel('Time of Flight (ns)')
    plt.title('Simulated Electron Time of Flight vs X-ray Intensity')
    
    # # Add annotations explaining the physics
    # plt.annotate('Higher initial energy\n→ faster electrons', xy=(50, 0.2), xytext=(100, 0.15),
    #             arrowprops=dict(arrowstyle='->'))
    # plt.annotate('Space charge effect\n→ slower electrons', xy=(300, 0.9), xytext=(350, 0.85),
    #             arrowprops=dict(arrowstyle='->'))
    
    plt.tight_layout()
    plt.show()

# Run the simulation
print("Starting electron time-of-flight simulation...")
intensities, times = simulate_electron_dynamics()
plot_results(intensities, times)
print("Simulation complete.")
