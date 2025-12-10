
import numpy as np
import matplotlib.pyplot as plt

# Physical constants
e = 1.602176634e-19  # elementary charge in Coulombs
epsilon_0 = 8.8541878128e-12  # vacuum permittivity in F/m
k_e = 1/(4*np.pi*epsilon_0)  # Coulomb constant (8.99e9 N⋅m²/C²)

class NanoparticleElectrostatics:
    """
    Class to handle electrostatic calculations for charged nanoparticles
    in X-ray photoelectron emission experiments
    """

    def __init__(self, diameter_nm, electrons_removed):
        self.diameter = diameter_nm * 1e-9  # convert to meters
        self.radius = self.diameter / 2
        self.electrons_removed = electrons_removed
        self.charge = electrons_removed * e  # total positive charge

    def surface_field_single(self):
        """Calculate electric field at surface for isolated sphere"""
        return self.charge / (4 * np.pi * epsilon_0 * self.radius**2)

    def surface_charge_density(self):
        """Calculate surface charge density"""
        surface_area = 4 * np.pi * self.radius**2
        return self.electrons_removed / surface_area

    def potential_at_surface(self):
        """Calculate electric potential at surface"""
        return self.charge / (4 * np.pi * epsilon_0 * self.radius)

def calculate_two_sphere_interaction(particle1, particle2, separation_distance):
    """
    Calculate electric field distribution for two charged spheres

    Args:
        particle1, particle2: NanoparticleElectrostatics objects
        separation_distance: center-to-center distance in meters

    Returns:
        Dictionary with field calculations at various surface points
    """

    Q1, Q2 = particle1.charge, particle2.charge
    r1, r2 = particle1.radius, particle2.radius
    d = separation_distance

    # Self-field for each particle (unchanged by presence of other)
    E1_self = Q1 / (4 * np.pi * epsilon_0 * r1**2)
    E2_self = Q2 / (4 * np.pi * epsilon_0 * r2**2)

    # Calculate field at different points on particle surfaces
    results = {
        'particle1': {},
        'particle2': {},
        'separation_nm': separation_distance * 1e9
    }

    # For particle 1 surface points
    # Point closest to particle 2
    d1_min = d - r1  # distance from particle 1 surface to particle 2 center
    E2_at_p1_closest = Q2 / (4 * np.pi * epsilon_0 * d1_min**2) if d1_min > 0 else np.inf

    # Point farthest from particle 2  
    d1_max = d + r1
    E2_at_p1_farthest = Q2 / (4 * np.pi * epsilon_0 * d1_max**2)

    # Side points (perpendicular to line connecting centers)
    d1_side = np.sqrt(d**2 + r1**2)
    E2_at_p1_side = Q2 / (4 * np.pi * epsilon_0 * d1_side**2)

    # Total fields at particle 1 surface
    results['particle1'] = {
        'self_field': E1_self,
        'closest_point_total': E1_self + E2_at_p1_closest,
        'farthest_point_total': E1_self + E2_at_p1_farthest,
        'side_point_total': E1_self + E2_at_p1_side,
        'field_from_p2_closest': E2_at_p1_closest,
        'field_from_p2_farthest': E2_at_p1_farthest,
        'field_from_p2_side': E2_at_p1_side
    }

    # For particle 2 surface points (similar calculations)
    d2_min = d - r2
    E1_at_p2_closest = Q1 / (4 * np.pi * epsilon_0 * d2_min**2) if d2_min > 0 else np.inf

    d2_max = d + r2
    E1_at_p2_farthest = Q1 / (4 * np.pi * epsilon_0 * d2_max**2)

    d2_side = np.sqrt(d**2 + r2**2)
    E1_at_p2_side = Q1 / (4 * np.pi * epsilon_0 * d2_side**2)

    results['particle2'] = {
        'self_field': E2_self,
        'closest_point_total': E2_self + E1_at_p2_closest,
        'farthest_point_total': E2_self + E1_at_p2_farthest,
        'side_point_total': E2_self + E1_at_p2_side,
        'field_from_p1_closest': E1_at_p2_closest,
        'field_from_p1_farthest': E1_at_p2_farthest,
        'field_from_p1_side': E1_at_p2_side
    }

    return results

def field_analysis_vs_separation(particle1, particle2, separations_nm):
    """Analyze field variation with separation distance"""

    analysis_results = []

    for sep_nm in separations_nm:
        sep_m = sep_nm * 1e-9

        # Skip if particles would overlap
        min_separation = (particle1.radius + particle2.radius) * 1e9
        if sep_nm <= min_separation:
            continue

        result = calculate_two_sphere_interaction(particle1, particle2, sep_m)

        # Convert fields to V/nm for easier interpretation
        p1_data = {
            'separation_nm': sep_nm,
            'self_field_V_per_nm': result['particle1']['self_field'] * 1e-9,
            'closest_total_V_per_nm': result['particle1']['closest_point_total'] * 1e-9,
            'farthest_total_V_per_nm': result['particle1']['farthest_point_total'] * 1e-9,
            'side_total_V_per_nm': result['particle1']['side_point_total'] * 1e-9
        }

        analysis_results.append(p1_data)

    return analysis_results

# Main execution
if __name__ == "__main__":

    print("=== X-RAY PHOTOELECTRON EMISSION: TWO-PARTICLE ANALYSIS ===\n")

    # Define the experimental setup
    diameter_nm = 300
    electrons_per_particle = 40000  # half of single particle case

    # Create two identical particles
    particle1 = NanoparticleElectrostatics(diameter_nm, electrons_per_particle)
    particle2 = NanoparticleElectrostatics(diameter_nm, electrons_per_particle)

    print(f"Particle diameter: {diameter_nm} nm")
    print(f"Particle radius: {particle1.radius*1e9:.1f} nm")
    print(f"Electrons removed per particle: {electrons_per_particle:,}")
    print(f"Charge per particle: {particle1.charge*1e12:.3f} pC")
    print(f"Self-field per particle: {particle1.surface_field_single()*1e-9:.3f} V/nm")
    print()

    # Analyze different separation distances
    separations = np.array([350, 400, 500, 600, 800, 1000, 1500, 2000])  # nm

    print("ELECTRIC FIELD AT PARTICLE SURFACE vs SEPARATION DISTANCE")
    print("=" * 80)
    print("Sep(nm) | Self(V/nm) | Closest(V/nm) | Farthest(V/nm) | Side(V/nm) | Enhancement")
    print("-" * 80)

    analysis = field_analysis_vs_separation(particle1, particle2, separations)

    for data in analysis:
        enhancement = data['closest_total_V_per_nm'] / data['self_field_V_per_nm']
        print(f"{data['separation_nm']:7.0f} | {data['self_field_V_per_nm']:10.3f} | "
              f"{data['closest_total_V_per_nm']:13.3f} | {data['farthest_total_V_per_nm']:14.3f} | "
              f"{data['side_total_V_per_nm']:10.3f} | {enhancement:11.2f}x")

    print()
    print("KEY INSIGHTS:")
    print("- Self-field: Electric field due to particle's own charge only")
    print("- Closest: Field at surface point nearest to other particle") 
    print("- Farthest: Field at surface point farthest from other particle")
    print("- Side: Field at surface points perpendicular to inter-particle axis")
    print("- Enhancement: Factor by which closest-point field exceeds self-field")
    print()

    # Detailed calculation for a specific case
    separation_example = 400  # nm
    sep_m = separation_example * 1e-9

    print(f"DETAILED ANALYSIS FOR {separation_example} nm SEPARATION:")
    print("=" * 60)

    detailed_result = calculate_two_sphere_interaction(particle1, particle2, sep_m)

    p1 = detailed_result['particle1']
    print(f"Particle 1 Surface Fields:")
    print(f"  Self-field: {p1['self_field']*1e-9:.3f} V/nm")
    print(f"  Field from P2 at closest point: {p1['field_from_p2_closest']*1e-9:.3f} V/nm")
    print(f"  Field from P2 at farthest point: {p1['field_from_p2_farthest']*1e-9:.3f} V/nm")
    print(f"  Field from P2 at side points: {p1['field_from_p2_side']*1e-9:.3f} V/nm")
    print()
    print(f"  Total field at closest point: {p1['closest_point_total']*1e-9:.3f} V/nm")
    print(f"  Total field at farthest point: {p1['farthest_point_total']*1e-9:.3f} V/nm")  
    print(f"  Total field at side points: {p1['side_point_total']*1e-9:.3f} V/nm")

    print()
    print("COMPARISON WITH SINGLE PARTICLE CASE:")
    print("=" * 50)

    # Single particle with 80,000 electrons
    single_particle = NanoparticleElectrostatics(diameter_nm, 80000)
    single_field = single_particle.surface_field_single() * 1e-9

    print(f"Single particle (80k electrons): {single_field:.3f} V/nm")
    print(f"Two particles - self field only: {p1['self_field']*1e-9:.3f} V/nm")
    print(f"Two particles - closest point: {p1['closest_point_total']*1e-9:.3f} V/nm")
    print()
    print(f"Ratio (single/two-self): {single_field/(p1['self_field']*1e-9):.2f}")
    print(f"Ratio (two-closest/single): {(p1['closest_point_total']*1e-9)/single_field:.2f}")
