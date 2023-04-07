from acsmuthi.simulation import Simulation
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_two_steel_spheres_in_water():
    # parameters of medium
    rho_fluid = 997  # [kg/m^3]
    c_fluid = 1403  # [m/s]

    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 60_000  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 0.01  # [m]
    ro_sph = 7700  # [kg/m^3]
    c_sph_l = 5740  # [m/s]
    c_sph_t = 3092  # [m/s]

    pos1 = np.array([-1.6 * r_sph, 0, 0])  # [m]
    pos2 = np.array([1.6 * r_sph, 0, 0])  # [m]

    order = 10

    incident_field = PlaneWave(k_l=k_l,
                               amplitude=p0,
                               direction=direction)

    fluid = Medium(density=rho_fluid, speed_l=c_fluid)

    sphere1 = Particle(position=pos1,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere2 = Particle(position=pos2,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    particles = np.array([sphere1, sphere2])

    sim = Simulation(
        particles=particles,
        medium=fluid,
        initial_field=incident_field,
        frequency=freq,
        order=order,
        store_t_matrix=True
    )
    sim.run()

    scs = cs.extinction_cs(particles=particles,
                           medium=fluid,
                           incident_field=incident_field,
                           freq=freq)

    frcs = forces.all_forces(particles_array=particles,
                             medium=fluid,
                             incident_field=incident_field)

    # comsol_scs = 6.868E-4  # 6.9386E-4
    # comsol_frcs = np.array([[4.5201E-14, 0, 5.1318E-14], [5.1488E-14, 0, 5.9230E-14]])
    comsol_scs = 6.9092E-4  # 6.8824E-4
    comsol_frcs = np.array([[4.5198E-14, 0, 5.1185E-14], [5.1514E-14, 0, 5.9213E-14]])

    np.testing.assert_allclose(np.where(np.abs(frcs) <= 1e-14, 0, frcs), comsol_frcs, rtol=5e-1)
    assert np.round(scs, 4) == np.round(comsol_scs, 4)