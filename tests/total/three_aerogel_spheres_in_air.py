from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import StandingWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_three_aerogel_spheres_in_air():
    # parameters of medium
    rho_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([1., 0, 0])
    freq = 80  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    poisson = 0.12  # [1]
    young = 197920  # [Pa]
    g = 0.5 * young / (1 + poisson)
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    r_sph1 = 1  # [m]
    r_sph2 = 1.3 * r_sph1  # [m]

    pos1 = np.array([0, 0, 3.1 * r_sph1])  # [m]
    pos3 = np.array([0, 0, -3.1 * r_sph1])  # [m]
    pos2 = np.array([0., 0., 0.])  # [m]

    order = 10

    incident_field = StandingWave(k=k_l, amplitude=p0, direction=direction)

    fluid = Medium(density=rho_fluid, pressure_velocity=c_fluid)

    sphere1 = SphericalParticle(position=pos1, radius=r_sph1, density=ro_sph, pressure_velocity=c_sph_l, order=order,
                                shear_velocity=c_sph_t)

    sphere2 = SphericalParticle(position=pos2, radius=r_sph2, density=ro_sph, pressure_velocity=c_sph_l, order=order,
                                shear_velocity=c_sph_t)

    sphere3 = SphericalParticle(position=pos3, radius=r_sph1, density=ro_sph, pressure_velocity=c_sph_l, order=order,
                                shear_velocity=c_sph_t)

    particles = np.array([sphere1, sphere2, sphere3])
    sim = Simulation(
        particles=particles,
        medium=fluid,
        initial_field=incident_field,
        frequency=freq,
        order=order,
        store_t_matrix=True
    )
    sim.run()

    scs = cs.extinction_cs(simulation=sim)

    frcs = forces.all_forces(sim)

    comsol_scs = 24.408  # 24.412
    comsol_frcs = np.array([[0, 0, 1.7290E-5], [0, 0, 0], [0, 0, -1.7292E-5]])

    np.testing.assert_allclose(np.where(np.abs(frcs) <= 1e-14, 0, frcs), comsol_frcs, rtol=2e-2)
    assert np.round(scs, 1) == np.round(comsol_scs, 1)
