import numpy as np
from acsmuthi.initial_field import PlaneWave
from acsmuthi.medium import Medium
from acsmuthi import particles
from acsmuthi.simulation import Simulation
from acsmuthi.postprocessing import cross_sections as cs, fields, forces
from tests.one_sphere_simulation import OneSphericalParticleSimulation


def test_one_sphere():
    rho_fluid, c_fluid = 1.225, 331
    p0, freq = 1, 82
    k_l = 2 * np.pi * freq / c_fluid
    direction, position = np.array([0, 0, 1]), np.array([0, 0, 0])
    r_sph, rho_sph, c_sph = 1., 1050, 1403

    xx, zz = np.meshgrid(np.linspace(-6, 6, 201), np.linspace(-6, 6, 201))
    yy = np.full_like(xx, 0.)

    order = 6

    incident_field = PlaneWave(k_l, p0, direction)
    fluid = Medium(rho_fluid, c_fluid)
    spheres = np.array([particles.SphericalParticle(position, r_sph, rho_sph, c_sph, order)])
    sim = Simulation(spheres, fluid, incident_field, freq, order)
    sim.run()
    scs, ecs = cs.cross_section(sim)
    frc = forces.all_forces(sim)[0][2]

    actual_field = fields.compute_inner_field(xx, yy, zz, sim) + fields.compute_scattered_field(xx, yy, zz, sim)

    sim_one_sphere = OneSphericalParticleSimulation(freq, rho_sph, c_sph, r_sph, rho_fluid, c_fluid, p0, order)
    sim_one_sphere.evaluate_solution()
    scs1, ecs1 = sim_one_sphere.scattering_cs(), sim_one_sphere.extinction_cs()
    frc1 = sim_one_sphere.forces()
    desired_field = sim_one_sphere.compute_pressure_field(xx, yy, zz)

    np.testing.assert_allclose(actual_field, desired_field, rtol=1e-10)
    np.testing.assert_allclose(ecs / (np.pi * r_sph ** 2), ecs1)
    np.testing.assert_allclose(scs, scs1)
    np.testing.assert_allclose(frc, frc1)
