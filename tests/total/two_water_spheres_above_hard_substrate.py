from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, rendering, cross_sections as cs
import numpy as np

import matplotlib, matplotlib.pyplot as plt

plt.rcdefaults()
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['axes.formatter.min_exponent'] = 1


def test_two_water_spheres_above_hard_substrate():
    # parameters of medium
    rho_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.70711, 0., -0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 1  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph = 1403  # [m/s]

    pos1 = np.array([-1.5, 0, 1.7])  # [m]
    pos2 = np.array([1.7, 0., 1.6])  # [m]

    order = 8

    incident_field = PlaneWave(k=k_l, amplitude=p0, direction=direction)

    fluid = Medium(density=rho_fluid, pressure_velocity=c_fluid, is_substrate=True)

    sphere1 = SphericalParticle(position=pos1, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    sphere2 = SphericalParticle(position=pos2, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    particles = np.array([sphere1, sphere2])

    sim = Simulation(particles=particles, medium=fluid, initial_field=incident_field, frequency=freq, order=order)
    sim.run()

    scs = cs.extinction_cs(simulation=sim)
    frcs = forces.all_forces(sim)

    comsol_scs = 5.6212
    comsol_frcs = np.array([[3.3599e-6, 0., -1.0000e-5], [1.8190e-6, 0., -9.2240e-6]])

    np.testing.assert_allclose(np.where(np.abs(frcs) > 1e-10, frcs, 0), comsol_frcs, rtol=1e-2)
    assert np.round(scs, 1) == np.round(comsol_scs, 1)
