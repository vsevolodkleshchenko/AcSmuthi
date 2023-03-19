from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_three_aerogel_spheres_in_air():
    # parameters of medium
    rho_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([1, 0, 0])
    freq = 7323.4  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    poisson = 0.12  # [1]
    young = 197920  # [Pa]
    g = 0.5 * young / (1 + poisson)
    ro_sph = 80  # [kg/m^3]
    c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
    c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

    r_sph1 = 0.01  # [m]
    r_sph2 = 0.013  # [m]

    pos1 = np.array([0, 0, -2.5 * r_sph2])  # [m]
    pos2 = np.array([0, 0, 2.5 * r_sph2])  # [m]
    pos3 = np.array([0, 0, 0])  # [m]

    order = 8

    incident_field = PlaneWave(k_l=k_l,
                               amplitude=p0,
                               direction=direction)

    fluid = Medium(density=rho_fluid, speed_l=c_fluid)

    sphere1 = Particle(position=pos1,
                       radius=r_sph1,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere2 = Particle(position=pos2,
                       radius=r_sph1,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    sphere3 = Particle(position=pos3,
                       radius=r_sph2,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order,
                       speed_t=c_sph_t)

    particles = np.array([sphere1, sphere2, sphere3])
    linear_system = LinearSystem(particles=particles,
                                 medium=fluid,
                                 initial_field=incident_field,
                                 frequency=freq,
                                 order=order,
                                 store_t_matrix=True)
    linear_system.prepare()
    linear_system.solve()

    scs = cs.extinction_cs(particles=particles,
                           medium=fluid,
                           incident_field=incident_field,
                           freq=freq)

    frcs = forces.all_forces(particles_array=particles,
                             medium=fluid,
                             incident_field=incident_field)

    comsol_scs = 9.5773E-4  # 9.6687E-4
    comsol_frcs = np.array([[3.3725E-14, 3.1652E-14, -3.1027E-14],
                            [2.7348E-14, 3.5052E-14, -2.8459E-14],
                            [2.7348E-14, 3.5052E-14, -2.8459E-14]])

    # np.testing.assert_allclose(frcs, comsol_frcs, rtol=1e-1)
    assert scs == 9.5773E-4  # 9.6687E-4