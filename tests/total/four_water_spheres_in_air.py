from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, cross_sections as cs
import numpy as np


def test_four_water_spheres_in_air():
    # parameters of medium
    rho_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]

    # parameters of incident field
    direction = np.array([0.57735, 0.57735, -0.57735])
    freq = 80  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    k_l = 2 * np.pi * freq / c_fluid  # [1/m]

    # parameters of the spheres
    r_sph = 1  # [m]
    ro_sph = 997  # [kg/m^3]
    c_sph_l = 1403  # [m/s]

    pos2 = np.array([1.7 * r_sph, 1.6 * r_sph, 0.1 * r_sph])  # [m]
    pos3 = np.array([-1.5 * r_sph, 2 * r_sph, -1.4 * r_sph])  # [m]
    pos4 = np.array([-1.6 * r_sph, -1.3 * r_sph, 2.1 * r_sph])  # [m]
    pos1 = np.array([1.2 * r_sph, -1.5 * r_sph, -1.7 * r_sph])  # [m]

    order = 9

    incident_field = PlaneWave(k_l=k_l,
                               amplitude=p0,
                               direction=direction)

    fluid = Medium(density=rho_fluid, speed_l=c_fluid)

    sphere1 = Particle(position=pos1,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order)

    sphere2 = Particle(position=pos2,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order)

    sphere3 = Particle(position=pos3,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order)

    sphere4 = Particle(position=pos4,
                       radius=r_sph,
                       density=ro_sph,
                       speed_l=c_sph_l,
                       order=order)

    particles = np.array([sphere1, sphere2, sphere3, sphere4])

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

    comsol_scs = 8.0687	 # 8.0799
    comsol_frcs = np.array([[4.2871E-6,	6.8757E-6, -3.4738E-6],	[6.3812E-6,	5.9692E-6, -4.8634E-6],
                            [4.9670E-6,	3.1631E-6, -2.7768E-6],	[5.8094E-6,	6.0099E-6, -6.1842E-6]])
    np.testing.assert_allclose(frcs, comsol_frcs, rtol=5e-2)
    assert np.round(scs, 1) == np.round(comsol_scs, 1)