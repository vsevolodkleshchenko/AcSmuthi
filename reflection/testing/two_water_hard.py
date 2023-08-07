import matplotlib, matplotlib.pyplot as plt

from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import forces, rendering, cross_sections as cs
import numpy as np


plt.rcdefaults()
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['axes.formatter.min_exponent'] = 1


def two_water_spheres_above_hard_substrate(order):
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

    incident_field = PlaneWave(k=k_l, amplitude=p0, direction=direction)

    fluid = Medium(density=rho_fluid, pressure_velocity=c_fluid, is_substrate=True)

    sphere1 = SphericalParticle(position=pos1, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    sphere2 = SphericalParticle(position=pos2, radius=r_sph, density=ro_sph, pressure_velocity=c_sph, order=order)
    particles = np.array([sphere1, sphere2])

    sim = Simulation(particles=particles, medium=fluid, initial_field=incident_field, frequency=freq, order=order)
    sim.run()

    # scs = cs.extinction_cs(simulation=sim)
    frcs = forces.all_forces(sim)
    return frcs[0][0]


def convergence():
    orders = np.arange(2, 9)
    frc = []
    for order in orders:
        frc.append(two_water_spheres_above_hard_substrate(order))
    print(frc)


def plot_conv():
    fig, ax = plt.subplots()
    orders = np.arange(2, 9)
    frcx1 = [2.339044276003572e-06, 3.3367321796652886e-06, 3.365986130343006e-06, 3.367479027276258e-06,
             3.3675054930481067e-06, 3.3675074218550167e-06, 3.3675101883149036e-06]
    ax.set_xlabel('order')
    ax.set_ylabel('Fx')
    plt.plot(orders, frcx1, linewidth=3)
    plt.show()


# plot_conv()
