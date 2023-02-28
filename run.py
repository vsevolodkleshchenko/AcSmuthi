from acsmuthi.simulation import Simulation
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
import numpy as np


# parameters of medium
rho_fluid = 1.225  # [kg/m^3]
c_fluid = 331  # [m/s]

# parameters of incident field
direction = np.array([0, 0, 1])
freq = 7323.4  # [Hz]
p0 = 1  # [kg/m/s^2] = [Pa]
k_l = 2 * np.pi * freq / c_fluid  # [1/m]

# parameters of the spheres
poisson = 0.12
young = 197920
g = 0.5 * young / (1 + poisson)
pos1 = np.array([-0.015, 0, -0.015/2])  # [m]
pos2 = np.array([0.015, 0, -0.015/2])  # [m]
pos3 = np.array([0, 0, 0.015])
# [m]
r_sph = 0.01  # [m]
ro_sph = 80  # [kg/m^3]
c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

order = 18

incident_field = PlaneWave(k_l, p0, direction)

fluid = Medium(rho_fluid, c_fluid)
sphere1 = Particle(pos1, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
sphere2 = Particle(pos2, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
sphere3 = Particle(pos3, r_sph, ro_sph, c_sph_l, order, speed_t=c_sph_t)
particles = np.array([sphere1, sphere2, sphere3])

bound, number_of_points = 0.045, 151
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

simulation = Simulation(particles, fluid, incident_field, freq, order, bound=bound, number_of_points=number_of_points,
                 plane=plane, plane_number=plane_number, store_t_matrix=True)

simulation.run(cross_sections_flag=True, forces_flag=True, plot_flag=False)
