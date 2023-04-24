import matplotlib.pyplot as plt

from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import fields
from acsmuthi.postprocessing import cross_sections as cs, forces

import numpy as np


# parameters of medium
rho_fluid = 1.225  # [kg/m^3]
c_fluid = 331  # [m/s]

# parameters of incident field
direction = np.array([1, 0, 0])
freq = 18.11  # [Hz]
p0 = 1  # [kg/m/s^2] = [Pa]
k_l = 2 * np.pi * freq / c_fluid  # [1/m]

# parameters of the spheres
poisson = 0.12
young = 197920
g = 0.5 * young / (1 + poisson)
pos1 = np.array([-1., 0, 5])  # [m]
pos2 = np.array([1., 0, -5])  # [m]
# [m]
r_sph = 1.  # [m]
ro_sph = 80  # [kg/m^3]
c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

order = 3

incident_field = PlaneWave(
    k_l=k_l,
    amplitude=p0,
    direction=direction
)

fluid = Medium(density=rho_fluid, speed_l=c_fluid)

sphere1 = SphericalParticle(
    position=pos1,
    radius=r_sph,
    density=ro_sph,
    speed_l=c_sph_l,
    order=order,
    speed_t=c_sph_t
)
sphere2 = SphericalParticle(
    position=pos2,
    radius=r_sph,
    density=ro_sph,
    speed_l=c_sph_l,
    order=order,
    speed_t=c_sph_t
)

particles = np.array([sphere1, sphere2])

simulation = Simulation(
    particles=particles,
    medium=fluid,
    initial_field=incident_field,
    frequency=freq,
    order=order,
    store_t_matrix=True
)

print("Time:", simulation.run())

ecs = cs.extinction_cs(particles, fluid, incident_field, freq)
frcs = forces.all_forces(particles, fluid, incident_field)
print(ecs, *frcs, sep='\n')

xx, zz = np.meshgrid(np.linspace(-16, 16, 251), np.linspace(-16, 16, 251))
yy = np.full_like(xx, 0.)

p_field = np.real(
    fields.compute_scattered_field(
        x=xx, y=yy, z=zz,
        particles=particles
    )
)

plt.imshow(p_field, origin='lower')
plt.show()
