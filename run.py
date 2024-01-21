from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import cross_sections as cs, forces
from acsmuthi.postprocessing import rendering
from acsmuthi.postprocessing import fields

import numpy as np

# parameters of surrounded medium (air)
rho_fluid, c_fluid = 1.225, 331
# parameters of acoustic field (plane wave)
p0, freq = 1, 82
direction = np.array([0.70711, 0, -0.70711])
k = 2 * np.pi * freq / c_fluid
# parameters of particles
r_particle, rho_particle, c_particle = 1., 997, 1403

# order of multipole expansion
order = 3

# creating acoustic field
incident_field = PlaneWave(k=k, amplitude=p0, direction=direction)

# creating surrounded medium
medium = Medium(density=rho_fluid, pressure_velocity=c_fluid, hard_substrate=True)

# creating 3 spherical particles
sphere1 = SphericalParticle(
    position=np.array([-2., 0, 5.5]),
    radius=r_particle,
    density=rho_particle,
    pressure_velocity=c_particle,
    order=order
)
sphere2 = SphericalParticle(
    position=np.array([3., 0, 2.5]),
    radius=r_particle,
    density=rho_particle,
    pressure_velocity=c_particle,
    order=order
)
sphere3 = SphericalParticle(
    position=np.array([-0.5, 0, 1.5]),
    radius=r_particle,
    density=rho_particle,
    pressure_velocity=c_particle,
    order=order
)
particles = np.array([sphere1, sphere2, sphere3])

# creating simulation object
simulation = Simulation(particles=particles, medium=medium, initial_field=incident_field, frequency=freq, order=order)
# by default - solver is LU, but it is possible to use GMRES:
# simulation = Simulation(..., solver='GMRES')

# simulation.run() method returns preparation and solving matrix system time
print("Time:", simulation.run())

# computing extinction cross-section
ecs = cs.extinction_cs(simulation, by_multipoles=False)

# computing forces
frcs = forces.all_forces(simulation)

print("Extinction cross-section:", ecs, "Forces:", *frcs, sep='\n')

# easy way to draw total field (also it's possible to show only 'scattered' or 'incident' field) - may take time
rendering.show_pressure_field(
    simulation=simulation,
    x_min=-6, x_max=6, y_min=0, y_max=0, z_min=-3, z_max=9, num=201,
    field_type='total',
    cmap='RdBu_r',
    particle_color='gold',
    particle_linewidth=1.5
)


# but also it is possible to compute field and draw it manually:
# xx, zz = np.meshgrid(np.linspace(-6, 6, 201), np.linspace(-1, 11, 201))
# yy = np.full_like(xx, 0)
# total_field = fields.compute_total_field(xx, yy, zz, simulation)
# import matplotlib.pyplot as plt
# plt.imshow(total_field)
