from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import cross_sections as cs, forces
from acsmuthi.postprocessing import rendering

import numpy as np

# parameters of surrounded medium
rho_fluid, c_fluid = 1.225, 331
# parameters of acoustic field (plane wave)
p0, freq = 1, 82
direction = np.array([0.70711, 0, 0.70711])
k = 2 * np.pi * freq / c_fluid
# parameters of particles
r_sph, rho_sph, c_sph = 1., 997, 1403

# order of multipole expansion
order = 8

# creating acoustic field
incident_field = PlaneWave(k=k, amplitude=p0, direction=direction)
# creating surrounded medium
medium = Medium(density=rho_fluid, pressure_velocity=c_fluid)
# creating 3 spherical particles
sphere1 = SphericalParticle(
    position=np.array([-2., 0, 3]),
    radius=r_sph,
    density=rho_sph,
    pressure_velocity=c_sph,
    order=order
)
sphere2 = SphericalParticle(
    position=np.array([3., 0, -1]),
    radius=r_sph,
    density=rho_sph,
    pressure_velocity=c_sph,
    order=order
)
sphere3 = SphericalParticle(
    position=np.array([0., 0, -1]),
    radius=r_sph,
    density=rho_sph,
    pressure_velocity=c_sph,
    order=order
)
particles = np.array([sphere1, sphere2, sphere3])
# creating simulation object
simulation = Simulation(
    particles=particles,
    medium=medium,
    initial_field=incident_field,
    frequency=freq,
    order=order
)

# simulation.run() method returns preparation and solving matrix system time
print("Time:", simulation.run())

# computing extinction cross-section
ecs = cs.extinction_cs(simulation, by_multipoles=False)
# computing forces
frcs = forces.all_forces(simulation)
print(ecs, *frcs, sep='\n')
# easy way to draw total field (also it's possible to show only 'scattered' or 'incident' field)
rendering.show_pressure_field(simulation, -6, 6, 0, 0, -6, 6, 151, 'total')
