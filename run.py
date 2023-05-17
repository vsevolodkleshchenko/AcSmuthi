from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import cross_sections as cs, forces
from acsmuthi.postprocessing import rendering

import numpy as np


rho_fluid, c_fluid = 1.225, 331
p0, freq = 1, 82
direction = np.array([0.70711, 0, 0.70711])
k = 2 * np.pi * freq / c_fluid
r_sph, rho_sph, c_sph = 1., 997, 1403

order = 8

incident_field = PlaneWave(k=k, amplitude=p0, direction=direction)
medium = Medium(density=rho_fluid, pressure_velocity=c_fluid)
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
simulation = Simulation(
    particles=particles,
    medium=medium,
    initial_field=incident_field,
    frequency=freq,
    order=order
)

print("Time:", simulation.run())

ecs = cs.extinction_cs(simulation, by_multipoles=False)
frcs = forces.all_forces(simulation)
print(ecs, *frcs, sep='\n')

rendering.show_pressure_field(simulation, -6, 6, 0, 0, -6, 6, 151, 'total')
