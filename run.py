from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import fields
from acsmuthi.postprocessing import rendering
from acsmuthi.postprocessing import cross_sections as cs, forces

import numpy as np


# parameters of medium
rho_fluid = 1.225  # [kg/m^3]
c_fluid = 331  # [m/s]

# parameters of incident field
direction = np.array([1, 0, 0])
freq = 31.64  # [Hz]
p0 = 1  # [kg/m/s^2] = [Pa]
k_l = 2 * np.pi * freq / c_fluid  # [1/m]

# parameters of the spheres
poisson = 0.12
young = 197920
g = 0.5 * young / (1 + poisson)
pos1 = np.array([0., 0, 2])  # [m]
pos2 = np.array([0., 0, -2])  # [m]
# [m]
r_sph = 1.  # [m]
ro_sph = 80  # [kg/m^3]
c_sph_l = np.sqrt(2 * g * (1 - poisson) / ro_sph / (1 - 2 * poisson))  # [m/s]
c_sph_t = np.sqrt(g / ro_sph)  # [m/s]

order = 3

incident_field = PlaneWave(k_l=k_l,
                           amplitude=p0,
                           direction=direction)

fluid = Medium(density=rho_fluid, speed_l=c_fluid)

sphere1 = SphericalParticle(position=pos1,
                   radius=r_sph,
                   density=ro_sph,
                   speed_l=c_sph_l,
                   order=order,
                   speed_t=c_sph_t)
sphere2 = SphericalParticle(position=pos2,
                   radius=r_sph,
                   density=ro_sph,
                   speed_l=c_sph_l,
                   order=order,
                   speed_t=c_sph_t)

particles = np.array([sphere1, sphere2])

bound, number_of_points = 16, 171
plane = 'xz'
plane_number = int(number_of_points / 2) + 1
span = rendering.build_discretized_span(
    bound=bound,
    number_of_points=number_of_points
)
x_p, y_p, z_p, span_v, span_h = rendering.build_slice(
    span=span,
    plane_number=plane_number,
    plane=plane
)

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

tot_field = np.real(fields.compute_total_field(
    particles=particles,
    incident_field=incident_field,
    x=x_p, y=y_p, z=z_p))
rendering.slice_plot(
    tot_field=tot_field,
    span_v=span_v,
    span_h=span_h,
    plane=plane
)

