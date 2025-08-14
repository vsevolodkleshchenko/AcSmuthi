import numpy as np
import matplotlib.pyplot as plt

from acsmuthi.simulation import Simulation
from acsmuthi.particles import SphericalParticle
from acsmuthi.medium import FluidMedium, ElasticMedium, MediumSystem
from acsmuthi.initial_field import PlaneWave
from acsmuthi.postprocessing import cross_sections as cs, forces
from acsmuthi.postprocessing import rendering
from acsmuthi.postprocessing import fields
from acsmuthi.linear_system.coupling.coupling_basics import k_contour


# parameters of surrounded medium (air)
rho_fluid, c_fluid = 1.225, 331
# parameters of acoustic field (plane wave)
p0, freq = 1, 82
direction = np.array([0.70711, 0, -0.70711])  # todo: check the datatype
k = 2 * np.pi * freq / c_fluid
# parameters of particles
r_particle, rho_particle, c_particle = 1., 997, 1403

# order of multipole expansion
order = 3

# creating acoustic field
incident_field = PlaneWave(frequency=freq, amplitude=p0, direction=direction)

# creating surrounded medium
medium = MediumSystem(
    [
        FluidMedium(
            density=rho_fluid,
            sound_speed_longitudinal=c_fluid
        ),
        ElasticMedium(
            density=2650,
            sound_speed_longitudinal=5900,
            sound_speed_transversal=3400
        )
    ]
)

# creating 3 spherical particles
sphere1 = SphericalParticle(
    position=np.array([-2., 0, 5.5]),
    radius=r_particle,
    density=rho_particle,
    sound_speed_longitudinal=c_particle,
    multipole_order=order
)
sphere2 = SphericalParticle(
    position=np.array([3., 0, 2.5]),
    radius=r_particle,
    density=rho_particle,
    sound_speed_longitudinal=c_particle,
    multipole_order=order
)
sphere3 = SphericalParticle(
    position=np.array([-0.5, 0, 1.5]),
    radius=r_particle,
    density=rho_particle,
    sound_speed_longitudinal=c_particle,
    multipole_order=order
)
particles = [sphere1, sphere2, sphere3]

kpar = k_contour(
    imag_deflection=3e-2,
    step=1e-2,
    problems=np.array([c_fluid / 5900, c_fluid / 3400])
)

# creating simulation object
simulation = Simulation(
    particles=particles,
    medium=medium,
    initial_field=incident_field,
    order=order,
    k_parallel=kpar*k
)
# by default - solver is LU, but it is possible to use GMRES:
# simulation = Simulation(..., solver='GMRES')

# simulation.run() method returns preparation and solving matrix system time
print("Time:", simulation.run())

# computing extinction cross-section
ecs = cs.extinction_cs(simulation, by_multipoles=False)

# computing forces
frcs = forces.all_forces(simulation)

print("Extinction cross-section:", ecs, "Forces:", *frcs, sep='\n')

# easy way to draw total field (also it's possible to show only 'scattered'
# or 'incident' field) - may take time
rendering.show_pressure_field(
    simulation=simulation,
    x_min=-6, x_max=6, y_min=0, y_max=0, z_min=-3, z_max=9, num=201,
    field_type='total',
    cmap='RdBu_r',
    particle_color='gold',
    particle_linewidth=1.5
)
plt.tight_layout(), plt.show()


# but also it is possible to compute field and draw it manually:
xx, zz = np.meshgrid(np.linspace(-6, 6, 201), np.linspace(-1, 11, 201))
yy = np.full_like(xx, 0)
total_field = fields.compute_total_field(xx, yy, zz, simulation)
plt.imshow(total_field)
