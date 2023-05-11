import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from acsmuthi.postprocessing import fields


def draw_particles(simulation):
    fig = plt.gcf()
    ax = fig.gca()
    for particle in simulation.particles:
        circle = plt.Circle((particle.position[0], particle.position[2]), particle.radius, linewidth=1.5, fill=False, color='white')
        ax.add_patch(circle)


def show_pressure_field(simulation, x_min, x_max, y_min, y_max, z_min, z_max, num, field_type='total', cmap='cividis'):
    if x_min == x_max:
        yy, zz = np.meshgrid(np.linspace(y_min, y_max, num), np.linspace(z_min, z_max, num))
        xx = np.full_like(yy, x_min)
        extent = [y_min, y_max, z_min, z_max]
    elif y_min == y_max:
        xx, zz = np.meshgrid(np.linspace(x_min, x_max, num), np.linspace(z_min, z_max, num))
        yy = np.full_like(xx, y_min)
        extent = [x_min, x_max, z_min, z_max]
    elif z_min == z_max:
        xx, yy = np.linspace(x_min, x_max, num), np.linspace(y_min, y_max, num)
        zz = np.full_like(xx, z_min)
        extent = [x_min, x_max, y_min, y_max]

    if field_type == 'total':
        p_field = fields.compute_total_field(xx, yy, zz, simulation.particles, simulation.initial_field)
    elif field_type == 'scattered':
        p_field = fields.compute_scattered_field(xx, yy, zz, simulation.particles) + \
                  fields.compute_inner_field(xx, yy, zz, simulation.particles)
    elif field_type == 'incident':
        p_field = fields.compute_incident_field(xx, yy, zz, simulation.particles, simulation.initial_field)

    fig, ax = plt.subplots()
    im = ax.imshow(p_field, origin='lower', extent=extent, cmap=sns.color_palette("cividis", as_cmap=True))
    plt.colorbar(im)
    draw_particles(simulation)
    plt.show()




