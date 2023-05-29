import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import numpy as np
from acsmuthi.postprocessing import fields


def draw_particles(simulation):
    fig = plt.gcf()
    ax = fig.gca()
    for particle in simulation.particles:
        circle = plt.Circle((particle.position[0], particle.position[2]), particle.radius, linewidth=3.5, fill=False, color='white')
        ax.add_patch(circle)


def show_pressure_field(simulation, x_min, x_max, y_min, y_max, z_min, z_max, num, field_type='total', cmap='RdBu_r'):
    if x_min == x_max:
        yy, zz = np.meshgrid(np.linspace(y_min, y_max, num), np.linspace(z_min, z_max, num))
        xx = np.full_like(yy, x_min)
        extent = [y_min, y_max, z_min, z_max]
        x_label, y_label, title = 'y', 'z', 'x = ' + str(x_min)
    elif y_min == y_max:
        xx, zz = np.meshgrid(np.linspace(x_min, x_max, num), np.linspace(z_min, z_max, num))
        yy = np.full_like(xx, y_min)
        extent = [x_min, x_max, z_min, z_max]
        x_label, y_label, title = 'x', 'z', 'y = ' + str(y_min)
    elif z_min == z_max:
        xx, yy = np.linspace(x_min, x_max, num), np.linspace(y_min, y_max, num)
        zz = np.full_like(xx, z_min)
        extent = [x_min, x_max, y_min, y_max]
        x_label, y_label, title = 'x', 'y', 'z = ' + str(z_min)

    if field_type == 'total':
        p_field = fields.compute_total_field(xx, yy, zz, simulation)
    elif field_type == 'scattered':
        p_field = fields.compute_scattered_field(xx, yy, zz, simulation) + \
                  fields.compute_inner_field(xx, yy, zz, simulation)
    elif field_type == 'incident':
        p_field = fields.compute_incident_field(xx, yy, zz, simulation)

    fig, ax = plt.subplots()
    im = ax.imshow(p_field, origin='lower', extent=extent, norm=colors.CenteredNorm(), cmap=cmap)
    ax.set_xlabel(x_label + ', м')
    ax.set_ylabel(y_label + ', м')
    ax.set_title('Pressure field at ' + title + ', Па')
    plt.colorbar(im)
    draw_particles(simulation)
    plt.show()
