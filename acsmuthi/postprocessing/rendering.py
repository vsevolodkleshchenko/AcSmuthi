import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import numpy as np
from typing import Literal
from acsmuthi.postprocessing import fields


def draw_particles(simulation, x_label, y_label, color='black', linewidth=1):
    fig = plt.gcf()
    ax = fig.gca()
    if x_label == 'x' and y_label == 'z':
        x_index, y_index = 0, 2
    elif x_label == 'y' and y_label == 'z':
        x_index, y_index = 1, 2
    elif x_label == 'x' and y_label == 'y':
        x_index, y_index = 0, 1
    for particle in simulation.particles:
        circle = plt.Circle(
            (particle.position[x_index], particle.position[y_index]),
            particle.radius,
            linewidth=linewidth,
            fill=False,
            color=color
        )
        ax.add_patch(circle)


def show_pressure_field(
        simulation,
        x_min, x_max, y_min, y_max, z_min, z_max, num,
        field_type: Literal['total', 'scattered', 'incident', 'scattered', 'incident', 'scattered+inner', 'incident+scattered', 'incident+inner'] ='total',
        cmap='RdBu_r',
        particle_color='black',
        particle_linewidth=1
):
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
        p_field = fields.compute_scattered_field(xx, yy, zz, simulation)
    elif field_type == 'incident':
        p_field = fields.compute_incident_field(xx, yy, zz, simulation)
    elif field_type == 'scattered+inner' or field_type == 'inner+scattered':
        p_field = fields.compute_inner_field(xx, yy, zz, simulation) + \
                  fields.compute_scattered_field(xx, yy, zz, simulation)
    elif field_type == 'scattered+incident' or field_type == 'incident+scattered':
        p_field = fields.compute_incident_field(xx, yy, zz, simulation) + \
                  fields.compute_scattered_field(xx, yy, zz, simulation)
    elif field_type == 'inner+incident' or field_type == 'incident+inner':
        p_field = fields.compute_incident_field(xx, yy, zz, simulation) + \
                  fields.compute_inner_field(xx, yy, zz, simulation)

    fig, ax = plt.subplots()
    im = ax.imshow(p_field, origin='lower', extent=extent, norm=colors.CenteredNorm(), cmap=cmap)
    ax.set_xlabel(x_label + ', м')
    ax.set_ylabel(y_label + ', м')
    ax.set_title('Pressure field at ' + title + ', Па')
    plt.colorbar(im)
    draw_particles(simulation, x_label, y_label, color=particle_color, linewidth=particle_linewidth)
    plt.show()
