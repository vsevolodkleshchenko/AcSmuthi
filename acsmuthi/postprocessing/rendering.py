import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from acsmuthi.postprocessing import fields
from acsmuthi.simulation import Simulation


def draw_particles(
    simulation: Simulation,
    x_label: str,
    y_label: str,
    ax,
    color: str = 'black',
    linewidth: float = 1
):
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
    simulation: Simulation,
    x_min: float, x_max: float,
    y_min: float, y_max: float,
    z_min: float, z_max: float,
    num: int,
    field_type: Literal[
        'total', 'scattered', 'incident', 'inner', 'scattered+inner', 'inner+scattered',
        'scattered+incident', 'incident+scattered', 'incident+inner', 'inner+incident'
    ] = 'total',
    cmap='RdBu_r',
    particle_color='black',
    particle_linewidth: float = 1,
    ax=None, figsize=None
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
        p_field = fields.compute_total_field(x=xx, y=yy, z=zz, simulation=simulation)
    elif field_type == 'scattered':
        p_field = fields.compute_scattered_field(x=xx, y=yy, z=zz, simulation=simulation)
    elif field_type == 'incident':
        p_field = fields.compute_incident_field(x=xx, y=yy, z=zz, simulation=simulation)
    elif field_type == 'inner':
        p_field = fields.compute_inner_field(x=xx, y=yy, z=zz, simulation=simulation)
    elif field_type == 'scattered+inner' or field_type == 'inner+scattered':
        scattered_field = fields.compute_scattered_field(x=xx, y=yy, z=zz, simulation=simulation)
        inner_field = fields.compute_inner_field(x=xx, y=yy, z=zz, simulation=simulation)
        p_field = scattered_field + inner_field
    elif field_type == 'scattered+incident' or field_type == 'incident+scattered':
        scattered_field = fields.compute_scattered_field(x=xx, y=yy, z=zz, simulation=simulation)
        incident_field = fields.compute_incident_field(x=xx, y=yy, z=zz, simulation=simulation)
        p_field = scattered_field + incident_field
    elif field_type == 'inner+incident' or field_type == 'incident+inner':
        inner_field = fields.compute_inner_field(x=xx, y=yy, z=zz, simulation=simulation)
        incident_field = fields.compute_incident_field(x=xx, y=yy, z=zz, simulation=simulation)
        p_field = inner_field + incident_field
    else:
        raise ValueError('Unknown field type: ' + field_type)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(p_field, origin='lower', extent=extent, norm=colors.CenteredNorm(), cmap=cmap)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title('Pressure field at ' + title)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    draw_particles(
        simulation=simulation,
        x_label=x_label,
        y_label=y_label,
        ax=ax,
        color=particle_color,
        linewidth=particle_linewidth
    )
    # plt.show()
