import numpy as np

from acsmuthi.utility import mathematics as mths


def compute_scattered_field(particles_array, x, y, z):
    r"""Counts scattered field on mesh x, y, z"""
    scattered_field_array = np.zeros((len(particles_array), len(x)), dtype=complex)
    for s, particle in enumerate(particles_array):
        particle.scattered_field.compute_pressure_field(x, y, z)
        scattered_field_array[s] = particle.scattered_field.field
    scattered_field = mths.spheres_fsum(scattered_field_array, len(x))
    return scattered_field


def compute_inner_field(particles_array, x, y, z):
    r"""Counts inner field in every sphere on mesh x, y, z"""
    inner_fields_array = np.zeros((len(particles_array), len(x)), dtype=complex)
    for s, particle in enumerate(particles_array):
        particle.inner_field.compute_pressure_field(x, y, z)
        rx, ry, rz = x - particle.pos[0], y - particle.pos[1], z - particle.pos[2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        particle.inner_field.field = np.where(r <= particle.r, particle.inner_field.field, 0)
        inner_fields_array[s] = particle.inner_field.field
    return mths.spheres_fsum(inner_fields_array, len(x))


def compute_incident_field(incident_field, x, y, z):
    inc_field = incident_field.compute_exact_field(x, y, z)
    return inc_field


def compute_total_field(particles, incident_field, x, y, z):
    incident_field = compute_incident_field(incident_field, x, y, z)
    scattered_field = compute_scattered_field(particles, x, y, z)
    outside_field = scattered_field + incident_field
    for s, particle in enumerate(particles):
        rx, ry, rz = x - particle.pos[0], y - particle.pos[1], z - particle.pos[2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        outside_field = np.where(r > particle.r, outside_field, 0)
    # inner_field = compute_inner_field(particles, x, y, z)
    total_field = outside_field  # + inner_field
    return total_field
