import numpy as np


def compute_incident_field(x, y, z, particles, initial_field):
    incident_field = initial_field.compute_exact_field(x, y, z)
    for s, particle in enumerate(particles):
        xr, yr, zr = x - particle.position[0], y - particle.position[1], z - particle.position[2]
        r = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)
        incident_field = np.where(r > particle.radius, incident_field, 0)
    return incident_field


def compute_scattered_field(x, y, z, particles):
    r"""Counts scattered field on mesh x, y, z"""
    scattered_field_array = np.zeros((len(particles), *x.shape), dtype=complex)
    for s, particle in enumerate(particles):
        scattered_field_array[s] = particle.scattered_field.compute_pressure_field(x, y, z)
    scattered_field = np.sum(scattered_field_array, axis=0)
    return scattered_field


def compute_inner_field(x, y, z, particles):
    r"""Counts inner field in every sphere on mesh x, y, z"""
    inner_fields_array = np.zeros((len(particles), *x.shape), dtype=complex)
    for s, particle in enumerate(particles):
        inner_fields_array[s] = particle.inner_field.compute_pressure_field(x, y, z)
    return np.sum(inner_fields_array, axis=0)


def compute_total_field(x, y, z, particles, initial_field):
    incident_field = compute_incident_field(x, y, z, particles, initial_field)
    scattered_field = compute_scattered_field(x, y, z, particles)
    inner_field = compute_inner_field(x, y, z, particles)
    total_field = incident_field + scattered_field + inner_field
    return total_field
