import numpy as np

from acsmuthi import layers
from acsmuthi.utility import mathematics as mths, reflection


def compute_reflected_field(medium, x, y, z):
    r"""Counts reflected scattered field on mesh x, y, z"""
    medium.reflected_field.compute_pressure_field(x, y, z)
    reflected_field = medium.reflected_field.field
    return reflected_field


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


def compute_incident_field(medium, freq, x, y, z, layer=None):
    direct = medium.incident_field.dir
    k = 2 * np.pi * freq / medium.speed_l
    # inc_field = np.exp(1j * k * (x * direct[0] + y * direct[1] + z * direct[2]))
    medium.incident_field.compute_exact_field(x, y, z)
    inc_field = medium.incident_field.exact_field
    if layer:
        ref_direct = reflection.reflection_dir(direct, layer.normal)
        ref_coef = layers.reflection_amplitude(medium, layer, freq)
        image_o = - 2 * layer.normal * layer.int_dist0
        inc_field += ref_coef * np.exp(1j * k * ((x - image_o[0]) * ref_direct[0] + (y - image_o[1]) *
                                                 ref_direct[1] + (z - image_o[2]) * ref_direct[2]))
    return inc_field


def compute_total_field(freq, medium, particles, x, y, z, layer=None):
    incident_field = compute_incident_field(medium, freq, x, y, z, layer=layer)
    scattered_field = compute_scattered_field(particles, x, y, z)
    outside_field = scattered_field + incident_field
    if layer:
        reflected_field = compute_reflected_field(medium, x, y, z)
        outside_field += reflected_field
    for s, particle in enumerate(particles):
        rx, ry, rz = x - particle.pos[0], y - particle.pos[1], z - particle.pos[2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        outside_field = np.where(r > particle.r, outside_field, 0)
    # inner_field = compute_inner_field(particles, x, y, z)
    total_field = outside_field  # + inner_field
    return total_field
