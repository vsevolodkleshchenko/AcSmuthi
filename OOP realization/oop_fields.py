import numpy as np
import wavefunctions as wvfs
import mathematics as mths
import reflection


class SphericalWaveExpansion:
    def __init__(self, amplitude, k, kind, order, coefficients=None):
        self.ampl = amplitude
        self.k = k
        self.field = None
        self.kind = kind  # 'regular' or 'outgoing'
        self.order = order
        self.coefficients = coefficients

    def compute_pressure_field(self, x, y, z):
        coefficients_array = np.split(np.repeat(self.coefficients, len(x)), (self.order + 1) ** 2)
        if self.kind == 'regular':
            wave_functions_array = wvfs.regular_wvfs_array(self.order, x, y, z, self.k)
        else:
            wave_functions_array = wvfs.outgoing_wvfs_array(self.order, x, y, z, self.k)
        field_array = coefficients_array * wave_functions_array
        self.field = mths.multipoles_fsum(field_array, len(x))


class PlaneWave(SphericalWaveExpansion):
    def __init__(self, amplitude, k, kind, order, direction):
        super().__init__(amplitude, k, kind, order)
        self.dir = direction
        self.coefficients = wvfs.incident_coefficients(direction, order)

    def intensity(self, density, sound_speed):
        return self.ampl ** 2 / (2 * density * sound_speed)


def compute_reflected_field(layer, x, y, z):
    r"""Counts reflected scattered field on mesh x, y, z"""
    layer.reflected_field.compute_pressure_field(x, y, z)
    reflected_field = layer.reflected_field.field
    return reflected_field


def compute_scattered_field(particles_array, x, y, z):
    r"""Counts scattered field on mesh x, y, z"""
    scattered_field_array = np.zeros((len(particles_array), len(x)), dtype=complex)
    for s, particle in enumerate(particles_array):
        particle.scattered_field.compute_pressure_field(x - particle.pos[0], y - particle.pos[1], z - particle.pos[2])
        scattered_field_array[s] = particle.scattered_field.field
    scattered_field = mths.spheres_fsum(scattered_field_array, len(x))
    return scattered_field


def compute_inner_field(particles_array, x, y, z):
    r"""Counts inner field in every sphere on mesh x, y, z"""
    inner_fields_array = np.zeros((len(particles_array), len(x)), dtype=complex)
    for s, particle in enumerate(particles_array):
        particle.inner_field.compute_pressure_field(x - particle.pos[0], y - particle.pos[1], z - particle.pos[2])
        rx, ry, rz = x - particle.pos[0], y - particle.pos[1], z - particle.pos[2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        particle.inner_field.field = np.where(r <= particle.r, particle.inner_field.field, 0)
        inner_fields_array[s] = particle.inner_field.field
    return mths.spheres_fsum(inner_fields_array, len(x))


def compute_incident_field(ps, x, y, z):
    direct = ps.incident_field.dir
    inc_field = np.exp(1j * ps.k_fluid * (x * direct[0] + y * direct[1] + z * direct[2]))

    if ps.interface:
        ref_direct = reflection.reflection_dir(direct, ps.interface.normal)
        h = ps.k_fluid * np.sqrt(1 - np.dot(ps.incident_field.dir, ps.interface.normal) ** 2)
        ref_coef = reflection.ref_coef_h(h, ps.omega, ps.fluid.speed, ps.interface.speed, ps.fluid.rho,
                                         ps.interface.rho)
        image_o = - 2 * ps.interface.normal * ps.interface.int_dist0
        inc_field += ref_coef * np.exp(1j * ps.k_fluid * ((x - image_o[0]) * ref_direct[0] + (y - image_o[1]) *
                                                          ref_direct[1] + (z - image_o[2]) * ref_direct[2]))
    return inc_field


def compute_total_field(ps, x, y, z):
    incident_field = compute_incident_field(ps, x, y, z)
    scattered_field = compute_scattered_field(ps.spheres, x, y, z)
    outside_field = scattered_field + incident_field
    if ps.interface:
        reflected_field = compute_reflected_field(ps.interface, x, y, z)
        outside_field += reflected_field
    for s, particle in enumerate(ps.spheres):
        rx, ry, rz = x - particle.pos[0], y - particle.pos[1], z - particle.pos[2]
        r = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)
        outside_field = np.where(r > particle.r, outside_field, 0)
    inner_field = compute_inner_field(ps.spheres, x, y, z)
    total_field = outside_field + inner_field
    return total_field
