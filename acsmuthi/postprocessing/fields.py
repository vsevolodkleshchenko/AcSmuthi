from typing import Sequence
import numpy as np

from acsmuthi.simulation import Simulation
from acsmuthi.particles import Particle, SphericalParticle
from acsmuthi import fields_expansions as fldsex
from acsmuthi.utility import wavefunctions as wvfs
import scipy.special as ss
from acsmuthi.utility import mathematics as mths

# todo: warn about substrate


def compute_incident_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    simulation: Simulation
) -> np.ndarray:
    """Compute incident field at given points.

    Shape of output array equal to shape of coordinate arrays.
    """
    incident_field = simulation.initial_field.pressure_field(x=x, y=y, z=z, medium=simulation.medium)
    incident_field = cut_particles(field=incident_field, particles=simulation.particles, x=x, y=y, z=z)
    return incident_field.real


def compute_scattered_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    simulation: Simulation
) -> np.ndarray:
    """Compute particles scattered fields at given points.

    Shape of output array equal to shape of coordinate arrays.
    """
    particles = simulation.particles

    scattered_fields = np.zeros((len(particles), *x.shape), dtype=complex)
    for s, particle in enumerate(particles):
        scattered_fields[s] = particle.scattered_field.pressure_field(x=x, y=y, z=z)
    scattered_field = scattered_fields.sum(axis=0)

    scattered_field = cut_particles(field=scattered_field, particles=particles, x=x, y=y, z=z)
    if simulation.medium.is_substrate:
        scattered_field = np.where(z >= 0, scattered_field, 0)
    return scattered_field.real


def compute_inner_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    simulation: Simulation
) -> np.ndarray:
    """Compute particles inner fields at given points.

    Shape of output array equal to shape of coordinate arrays.
    """
    particles = simulation.particles
    inner_fields = np.zeros((len(particles), *x.shape), dtype=complex)
    for s, particle in enumerate(particles):
        if isinstance(particle, SphericalParticle):
            particle.inner_field = inner_field_spherical_particle(
                particle=particle,
                simulation=simulation
            )
        if particle.inner_field is not None:
            inner_fields[s] = particle.inner_field.pressure_field(x=x, y=y, z=z)
    return inner_fields.sum(axis=0).real


def compute_total_field(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    simulation: Simulation
) -> np.ndarray:
    """Compute total field (incident+scattered and inner) at given points.

    Shape of output array equal to shape of coordinate arrays.
    """
    incident_field = compute_incident_field(x=x, y=y, z=z, simulation=simulation)
    scattered_field = compute_scattered_field(x=x, y=y, z=z, simulation=simulation)
    inner_field = compute_inner_field(x=x, y=y, z=z, simulation=simulation)
    return incident_field + scattered_field + inner_field


def cut_particles(
    field: np.ndarray,
    particles: Sequence[Particle],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray
) -> np.ndarray:
    """Set field values inside particles to zero.
    """
    for particle in particles:
        xr = x - particle.position[0]
        yr = y - particle.position[1]
        zr = z - particle.position[2]
        r = np.sqrt(xr ** 2 + yr ** 2 + zr ** 2)
        field = np.where(r > particle.radius, field, 0)
    return field


def inner_field_spherical_particle(
    particle: SphericalParticle,
    simulation: Simulation
) -> fldsex.SphericalWaveExpansion:
    """Compute inner field of a spherical particle as a SphericalWaveExpansion."
    """
    sc_coefs = particle.scattered_field.coefficients
    inc_coefs_eff = np.linalg.inv(particle.t_matrix) @ sc_coefs
    in_coefs = np.zeros_like(sc_coefs)

    k = particle.incident_field.k
    k_p = 2 * np.pi * simulation.initial_field.freq / particle.c_longitudinal
    for m, n in wvfs.mn_idx(particle.n_max):
        imn = n ** 2 + n + m

        jn_ka = ss.spherical_jn(n, k * particle.radius)
        h1n_ka = mths.spherical_h1n(n, k * particle.radius)
        jn_kpa = ss.spherical_jn(n, k_p * particle.radius)
        in_coefs[imn] = (jn_ka * inc_coefs_eff[imn] + h1n_ka * sc_coefs[imn]) / jn_kpa

    return fldsex.SphericalWaveExpansion(
        amplitude=particle.incident_field.ampl,
        k=k_p,
        reference_point=particle.position,
        kind='regular',
        n_max=particle.n_max,
        inner_r=0,
        outer_r=particle.circumscribing_sphere_radius,
        lower_z=-np.inf,
        coefficients=in_coefs
    )
