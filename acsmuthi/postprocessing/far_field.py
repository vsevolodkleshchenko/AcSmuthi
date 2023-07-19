import numpy as np
import scipy.special as ss
from typing import Union, Literal

from ..linear_system.coupling_matrix import translation_block
from ..simulation import Simulation
import acsmuthi.utility.wavefunctions as wvfs


def far_field_pattern(
        simulation: Simulation,
        reference_point: np.ndarray = np.array([0., 0., 0.]),
        azimuthal_angles: Union[np.ndarray, Literal['default']] = 'default',
        polar_angles: Union[np.ndarray, Literal['default']] = 'default',
        angular_resolution: float = np.pi / 360,
):
    if azimuthal_angles == 'default' or polar_angles == 'default':
        phi = np.arange(0, 2 * np.pi + 0.5 * angular_resolution, angular_resolution, dtype=float),
        theta = np.arange(0, np.pi + 0.5 * angular_resolution, angular_resolution, dtype=float)
        azimuthal_angles, polar_angles = np.meshgrid(phi, theta)

    order = simulation.order

    scattered_coefs = np.zeros((order + 1) ** 2, dtype=complex)
    for particle in simulation.particles:
        scattered_coefs += translation_block(
            order=simulation.order,
            k_medium=simulation.initial_field.k,
            distance=reference_point - particle.position
        ) @ particle.scattered_field.coefficients

    i_spherical_harmonics = np.zeros(((order + 1) ** 2, *azimuthal_angles.shape), dtype=complex)
    for m, n in wvfs.multipoles(order):
        i_spherical_harmonics[n ** 2 + n + m] = (-1j) ** n * ss.sph_harm(m, n, azimuthal_angles, polar_angles)

    coefficients_array = np.broadcast_to(scattered_coefs, i_spherical_harmonics.T.shape).T
    far_field_array = coefficients_array * i_spherical_harmonics
    far_field = np.sum(far_field_array, axis=0)

    return far_field
    # return np.real(1j / simulation.initial_field.k * far_field)
