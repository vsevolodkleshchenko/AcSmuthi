import numpy as np

from acsmuthi.utility import wavefunctions as wvfs
from acsmuthi.initial_field import PlaneWave
from acsmuthi.fields_expansions import SphericalWaveExpansion


freq = 82  # [Hz]
c = 331  # [m/s]
k = 2 * np.pi * freq / c
direction = np.array([0.70711, 0, 0.70711])

# coordinates
x_p, z_p = np.meshgrid(np.linspace(-5, 5, 200), np.linspace(-5, 5, 200))
y_p = np.full_like(x_p, 0.)

# order of decomposition
order = 17


def test_incident_field_decomposition():
    incident_field = PlaneWave(k, 1, direction)
    desired_field = incident_field.compute_exact_field(x_p, y_p, z_p)
    incident_swe = SphericalWaveExpansion(1, k, np.array([0, 0, 0]), 'regular', order,
                                          coefficients=wvfs.incident_coefficients(direction, order))
    actual_field = incident_swe.compute_pressure_field(x_p, y_p, z_p)
    np.testing.assert_allclose(actual_field, desired_field, rtol=1e-2)
