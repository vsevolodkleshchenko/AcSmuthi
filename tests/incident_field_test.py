import numpy as np
from postprocessing import rendering
from utility import wavefunctions as wvfs


freq = 82  # [Hz]
c = 331  # [m/s]
k = 2 * np.pi * freq / c
direction = np.array([0.70711, 0, 0.70711])

# coordinates
bound, number_of_points = 5, 200
span = rendering.build_discretized_span(bound, number_of_points)
plane = 'xz'
plane_number = int(number_of_points / 2) + 1
x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)

# order of decomposition
order = 17


def incident_field_test():
    desired_incident_field = np.exp(1j * k * (direction[0] * x_p + direction[1] * y_p + direction[2] * z_p))
    actual_incident_field_array = wvfs.incident_coefficients_array(direction, len(x_p), order) * \
                                  wvfs.regular_wvfs_array(order, x_p, y_p, z_p, k)
    # actual_incident_field = mths.multipoles_fsum(actual_incident_field_array, len(x_p))
    actual_incident_field = np.sum(actual_incident_field_array, axis=0)
    rendering.plots_for_tests(actual_incident_field, desired_incident_field, span_v, span_h)
    np.testing.assert_allclose(actual_incident_field, desired_incident_field, rtol=1e-2)


incident_field_test()
