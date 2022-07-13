import numpy as np
import rendering
import wavefunctions as wvfs
import mathematics as mths

# coordinates
bound, number_of_points = 5, 200
span = rendering.build_discretized_span(bound, number_of_points)

freq = 82

# parameters of fluid
c_fluid = 331
k_fluid = 2 * np.pi * freq / c_fluid

# parameters of the field
k_x = 0.70711 * k_fluid
k_y = 0
k_z = 0.70711 * k_fluid
k = np.array([k_x, k_y, k_z])

# order of decomposition
order = 17

# plane
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)

desired_incident_field = np.exp(1j * (k_x * x_p + k_y * y_p + k_z * z_p))
actual_incident_field_array = wvfs.coefficient_array(order, k, wvfs.incident_coefficient, len(x_p)) * \
                              wvfs.regular_wave_functions_array(order, x_p, y_p, z_p, k)
# actual_incident_field = mths.multipoles_fsum(actual_incident_field_array, len(x_p))
actual_incident_field = np.sum(actual_incident_field_array, axis=0)
rendering.plots_for_tests(actual_incident_field, desired_incident_field, span_v, span_h)
np.testing.assert_allclose(actual_incident_field, desired_incident_field, rtol=1e-2)
