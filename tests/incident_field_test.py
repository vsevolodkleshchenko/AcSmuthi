import numpy as np
import rendering
import wavefunctions as wvfs
import classes as cls

# coordinates
bound, number_of_points = 5, 200
span = rendering.build_discretized_span(bound, number_of_points)

direction = np.array([0.70711, 0, 0.70711])
freq = 82  # [Hz]
p0 = 1  # [kg/m/s^2] = [Pa]
incident_field = cls.PlaneWave(direction, freq, p0)

# parameters of fluid
ro_fluid = 1.225  # [kg/m^3]
c_fluid = 331  # [m/s]
fluid = cls.Fluid(ro_fluid, c_fluid)

k_fluid = 2 * np.pi * freq / c_fluid

# order of decomposition
order = 17

# plane
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)

desired_incident_field = np.exp(1j * k_fluid * (incident_field.dir[0] * x_p + incident_field.dir[1] * y_p +
                                                incident_field.dir[2] * z_p))
actual_incident_field_array = wvfs.coefficient_array_inc_cls(order, incident_field.dir, wvfs.incident_coefficient, len(x_p)) * \
                              wvfs.regular_wave_functions_array_cls(order, x_p, y_p, z_p, k_fluid)
# actual_incident_field = mths.multipoles_fsum(actual_incident_field_array, len(x_p))
actual_incident_field = np.sum(actual_incident_field_array, axis=0)
rendering.plots_for_tests(actual_incident_field, desired_incident_field, span_v, span_h)
np.testing.assert_allclose(actual_incident_field, desired_incident_field, rtol=1e-2)
