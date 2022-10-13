import numpy as np
from acsmuthi.postprocessing import rendering
from acsmuthi.fields_expansions import PlaneWave


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
    incident_field = PlaneWave(1, k, np.array([0, 0, 0]), 'regular', order, direction)
    incident_field.compute_pressure_field(x_p, y_p, z_p)
    actual_field = incident_field.field
    incident_field.compute_exact_field(x_p, y_p, z_p)
    desired_field = incident_field.exact_field
    rendering.plots_for_tests(actual_field, desired_field, span_v, span_h)
    np.testing.assert_allclose(actual_field, desired_field, rtol=1e-2)


incident_field_test()
