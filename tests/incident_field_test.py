import sphrs
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# coordinates
number_of_points = 200
l = 5
span_x = np.linspace(-l, l, number_of_points)
span_y = np.linspace(-l, l, number_of_points)
span_z = np.linspace(-l, l, number_of_points)
span = np.array([span_x, span_y, span_z])

# parameters of fluid
freq = 82
ro = 1.225
c_f = 331
k_fluid = 2 * np.pi * freq / c_f

# parameters of the spheres
c_sph = 1403
k_sph = 2 * np.pi * freq / c_sph
r_sph = 2
ro_sph = 1050
sphere = np.array([k_sph, r_sph, ro_sph])
spheres = np.array([sphere])

# parameters of configuration
pos1 = np.array([0, 0, 0])
pos2 = np.array([0, 0, 2.5])
poses = np.array([pos1])

# parameters of the field
k_x = 0  # 0.70711 * k_fluid
k_y = 0
k_z = k_fluid  # 0.70711 * k_fluid
k = np.array([k_x, k_y, k_z])

# order of decomposition
order = 5

# plane
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

x_p, y_p, z_p, span_v, span_h = sphrs.build_slice(span, plane_number, plane=plane)


def plots_for_tests(actual_data, desired_data):
    actual_data = np.real(actual_data).reshape(len(span_v), len(span_h))
    desired_data = np.real(desired_data).reshape(len(span_v), len(span_h))
    fig, ax = plt.subplots(1, 2)
    im1 = ax[0].imshow(actual_data, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_h.min(), span_h.max(), span_v.min(), span_v.max()])
    im2 = ax[1].imshow(desired_data, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_h.min(), span_h.max(), span_v.min(), span_v.max()])
    plt.show()


def incident_field_test():
    desired_incident_field = np.exp(1j * (k_x * x_p + k_y * y_p + k_z * z_p))
    actual_incident_field_array = sphrs.coefficient_array(order, k, sphrs.incident_coefficient, len(x_p)) * \
                                  sphrs.regular_wave_functions_array(order, x_p, y_p, z_p, k)
    # actual_incident_field = sphrs.accurate_mp_sum(actual_incident_field_array, len(x_p))
    actual_incident_field = np.sum(actual_incident_field_array, axis=0)
    plots_for_tests(actual_incident_field, desired_incident_field)
    np.testing.assert_allclose(actual_incident_field, desired_incident_field, rtol=1e-2)


incident_field_test()
