import sphrs
import numpy as np
import scipy
import scipy.special
import matplotlib.pyplot as plt
from matplotlib import colors
import time
import math
import pywigxjpf as wig

# coordinates
number_of_points = 200
l = 4
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
r_sph = 1
ro_sph = 1050
sphere = np.array([k_sph, r_sph, ro_sph])
spheres = np.array([sphere, sphere])

# parameters of configuration
pos1 = np.array([0, 0, -2.5])
pos2 = np.array([0, 0, 2.5])
poses = np.array([pos1, pos2])

# parameters of the field
k_x = 0.70711 * k_fluid
k_y = 0
k_z = 0.70711 * k_fluid
k = np.array([k_x, k_y, k_z])

# order of decomposition
order = 15

# the tests will be in xz plane
plane_number = int(number_of_points / 2) + 1

grid = np.vstack(np.meshgrid(span_y, span_x, span_z, indexing='ij')).reshape(3, -1).T
y, x, z = grid[:, 0], grid[:, 1], grid[:, 2]

x_p = x[(plane_number - 1) * len(span_x) * len(span_z):
        (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]
y_p = y[(plane_number - 1) * len(span_x) * len(span_z):
        (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]
z_p = z[(plane_number - 1) * len(span_x) * len(span_z):
        (plane_number - 1) * len(span_x) * len(span_z) + len(span_x) * len(span_z)]


def incident_field_test():
    desired_incident_field = np.exp(1j * (k_x * x_p + k_y * y_p + k_z * z_p))
    actual_incident_field_array = sphrs.coefficient_array(order, k, sphrs.inc_coef, len(x_p)) * \
                            sphrs.regular_wvfs_array(order, x_p, y_p, z_p, k)
    actual_incident_field = sphrs.accurate_mp_sum(actual_incident_field_array, len(x_p))
    actual_incident_field = np.sum(actual_incident_field_array, axis=0)
    xz_d = np.real(desired_incident_field).reshape(len(span_y), len(span_z))
    xz_a = np.real(actual_incident_field).reshape(len(span_y), len(span_z))
    fig, ax = plt.subplots(1, 2)
    im1 = ax[0].imshow(xz_d, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_z.min(), span_z.max(), span_x.min(), span_x.max()])
    im2 = ax[1].imshow(xz_a, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_z.min(), span_z.max(), span_x.min(), span_x.max()])
    plt.show()
    np.testing.assert_allclose(actual_incident_field, desired_incident_field, rtol=1e-2)


# incident_field_test()
