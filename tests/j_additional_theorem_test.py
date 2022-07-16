import wavefunctions as wvfs
import numpy as np
import rendering
import mathematics as mths
import classes as cls


# coordinates
bound, number_of_points = 7, 200
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
order = 13

# plane
plane = 'xz'
plane_number = int(number_of_points / 2) + 1

x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
m, n = 1, 1


dist = np.array([-1, 1, 1])
desired_j = wvfs.regular_wave_function_cls(m, n, x_p + dist[0], y_p + dist[1], z_p + dist[2], k_fluid)
srw_array = np.zeros(((order+1) ** 2, len(x_p)), dtype=complex)
i = 0
for munu in zip(wvfs.m_idx(order), wvfs.n_idx(order)):
    srw_array[i] = wvfs.regular_wave_function_cls(munu[0], munu[1], x_p, y_p, z_p, k_fluid) * \
                   wvfs.regular_separation_coefficient_cls(m, munu[0], n, munu[1], k_fluid, dist)
    i += 1
actual_j = mths.multipoles_fsum(srw_array, len(x_p))
rendering.plots_for_tests(actual_j, desired_j, span_v, span_h)
np.testing.assert_allclose(actual_j, desired_j, rtol=1e-2)
