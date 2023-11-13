import numpy as np
import scipy.special as ss
import scipy.integrate as si

from reflection.basics import legendre_normalized, k_contour_old, dec_to_cyl


def compute_transformation_integrand(kp, k, rho, z, m, n):
    g = np.sqrt(k ** 2 - kp ** 2)
    return legendre_normalized(m, n, g / k) * ss.jn(m, kp * rho) * np.exp(1j * g * z) * kp / g


def transform_cylindrical(m, n, rho, phi, z, k, k_parallel):
    integrand = [compute_transformation_integrand(kpi, k, rho, z, m, n) for kpi in k_parallel]
    integral = si.trapezoid(integrand, k_parallel)
    return 1j ** (m - n) / k * np.exp(1j * m * phi) * integral


def transform_cartesian(m, n, x, y, z, k, k_parallel):
    rho, phi, z = dec_to_cyl(x, y, z)
    return transform_cylindrical(m, n, rho, phi, z, k, k_parallel=k_parallel)


def wvf_transform_cartesian(m, n, xx, yy, zz, k, k_parallel):
    wvf_values = []
    x_1d, y_1d, z_1d = np.concatenate(xx), np.concatenate(yy), np.concatenate(zz)
    for x, y, z in zip(x_1d, y_1d, z_1d):
        wvf_values.append(transform_cartesian(m, n, x, y, z, k, k_parallel=k_parallel))
    # print(xx, zz, x_1d, z_1d, wvf_values, np.reshape(wvf_values, xx.shape), sep='\n')
    return np.reshape(wvf_values, xx.shape)


def wvf_transform_cylindrical(m, n, rr, pp, zz, k, k_parallel):
    wvf_values = []
    for rho, phi, z in zip(rr, pp, zz):
        wvf_values.append(transform_cylindrical(m, n, rho, phi, z, k, k_parallel=k_parallel))

    return np.reshape(wvf_values, rr.shape)

