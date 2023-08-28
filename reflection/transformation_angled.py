import numpy as np
import scipy.special as ss
import scipy.integrate as si

from reflection.basics import legendre_normalized, dec_to_cyl


def angle_contour(beta_max, d_beta):
    im_part = np.pi / 2 + 1j * np.arange(beta_max, 0. - d_beta, -d_beta)
    re_part = np.arange(np.pi / 2, np.pi + d_beta, d_beta) + 0j
    return np.concatenate([im_part, re_part])


def compute_transformation_integrand_angled(beta, k, rho, z, m, n):
    cos_b, sin_b = np.cos(beta), np.sin(beta)
    return legendre_normalized(m, n, cos_b) * ss.jn(m, k * sin_b * rho) * np.exp(1j * k * cos_b * z) * sin_b


def transform_cylindrical_angled(m, n, rho, phi, z, k, beta_max, d_beta):
    beta_contour = angle_contour(beta_max, d_beta)
    integrand = [compute_transformation_integrand_angled(beta, k, rho, z, m, n) for beta in beta_contour]
    integral = si.trapezoid(integrand, beta_contour)
    return 1j ** (m - n) * np.exp(1j * m * phi) * integral


def transform_cartesian_angled(m, n, x, y, z, k, beta_max, d_beta):
    rho, phi, z = dec_to_cyl(x, y, z)
    return transform_cylindrical_angled(m, n, rho, phi, z, k, beta_max, d_beta)


def wvf_transform_cartesian_angled(m, n, xx, yy, zz, k, beta_max=1, d_beta=0.01):
    wvf_values = []
    x_1d, y_1d, z_1d = np.concatenate(xx), np.concatenate(yy), np.concatenate(zz)
    for x, y, z in zip(x_1d, y_1d, z_1d):
        wvf_values.append(transform_cartesian_angled(m, n, x, y, z, k, beta_max, d_beta))
    return np.reshape(wvf_values, xx.shape)
