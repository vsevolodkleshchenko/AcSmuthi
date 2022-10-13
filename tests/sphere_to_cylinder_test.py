import numpy as np
import scipy.special as ss
import scipy.integrate
from acsmuthi.utility import mathematics as mths, wavefunctions as wvfs


def incident_field_test(m, n, x, y, z, k):
    r, phi, theta = mths.dec_to_sph(x, y, z)
    zeta = r * np.sin(theta)
    desired_field = wvfs.outgoing_wvf(m, n, x, y, z, k)
    a_mn = np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
    coefficient = 1j ** (n - m) * np.exp(1j * m * phi) * a_mn / k
    h = np.linspace(0, 2.5, 10000)
    gamma = np.emath.sqrt(k ** 2 - h ** 2)
    legendre = np.zeros(len(h), dtype=complex)
    for i in range(len(gamma)):
        legendre[i] = ss.clpmn(m, n, gamma[i] / k, type=2)[0][-1][-1]
    d_integral = legendre * ss.jv(m, h * zeta) * np.exp(-1j * gamma * z) * h / gamma
    actual_field = coefficient * scipy.integrate.simps(d_integral, h)
    error = np.abs((actual_field - desired_field) / desired_field)
    print(desired_field, actual_field)
    print("error:", error)


incident_field_test(-1, 1, 2, 2, -2, 1.5)


def reflected_field_test(m, n, x, y, z, k):
    x_s, y_s, z_s = 1, 1, 1
    x_i, y_i, z_i = 1, 1, -1
    delta = np.abs(z_s - 0)
    a, alpha = 0.997, 0
    r_s, phi_s, theta_s = mths.dec_to_sph(x - x_s, y - y_s, z - z_s)
    zeta_s = r_s * np.sin(theta_s)
    h = np.linspace(0, 2.5, 10000)
    gamma = np.emath.sqrt(k ** 2 - h ** 2)
    a_mn = np.sqrt((2 * n + 1) / 4 / np.pi * ss.factorial(n - m) / ss.factorial(n + m))
    coefficient = 1j ** (n - m) * (-1) ** (n + m) * np.exp(1j * m * phi_s) * a_mn / k * a
    legendre = np.zeros(len(h), dtype=complex)
    for i in range(len(gamma)):
        legendre[i] = ss.clpmn(m, n, - gamma[i] / k, type=2)[0][-1][-1]
    d_integral = legendre * ss.jv(m, h * zeta_s) * np.exp(1j * gamma * (z - z_s + 2 * delta - 1j * alpha)) * h / gamma
    actual_field = coefficient * scipy.integrate.simps(d_integral, h)
    desired_field = (-1) ** (n + m) * wvfs.outgoing_wvf(m, n, x - x_i, y - y_i, z - z_i, k)
    error = np.abs((actual_field - desired_field) / desired_field)
    print(desired_field, actual_field)
    print("error:", error)


print(reflected_field_test(1, 1, 1.1, 1.1, 0.5, 1.5))
