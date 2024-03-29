import numpy as np
import scipy
import scipy.special as ss


def spherical_h1n(n, z, derivative=False):
    r"""Spherical Hankel function of the first kind"""
    if derivative:
        return ss.spherical_jn(n, z, derivative=True) + 1j * ss.spherical_yn(n, z, derivative=True)
    else:
        return scipy.special.spherical_jn(n, z) + 1j * scipy.special.spherical_yn(n, z)


def spherical_jn_der2(n, z):
    r"""Second derivative of spherical Bessel function"""
    if n == 0:
        return - ss.spherical_jn(1, z, derivative=True)
    else:
        return ss.spherical_jn(n - 1, z, derivative=True) + (n + 1) / z**2 * ss.spherical_jn(n, z) - \
               (n + 1) / z * ss.spherical_jn(n, z, derivative=True)


def dec_to_sph(x, y, z):
    """Transition from cartesian coordinates to spherical coordinates"""
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    if phi.shape == ():
        if phi < 0:
            phi += 2 * np.pi
    else:
        phi[phi < 0] += 2 * np.pi
    return r, phi, theta