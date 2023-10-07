import matplotlib.pyplot as plt
from scipy.integrate import trapz
import scipy.special as ss
import numpy as np

from reflection.basics import k_contour
from reflection.testing.testing_plots import show_contour


def integrand_f(x):
    integrand = []
    for xi in x:
        zi = np.emath.sqrt(1 - xi)
        integrand.append(ss.clpmn(1, 2, xi, type=2)[0][-1][-1] * np.exp(1j * zi))
    return integrand


def compute_integral(path, integrand_function):
    integrand = integrand_function(path)
    return trapz(integrand, path)


def check_integrating():
    path0 = np.arange(0, 305, 0.005)
    path1 = k_contour(k_start_deflection=0.5, k_stop_deflection=1.5, dk_imag_deflection=0.02, k_finish=305, dk=0.005)
    path2 = k_contour(k_start_deflection=0.5, k_stop_deflection=1.5, dk_imag_deflection=0.2, k_finish=305, dk=0.005)

    for path in (path0, path1, path2):
        # show_contour(path)
        integral = compute_integral(path, integrand_f)
        print(integral)

# check_integrating()


# def r(k_rho, w, rho1, rho2, c1, c2):
#     k1, k2 = w / c1, w / c2
#     k1z = np.emath.sqrt(k1 ** 2 - k_rho ** 2)
#     k2z = np.emath.sqrt(k2 ** 2 - k_rho ** 2)
#     return (k1z * rho2 - k2z * rho1) / (k1z * rho2 + k2z * rho1)


def tested_f(m, n, x):
    f = np.zeros_like(x)
    z = np.emath.sqrt(1 - x ** 2)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            f[i, j] = ss.clpmn(m, n, z[i, j], type=2)[0][-1][-1]
            # f[i, j] = ss.clpmn(m, n, z[i, j], type=3)[0][-1][-1]
    return f


# fig, ax = plt.subplots(1, 2, figsize=(8, 3))
# re, im = np.linspace(0.5, 4, 100), np.linspace(-2, 2, 100)
# rr, ii = np.meshgrid(re, im)
# extent = [np.min(re), np.max(re), np.min(im), np.max(im)]
# # m, n = 1, 3
# # int2d = tested_f(m, n, rr + 1j * ii)
# ax[0].imshow(np.real(r2d), extent=extent)
# ax[1].imshow(np.imag(r2d), extent=extent)
# plt.show()
