import numpy as np
import matplotlib, matplotlib.pyplot as plt

from reflection.basics import k_contour
from reflection.transformation import wvf_transform_cartesian
from testing_plots import show_contour, show_field, show_integrand
from acsmuthi.utility.wavefunctions import outgoing_wvf


plt.rcdefaults()
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['axes.formatter.min_exponent'] = 1


def test_transformation():
    m, n, k = 1, 3, 1

    x, z = np.linspace(-5, 5, 50), np.linspace(-6, -1, 50)
    xx, zz = np.meshgrid(x, z)
    yy = np.full_like(xx, 0.)

    exact_wvf = outgoing_wvf(m, n, xx, yy, zz, k)

    # k_parallel = k_contour(
    #     k_start_deflection=k - 0.2,
    #     k_stop_deflection=k + 0.2,
    #     dk_imag_deflection=0.01,
    #     k_finish=6,
    #     dk=0.01
    # )
    # show_contour(k_parallel)
    # exact_wvf = wvf_transform_cartesian(m, n, xx, yy, zz, k, k_parallel=k_parallel)

    k_parallel = k_contour(k_start_deflection=0, dk_imag_deflection=0.005, k_stop_deflection=None)
    transformed_wvf = wvf_transform_cartesian(m, n, xx, yy, zz, k, k_parallel=k_parallel)

    show_contour(k_parallel)
    extent = [np.min(k * x), np.max(k * x), np.min(k * z), np.max(k * z)]
    show_field(np.abs(np.real(exact_wvf) - np.real(transformed_wvf)), extent=extent)
    show_field(np.real(exact_wvf), extent)
    show_field(np.real(transformed_wvf), extent)
    plt.show()


# test_transformation()


def check_integrand():
    m, n = -1, 2
    k, pos = 1, np.array([2, 0, 3])

    show_integrand(0., 4, -1., 1., 200, 'trf', k, pos, m, n)
    plt.show()


# check_integrand()
