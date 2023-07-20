import numpy as np
import matplotlib, matplotlib.pyplot as plt

from reflection_matrix import reflection_element
from transformation import wvf_transform_cartesian, wvf_transform_cylindrical
from basics import k_contour
from plots import show_contour, show_field
from acsmuthi.utility.wavefunctions import outgoing_wvf

plt.rcdefaults()
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['axes.formatter.min_exponent'] = 1


def test_integrator():
    m, n, mu, nu = 0, 1, 0, 1
    k, pos1, pos2 = 1, np.array([-1, 1, 2]), np.array([3, 4, 3])

    k_waypoint = np.linspace(1.5, 6, 40)
    els = []
    for k_tested in k_waypoint:
        k_parallel = k_contour(
            k_start_deflection=0,
            k_stop_deflection=None,
            dk_imag_deflection=0.05,
            k_finish=k_tested,
            dk=0.01
        )
        els.append(reflection_element(m, n, mu, nu, k, pos1, pos2, k_parallel))
    show_contour(k_parallel)

    els1 = []
    for k_tested in k_waypoint:
        k_parallel = k_contour(
            k_start_deflection=0.5,
            k_stop_deflection=None,
            dk_imag_deflection=0.05,
            k_finish=k_tested,
            dk=0.01
        )
        els1.append(reflection_element(m, n, mu, nu, k, pos1, pos2, k_parallel))
    show_contour(k_parallel)

    els2 = []
    for k_tested in k_waypoint:
        k_parallel = k_contour(
            k_start_deflection=0,
            k_stop_deflection=None,
            dk_imag_deflection=0.045,
            k_finish=k_tested,
            dk=0.01
        )
        els2.append(reflection_element(m, n, mu, nu, k, pos1, pos2, k_parallel))
    show_contour(k_parallel)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(k_waypoint, np.real(els), linewidth=3)
    ax.plot(k_waypoint, np.real(els1), linewidth=3)
    ax.plot(k_waypoint, np.real(els2), linewidth=3)
    plt.show()


# test_integrator()


def test_transformation():
    m, n, k = 1, 1, 1

    x, z = np.linspace(-5, 5, 70), np.linspace(-6, -1, 70)
    xx, zz = np.meshgrid(x, z)
    yy = np.full_like(xx, 0.)

    k_parallel = k_contour(
        k_start_deflection=k - 0.2,
        k_stop_deflection=k + 0.2,
        dk_imag_deflection=0.01,
        k_finish=6,
        dk=0.01
    )

    exact_wvf = outgoing_wvf(m, n, xx, yy, zz, k)
    transformed_wvf = wvf_transform_cartesian(m, n, xx, yy, zz, k, k_parallel=k_parallel)

    show_contour(k_parallel)
    extent = [np.min(k * x), np.max(k * x), np.min(k * z), np.max(k * z)]
    show_field(np.abs(exact_wvf - transformed_wvf), extent=extent)
    show_field(np.real(exact_wvf), extent)
    show_field(np.real(transformed_wvf), extent)
    plt.show()


# test_transformation()
