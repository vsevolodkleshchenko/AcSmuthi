import numpy as np
import matplotlib, matplotlib.pyplot as plt

from reflection.reflection import reflection_element_i
from acsmuthi.linear_system.substrate_coupling_matrix import substrate_coupling_element
from reflection.basics import k_contour

from testing_plots import show_contour, show_integrand

plt.rcdefaults()
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['axes.formatter.min_exponent'] = 1


def test_integrator():
    m, n, mu, nu = 0, 1, 0, 1
    k, pos1, pos2 = 1, np.array([-1, 1, 2]), np.array([3, 4, 3])

    k_waypoint = np.linspace(1.5, 6, 70)
    els = []
    for k_tested in k_waypoint:
        k_parallel = k_contour(k_start_deflection=0, dk_imag_deflection=0.005, k_stop_deflection=None, k_finish=k_tested)

        els.append(reflection_element_i(m, n, mu, nu, k, pos1, pos2, k_parallel))
    show_contour(k_parallel)

    els1 = []
    for k_tested in k_waypoint:
        k_parallel = k_contour(
            k_start_deflection=0,
            k_stop_deflection=None,
            dk_imag_deflection=0.05,
            k_finish=k_tested,
            dk=0.01
        )
        els1.append(reflection_element_i(m, n, mu, nu, k, pos1, pos2, k_parallel))
    show_contour(k_parallel)

    els2 = []
    for k_tested in k_waypoint:
        k_parallel = k_contour(
            k_start_deflection=k-0.2,
            k_stop_deflection=k+0.2,
            dk_imag_deflection=0.04,
            k_finish=k_tested,
            dk=0.01
        )
        els2.append(reflection_element_i(m, n, mu, nu, k, pos1, pos2, k_parallel))
    show_contour(k_parallel)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(k_waypoint, np.real(els), linewidth=3)
    ax.plot(k_waypoint, np.real(els1), linewidth=3)
    ax.plot(k_waypoint, np.real(els2), linewidth=3)
    ax.plot(k_waypoint, np.full_like(k_waypoint, substrate_coupling_element(m, n, mu, nu, k, pos1, pos2)), linewidth=3)
    plt.show()


# test_integrator()


def check_integrand():
    m, n, mu, nu = 0, 2, 0, 2
    k, pos1, pos2 = 1, np.array([-1, 1, 2]), np.array([3, 4, 3])

    show_integrand(0., 4, -1, 1, 300, 'ref', k, pos1, pos2, m, n, mu, nu)
    plt.show()


# check_integrand()
