import numpy as np
import matplotlib, matplotlib.pyplot as plt

from reflection.reflection import reflection_element_i, reflection_element_i_angled
from acsmuthi.linear_system.coupling.substrate_coupling_matrix import substrate_coupling_element
from reflection.basics import k_contour

from testing_plots import show_integrand

plt.rcdefaults()
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['axes.formatter.min_exponent'] = 1


def check_integrator():
    m, n, mu, nu = 3, 3, 3, 3
    k, pos1, pos2 = 1, np.array([-2, 0, 2]), np.array([-2, 2, 4])

    k_waypoint = 5 * np.logspace(-5, -2, 30, endpoint=True)

    els = np.zeros((5, *k_waypoint.shape), dtype=complex)
    for i, k_tested in enumerate(k_waypoint):
        k_parallel = k_contour(k_start_deflection=k - 0.1, k_stop_deflection=k + 0.1, dk_imag_deflection=0.0001,
                               k_finish=5, dk=k_tested)
        els[0, i] = reflection_element_i(m, n, mu, nu, k, pos1, pos2, k_parallel)
        k_parallel = k_contour(k_start_deflection=k - 0.1, k_stop_deflection=k + 0.1, dk_imag_deflection=0.0005,
                               k_finish=5, dk=k_tested)
        els[1, i] = reflection_element_i(m, n, mu, nu, k, pos1, pos2, k_parallel)
        k_parallel = k_contour(k_start_deflection=k - 0.1, k_stop_deflection=k + 0.1, dk_imag_deflection=0.001,
                               k_finish=5, dk=k_tested)
        els[2, i] = reflection_element_i(m, n, mu, nu, k, pos1, pos2, k_parallel)
        k_parallel = k_contour(k_start_deflection=k - 0.1, k_stop_deflection=k + 0.1, dk_imag_deflection=0.005,
                               k_finish=5, dk=k_tested)
        els[3, i] = reflection_element_i(m, n, mu, nu, k, pos1, pos2, k_parallel)
        k_parallel = k_contour(k_start_deflection=k - 0.1, k_stop_deflection=k + 0.1, dk_imag_deflection=0.01,
                               k_finish=5, dk=k_tested)
        els[4, i] = reflection_element_i(m, n, mu, nu, k, pos1, pos2, k_parallel)
    # show_contour(k_parallel)

    true_el = substrate_coupling_element(m, n, mu, nu, k, pos1, pos2)
    fig, ax = plt.subplots(figsize=(5, 4))
    for el in els:
        err = np.abs((np.array(el) - true_el) / true_el)
        ax.loglog(k_waypoint, err, linewidth=3)
    plt.show()


# check_integrator()


def check_integrand():
    m, n, mu, nu = 0, 0, 1, 1
    k, pos1, pos2 = 1, np.array([-1, 1, 2]), np.array([3, 4, 3])

    show_integrand(0., 4, -1, 1, 300, 'ref', k, pos1, pos2, m, n, mu, nu)
    plt.show()


# check_integrand()


def check_integrator_angled():
    m, n, mu, nu = 0, 1, 0, 1
    k, pos1, pos2 = 1, np.array([-1, 1, 2]), np.array([3, 4, 3])

    k_waypoint = np.linspace(0.0001, 0.05, 30)
    els, els1, els2, els3 = [], [], [], []
    for k_tested in k_waypoint:
        els.append(reflection_element_i_angled(m, n, mu, nu, k, pos1, pos2, beta_max=0, d_beta=k_tested))
        els1.append(reflection_element_i_angled(m, n, mu, nu, k, pos1, pos2, beta_max=1, d_beta=k_tested))
        els2.append(reflection_element_i_angled(m, n, mu, nu, k, pos1, pos2, beta_max=2, d_beta=k_tested))
        els3.append(reflection_element_i_angled(m, n, mu, nu, k, pos1, pos2, beta_max=5, d_beta=k_tested))

    exact = substrate_coupling_element(m, n, mu, nu, k, pos1, pos2)
    rel_err = np.abs((np.array(els) - exact) / exact)
    rel_err1 = np.abs((np.array(els1) - exact) / exact)
    rel_err2 = np.abs((np.array(els2) - exact) / exact)
    rel_err3 = np.abs((np.array(els3) - exact) / exact)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(k_waypoint, rel_err, linewidth=3, label='0')
    ax.loglog(k_waypoint, rel_err1, linewidth=3, label='1')
    ax.loglog(k_waypoint, rel_err2, linewidth=3, label='2')
    ax.loglog(k_waypoint, rel_err3, linewidth=3, linestyle='--', label='5')
    ax.legend()
    plt.show()


# check_integrator_angled()
