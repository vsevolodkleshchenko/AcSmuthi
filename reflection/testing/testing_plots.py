import numpy as np
from matplotlib import pyplot as plt, colors
from reflection.reflection import compute_reflection_integrand, compute_reflection_integrand_angled
from reflection.transformation import compute_transformation_integrand
from reflection.transformation_angled import compute_transformation_integrand_angled


def compute_reflection_integrand_2d(re_kp, im_kp, k, emitter_pos, receiver_pos, m, n, mu, nu):
    dist = receiver_pos - emitter_pos
    d_rho, dz = np.sqrt(dist[0] ** 2 + dist[1] ** 2), dist[2]
    ds = np.abs(emitter_pos[2])

    integrand_values = []
    re_kp_1d, im_kp_1d = np.concatenate(re_kp), np.concatenate(im_kp)
    for re, im in zip(re_kp_1d, im_kp_1d):
        integrand_values.append(compute_reflection_integrand(re + 1j * im, k, d_rho, dz, ds, m, n, mu, nu))
    return np.reshape(integrand_values, re_kp.shape)


def compute_reflection_integrand_2d_angled(re_kp, im_kp, k, emitter_pos, receiver_pos, m, n, mu, nu):
    dist = receiver_pos - emitter_pos
    d_rho, dz = np.sqrt(dist[0] ** 2 + dist[1] ** 2), dist[2]
    ds = np.abs(emitter_pos[2])

    integrand_values = []
    re_kp_1d, im_kp_1d = np.concatenate(re_kp), np.concatenate(im_kp)
    for re, im in zip(re_kp_1d, im_kp_1d):
        integrand_values.append(compute_reflection_integrand_angled(re + 1j * im, k, d_rho, dz, ds, m, n, mu, nu))
    return np.reshape(integrand_values, re_kp.shape)


def compute_transformation_integrand_2d(re_kp, im_kp, k, pos, m, n):
    rho, z = np.sqrt(pos[0] ** 2 + pos[1] ** 2), pos[2]

    integrand_values = []
    re_kp_1d, im_kp_1d = np.concatenate(re_kp), np.concatenate(im_kp)
    for re, im in zip(re_kp_1d, im_kp_1d):
        integrand_values.append(compute_transformation_integrand(re + 1j * im, k, rho, z, m, n))
    return np.reshape(integrand_values, re_kp.shape)


def compute_transformation_integrand_2d_angled(re_kp, im_kp, k, pos, m, n):
    rho, z = np.sqrt(pos[0] ** 2 + pos[1] ** 2), pos[2]

    integrand_values = []
    re_kp_1d, im_kp_1d = np.concatenate(re_kp), np.concatenate(im_kp)
    for re, im in zip(re_kp_1d, im_kp_1d):
        integrand_values.append(compute_transformation_integrand_angled(re + 1j * im, k, rho, z, m, n))
    return np.reshape(integrand_values, re_kp.shape)


def show_integrand(re_kp_min, re_kp_max, im_kp_min, im_kp_max, num_points, integrand_type, *args):
    if integrand_type == 'trf':
        func_2d = compute_transformation_integrand_2d
    elif integrand_type == 'ref':
        func_2d = compute_reflection_integrand_2d
    elif integrand_type == 'trf_a':
        func_2d = compute_transformation_integrand_2d_angled
    elif integrand_type == 'ref_a':
        func_2d = compute_reflection_integrand_2d_angled

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    re_kp, im_kp = np.linspace(re_kp_min, re_kp_max, num_points), np.linspace(im_kp_min, im_kp_max, num_points)
    re_kp_grid, im_kp_grid = np.meshgrid(re_kp, im_kp)
    integrand_2d = func_2d(re_kp_grid, im_kp_grid, *args)

    extent = [re_kp_min, re_kp_max, im_kp_min, im_kp_max]
    im = ax[0].imshow(np.abs(np.real(integrand_2d)), origin='lower', extent=extent, norm=colors.LogNorm(), cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, shrink=0.5)

    im = ax[1].imshow(np.abs(np.imag(integrand_2d)), origin='lower', extent=extent, norm=colors.LogNorm(), cmap='RdBu_r', aspect='equal')
    plt.colorbar(im, shrink=0.5)

    plt.tight_layout()


def show_contour(contour):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.real(contour), np.imag(contour), linewidth=3)
    ax.plot(np.real(contour), np.zeros_like(contour, dtype=float), '--r', linewidth=3)
    ax.set_xlabel('Re(k)')
    ax.set_ylabel('Im(k)')
    ax.set_title('Integration path')
    plt.tight_layout()
    # plt.show()


def show_field(field_map, extent):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(field_map, origin='lower', extent=extent, norm=colors.CenteredNorm(), cmap='RdBu_r')
    ax.set_xlabel('kx')
    ax.set_ylabel('kz')
    plt.colorbar(im, shrink=0.5)
    plt.tight_layout()
    # plt.show()
