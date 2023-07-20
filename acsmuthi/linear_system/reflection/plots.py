import numpy as np
from matplotlib import pyplot as plt, colors


def show_contour(contour):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.real(contour), np.imag(contour), linewidth=3)
    ax.plot(np.real(contour), np.zeros_like(contour, dtype=float), '--r', linewidth=3)
    ax.set_xlabel('Re(k)')
    ax.set_ylabel('Im(k)')
    ax.set_title('Integration path')
    plt.tight_layout()
    # plt.show()


def show_integrand(kp, integrand):
    fig, ax = plt.subplots(figsize=(5, 3))


def show_field(field_map, extent):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(field_map, origin='lower', extent=extent, norm=colors.CenteredNorm(), cmap='RdBu_r')
    ax.set_xlabel('kx')
    ax.set_ylabel('kz')
    plt.colorbar(im, shrink=0.5)
    plt.tight_layout()
    # plt.show()
