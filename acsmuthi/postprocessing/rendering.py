import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def build_discretized_span(bound, number_of_points):
    r"""Build a mesh"""
    span_x = np.linspace(-bound, bound, number_of_points)
    span_y = np.linspace(-bound, bound, number_of_points)
    span_z = np.linspace(-bound, bound, number_of_points)
    return np.array([span_x, span_y, span_z])


def build_slice(span, plane_number, plane='xz'):
    r""" Build np.arrays of points of grid to build a slice plot """
    span_x, span_y, span_z = span[0], span[1], span[2]
    x, y, z, span_v, span_h = None, None, None, None, None
    if plane == 'xz':
        grid = np.vstack(np.meshgrid(span_y, span_x, span_z, indexing='ij')).reshape(3, -1).T
        y, x, z = grid[:, 0], grid[:, 1], grid[:, 2]
        span_v, span_h = span_x, span_z
    if plane == 'yz':
        grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
        x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]
        span_v, span_h = span_y, span_z
    if plane == 'xy':
        grid = np.vstack(np.meshgrid(span_z, span_x, span_y, indexing='ij')).reshape(3, -1).T
        z, x, y = grid[:, 0], grid[:, 1], grid[:, 2]
        span_v, span_h = span_x, span_y

    x_p = x[(plane_number - 1) * len(span_v) * len(span_h):
            (plane_number - 1) * len(span_v) * len(span_h) + len(span_v) * len(span_h)]
    y_p = y[(plane_number - 1) * len(span_v) * len(span_h):
            (plane_number - 1) * len(span_v) * len(span_h) + len(span_v) * len(span_h)]
    z_p = z[(plane_number - 1) * len(span_v) * len(span_h):
            (plane_number - 1) * len(span_v) * len(span_h) + len(span_v) * len(span_h)]
    return x_p, y_p, z_p, span_v, span_h


def slice_plot(tot_field, span_v, span_h, plane='xz'):
    r""" build a 2D heat-plot of tot_field with spheres in:
     XZ plane for span_y[plane_number] : --->z
     YZ plane for span_x[plane_number] : --->z
     XY plane for span_z[plane_number] : --->y """
    tot_field_reshaped = tot_field.reshape(len(span_v), len(span_h))
    fig, ax = plt.subplots()
    plt.xlabel(plane[1])
    plt.ylabel(plane[0])
    im = ax.imshow(tot_field_reshaped, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_h.min(), span_h.max(), span_v.min(), span_v.max()])
    plt.colorbar(im)
    plt.show()


def plots_for_tests(actual_data, desired_data, span_v, span_h):
    r"""Draw 2 plots of actual and desired data"""
    actual_data = np.real(actual_data).reshape(len(span_v), len(span_h))
    desired_data = np.real(desired_data).reshape(len(span_v), len(span_h))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    im1 = ax[0].imshow(actual_data, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_h.min(), span_h.max(), span_v.min(), span_v.max()])
    plt.colorbar(im1)
    im2 = ax[1].imshow(desired_data, norm=colors.CenteredNorm(), cmap='seismic', origin='lower',
                   extent=[span_h.min(), span_h.max(), span_v.min(), span_v.max()])
    plt.colorbar(im2)
    plt.show()
