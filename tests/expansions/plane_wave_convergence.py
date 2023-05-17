import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import csv
import seaborn as sns

from acsmuthi.initial_field import PlaneWave

plt.rcdefaults()
plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')


def write_csv(data, fieldnames, filename):
    with open(".\\convergence_csv\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


def plane_wave_fields(order, x, y, z):
    k, c, direction = 1, 331, np.array([0., 1., 0])
    plane_wave = PlaneWave(k, 1, direction)
    desired_field = plane_wave.compute_exact_field(x, y, z)
    plane_swe = plane_wave.spherical_wave_expansion(np.array([0, 0, 0]), order)
    actual_field = plane_swe.compute_pressure_field(x, y, z)
    return actual_field, desired_field


def find_n_dependence_on_kr():
    xx, zz = np.meshgrid(np.linspace(-7.1, 7.1, 190), np.linspace(-7.1, 7.1, 190))
    yy, rr = np.full_like(xx, 0.), np.sqrt(xx ** 2 + zz ** 2)
    orders, tolerances = np.arange(13), [0.05, 0.01, 0.001]
    rmax = np.zeros((len(tolerances), len(orders) + 1), dtype=float)
    for i, order in enumerate(orders):
        actual, desired = plane_wave_fields(order, xx, yy, zz)
        nonzero = np.nonzero(actual)
        rel_err = np.abs((desired[nonzero]-actual[nonzero])/actual[nonzero])
        for j, tol in enumerate(tolerances):
            rel_max = np.max(rr[nonzero][np.argwhere(rel_err <= tol)])
            print('rel_max:', rel_max)
            rmax[j, i + 1] = np.round(rel_max, 3)
    rmax[:, 0] = tolerances
    write_csv(rmax, [0] + list(orders), 'rel_max')


def draw_decomposition_results():
    xx, zz = np.meshgrid(np.linspace(-5.1, 5.1, 351), np.linspace(-5.1, 5.1, 351))
    yy = np.full_like(xx, 0.)
    orders = [5, 8, 12]
    ims = []
    fig, ax = plt.subplots(1, len(orders), figsize=(6, 2.5))
    for i, n in enumerate(orders):
        actual, desired = plane_wave_fields(n, xx, yy, zz)
        rel_err = np.abs((actual - desired) / actual)
        data = np.where(rel_err > 1e-13, rel_err, 1e-13)
        im = ax[i].contourf(
            np.where(np.abs(actual) > 1e-13, data, np.inf),
            origin='lower',
            extent=[xx.min(), xx.max(), zz.min(), zz.max()],
            levels=15,
            norm=colors.LogNorm(vmin=1e-16, vmax=1e2),
            cmap=sns.color_palette("rocket_r", as_cmap=True),
            aspect='equal'
        )
        ims.append(im)
    cbar = fig.colorbar(ims[1], ax=ax.ravel().tolist())
    cbar.ax.locator_params(nbins=5)
    plt.show()


def show_convergence():
    rtable = np.loadtxt("convergence_csv/rel_max.csv", delimiter=",", dtype=float)
    orders, tolerances = rtable[0, 1:], rtable[1:, 0]
    fig, ax = plt.subplots()
    clrs = sns.color_palette("rocket_r", rtable.shape[0] - 1)
    for i in range(1, rtable.shape[0]):
        ax.scatter(orders[:-1:1], rtable[i, 1:-1:1], color=clrs[i-1], label="        ="+str(100 * tolerances[i-1]))
    ax.set_yticks(np.arange(0, 10, 2))
    ax.set_xticks(orders[:-1:2])
    ax.legend()
    plt.show()


# find_n_dependence_on_kr()
# show_convergence()
draw_decomposition_results()
