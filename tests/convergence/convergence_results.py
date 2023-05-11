import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns

plt.rcdefaults()  # reset to default
# plt.rcParams['text.usetex'] = True
plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
plt.rcParams['axes.formatter.min_exponent'] = 1


def orders_spheres():
    ecs_lst, frc_lst = [], []
    for i in range(1, 9):
        table = np.loadtxt(f"n_particles_order_csv/freq50/{i}sph.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)
        ecs_err, f_err = np.abs((ecs[:-1] - ecs[-1]) / ecs[-1]), np.abs((f[:-1] - f[-1]) / f[-1])
        ecs_lst.append(ecs_err)
        frc_lst.append(f_err)
    ecs_lst1, frc_lst1 = [], []
    for i in np.arange(3, 8) ** 2:
        table = np.loadtxt(f"n_particles_order_csv/freq50/{i}sph.csv", delimiter=",", dtype=str)
        orders1 = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)
        ecs_err, f_err = np.abs((ecs[:-1] - ecs[-1]) / ecs[-1]), np.abs((f[:-1] - f[-1]) / f[-1])
        ecs_lst1.append(ecs_err)
        frc_lst1.append(f_err)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3))
    ax[0].semilogy(orders[:-1], np.full_like(orders[:-1], 0.01), color='grey', linestyle='--')
    ax[1].semilogy(orders[:-1], np.full_like(orders[:-1], 0.01), color='grey', linestyle='--')
    colors = sns.color_palette("dark:salmon", 7)
    for i in [1, 4]:  # range(1, 9):
        ax[0].semilogy(orders[:-1], ecs_lst[i - 1], marker='.', label=f"{i}", color=colors[i-1])
        ax[1].semilogy(orders[:-1], frc_lst[i - 1], marker='.', color=colors[i-1])

    for i, s in enumerate(np.arange(3, 8) ** 2):
        ax[0].semilogy(orders1[:-1], ecs_lst1[i], marker='.', label=f"{s}", color=colors[i+2])
        ax[1].semilogy(orders1[:-1], frc_lst1[i], marker='.', color=colors[i+2])

    ax[0].set_xticks(orders[:-1])
    ax[1].set_xticks(orders[:-1])
    fig.legend(loc='outside lower right')
    plt.show()


def dist_2_ord():
    dl = np.array([0.001, 0.01, 0.1, 1, 5, 10, 20])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = sns.color_palette("dark:salmon", len(dl))
    for i, d in enumerate(dl):
        table = np.loadtxt(f"distance_order_csv/freq50/dl{d}"+".csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)
        ecs_err, f_err = np.abs((ecs[:-1] - ecs[-1]) / ecs[-1]), np.abs((f[:-1] - f[-1]) / f[-1])

        ax[0].semilogy(orders[:-1], ecs_err, marker='.', color=colors[i])
        ax[1].semilogy(orders[:-1], f_err, marker='.', color=colors[i], label=f"{d}")  # $\delta/\lambda$

    ax[0].set_xticks(orders[:-1])
    ax[1].set_xticks(orders[:-1])
    ax[0].set_yticks(np.logspace(-2, -14, 5))
    ax[1].set_yticks(np.logspace(-2, -14, 5))
    fig.legend(loc='outside lower right')
    plt.show()


def timing():
    sol, frc, ecs = [], [], []
    for i in range(1, 9):
        table = np.loadtxt(f"n_particles_order_csv/freq150/{i}sph.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        sol.append(table[1:, -3].astype(float))
        ecs.append(table[1:, -2].astype(float))
        frc.append(table[1:, -1].astype(float))

    sol1, frc1, ecs1 = [], [], []
    for i in np.arange(3, 8) ** 2:
        table = np.loadtxt(f"n_particles_order_csv/freq150/{i}sph.csv", delimiter=",", dtype=str)
        orders1 = table[1:, 0].astype(float)
        sol1.append(table[1:, -3].astype(float))
        ecs1.append(table[1:, -2].astype(float))
        frc1.append(table[1:, -1].astype(float))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    colors = sns.color_palette("crest", 6)
    for i in range(1, 9):
        if i != 4: continue
        ax[0].loglog(orders[:-1], sol[i-1][:-1], marker='.', color=colors[0])
        # ax[0].plot(orders, ecs[i-1], marker='.', label=f"{i}")
        ax[1].loglog(orders[:-1], frc[i-1][:-1], marker='.', label=f"{i}", color=colors[0])

    for i, s in enumerate(np.arange(3, 8) ** 2):
        ax[0].loglog(orders1[:-1], sol1[i][:-1], marker='.', color=colors[i+1])
        # ax[0].plot(orders1, ecs1[i], marker='.', label={i})
        ax[1].loglog(orders1[:-1], frc1[i][:-1], marker='.', label=f"{s}", color=colors[i+1])
    ax[0].loglog(orders[:-1], orders[:-1] ** 4, linestyle='--', color='k')
    ax[1].loglog(orders[:-1], 0.01 * orders[:-1] ** 2, linestyle='--', color='k')

    ax[0].set_xticks(orders[:-1])
    ax[1].set_xticks(orders[:-1])
    ax[0].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    fig.legend(loc='outside lower right')

    fig, ax = plt.subplots()
    colors = sns.color_palette("crest", 6)
    n_particles = np.concatenate((np.arange(1, 9), np.arange(3, 8) ** 2))
    for i, order in enumerate([2, 4, 6, 8]):
        sol_time = np.concatenate((np.array(sol)[:, order], np.array(sol1)[:, order]))
        ax.loglog(n_particles, sol_time, marker='.', label='        '+str(order), color=colors[i])
    ax.legend()
    ax.loglog(n_particles, n_particles ** 2, linestyle='--', color='k')
    ax.set_xticks(n_particles)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(loc='lower right')
    plt.show()


def regime():
    ka_sizes = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10])
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    colors = sns.color_palette("rocket_r", len(ka_sizes))
    for i, ka in enumerate(ka_sizes):
        table = np.loadtxt(f"particle_size_order_csv/ka{ka}.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)
        ecs_err, f_err = np.abs((ecs[:-1] - ecs[-1]) / ecs[-1]), np.abs((f[:-1] - f[-1]) / f[-1])

        ax.semilogy(orders[:-1], ecs_err, marker='.', label=f'{ka}', color=colors[i])
        # ax[1].semilogy(orders[:-1], f_err, marker='.', label=f'{ka}', color=colors[i])
    ax.semilogy(orders, np.full_like(orders, 0.01), color='grey', linestyle='--')
    # ax[1].semilogy(orders, np.full_like(orders, 0.01), color='grey', linestyle='--')
    ax.set_xticks(np.arange(1, 15, 2))
    # ax[1].set_xticks(np.arange(1, 15, 2))
    ax.legend()
    # ax[1].legend()
    plt.show()


# dist_2_ord()
# timing()
# orders_spheres()
# regime()
