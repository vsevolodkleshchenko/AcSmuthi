import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()  # reset to default
plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')


def orders_spheres():
    ecs_lst, frc_lst = [], []
    for i in range(1, 9):
        table = np.loadtxt(f"n_particles_order_csv/freq50/{i}sph.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs_lst.append(table[1:, 1].astype(float))
        frc_lst.append(np.abs(table[1:, 2].astype(float)))
    ecs_lst1, frc_lst1 = [], []
    for i in np.arange(3, 8) ** 2:
        table = np.loadtxt(f"n_particles_order_csv/freq50/{i}sph.csv", delimiter=",", dtype=str)
        orders1 = table[1:, 0].astype(float)
        ecs_lst1.append(table[1:, 1].astype(float))
        frc_lst1.append(np.abs(table[1:, 2].astype(float)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for i in range(1, 9):
        ax[0].plot(orders, ecs_lst[i - 1], marker='.', label=f"{i}")
        ax[1].plot(orders, frc_lst[i - 1], marker='.', label=f"{i}")

    for i, s in enumerate(np.arange(3, 8) ** 2):
        ax[0].plot(orders1, 100 * ecs_lst1[i], marker='.', label=f"{s}")
        ax[1].plot(orders1, 100 * frc_lst1[i], marker='.', label=f"{s}")

    ax[0].set_xlabel("N - order")
    ax[0].set_ylabel("sigma_ex")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel("N - order")
    ax[1].set_ylabel("f")
    plt.show()


def dist_2_ord():
    dl = np.array([0.001, 0.01, 0.1, 1, 5, 10, 15, 20])
    for d in dl:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        table = np.loadtxt(f"distance_order_csv/freq50/dl{d}"+".csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)

        ax[0].plot(orders, ecs, marker='.')
        ax[1].plot(orders, f, marker='.')

        ax[0].set_xlabel("order")
        ax[0].set_ylabel("scattering cs")
        ax[1].set_xlabel("order")
        ax[1].set_ylabel("force")
        plt.show()


def timing():
    sol, frc, ecs = [], [], []
    for i in range(1, 9):
        table = np.loadtxt(f"n_particles_order_csv/freq50/{i}sph.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        sol.append(table[1:, -3].astype(float))
        ecs.append(table[1:, -2].astype(float))
        frc.append(table[1:, -1].astype(float))

    sol1, frc1, ecs1 = [], [], []
    for i in np.arange(3, 8) ** 2:
        table = np.loadtxt(f"n_particles_order_csv/freq50/{i}sph.csv", delimiter=",", dtype=str)
        orders1 = table[1:, 0].astype(float)
        sol1.append(table[1:, -3].astype(float))
        ecs1.append(table[1:, -2].astype(float))
        frc1.append(table[1:, -1].astype(float))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(1, 9):
        ax[0].semilogy(orders, sol[i-1], marker='.', label=f"{i}")
        # ax[0].plot(orders, ecs[i-1], marker='.', label=f"{i}")
        ax[1].plot(orders, frc[i-1], marker='.', label=f"{i}")

    for i, s in enumerate(np.arange(3, 8) ** 2):
        ax[0].semilogy(orders1, sol1[i], marker='.', label=f"{s}")
        # ax[0].plot(orders1, ecs1[i], marker='.', label={i})
        ax[1].plot(orders1, frc1[i], marker='.', label=f"{s}")

    ax[0].set_xlabel("N - order")
    ax[0].set_ylabel("t, c")
    ax[0].legend()
    ax[1].legend()
    # ax[1].set_xlabel("order")
    # ax[1].set_ylabel("t_cs")
    ax[1].set_xlabel("N - order")
    ax[1].set_ylabel("t, c")
    fig.suptitle(f"Solving time, cross-sections time, forces time")
    plt.show()


def regime():
    ka_sizes = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10])
    for ka in ka_sizes:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        table = np.loadtxt(f"particle_size_order_csv/ka{ka}.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)

        ax[0].plot(orders, ecs, marker='.')
        ax[1].plot(orders, f, marker='.')

        ax[0].set_xlabel("order")
        ax[0].set_ylabel("sigma_ex")
        ax[1].set_xlabel("order")
        ax[1].set_ylabel("f")
        plt.show()


# dist_2_ord()
# timing()
# orders_spheres()
# regime()
