import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()  # reset to default
plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')


def orders_spheres():
    ecs_lst, frc_lst = [], []
    for i in range(1, 9):
        table = np.loadtxt(f"n_particles_order_csv/Dl_01/{i}_sph_order.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs_lst.append(table[1:, 1].astype(float))
        frc_lst.append(np.abs(table[1:, 2].astype(float)))
    ecs_lst1, frc_lst1 = [], []
    for i in np.arange(3, 8) ** 2:
        table = np.loadtxt(f"n_particles_order_csv/Dl_01/{i}_sph_order.csv", delimiter=",", dtype=str)
        orders1 = table[1:, 0].astype(float)
        ecs_lst1.append(table[1:, 1].astype(float))
        frc_lst1.append(np.abs(table[1:, 2].astype(float)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for i in range(1, 9):
        ax[0].plot(orders, ecs_lst[i - 1], marker='.', label=f"{i}")
        ax[1].plot(orders, frc_lst[i - 1], marker='.', label=f"{i}")

    # for i, s in enumerate(np.arange(3, 8) ** 2):
    #     ax[0].plot(orders1, 100 * ecs_lst1[i], marker='.', label=f"{s}")
    #     ax[1].plot(orders1, 100 * frc_lst1[i], marker='.', label=f"{s}")

    ax[0].set_xlabel("N - order")
    ax[0].set_ylabel("sigma_ex")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_xlabel("N - order")
    ax[1].set_ylabel("f")
    fig.suptitle("Convergence for d/l ~ 1, D/l ~ 1")
    plt.show()


# orders_spheres()


def dist_2_ord():
    lda = 331 / 18.8
    distances = np.linspace(2.2, 4 * lda, 31)
    for d in distances:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        table = np.loadtxt("distance_order_csv/Dl0_11/2sph_dl_"+str(np.round(d / lda, 2))+".csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)

        ax[0].plot(orders, ecs, marker='.')
        ax[1].plot(orders, f, marker='.')

        ax[0].set_xlabel("order")
        ax[0].set_ylabel("scattering cs")
        ax[1].set_xlabel("order")
        ax[1].set_ylabel("force")
        fig.suptitle(f"Convergence for d/lambda = {np.round(d/lda, 2)}")
        plt.show()


# dist_2_ord()


def timing():
    sol, frc, ecs = [], [], []
    for i in np.arange(1, 9):
        table = np.loadtxt(f"n_particles_order_csv/Dll_1/{i}_sph_order.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        sol.append(table[1:, -3].astype(float))
        ecs.append(table[1:, -2].astype(float))
        frc.append(table[1:, -1].astype(float))

    sol1, frc1, ecs1 = [], [], []
    for i in np.arange(3, 8) ** 2:
        table = np.loadtxt(f"n_particles_order_csv/Dll_1/{i}_sph_order.csv", delimiter=",", dtype=str)
        orders1 = table[1:, 0].astype(float)
        sol1.append(table[1:, -3].astype(float))
        ecs1.append(table[1:, -2].astype(float))
        frc1.append(table[1:, -1].astype(float))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(1, 9):
        ax[0].semilogy(orders, sol[i-1], marker='.', label=f"{i}")
        # ax[0].plot(orders, ecs[i-1], marker='.', label=f"{i}")
        ax[1].plot(orders, frc[i-1], marker='.', label=f"{i}")

    # for i, s in enumerate(np.arange(3, 8) ** 2):
    #     ax[0].semilogy(orders1, sol1[i], marker='.', label=f"{s}")
    #     # ax[0].plot(orders1, ecs1[i], marker='.', label={i})
    #     ax[1].plot(orders1, frc1[i], marker='.', label=f"{s}")

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


timing()


def regime():
    lda = 331 / 140
    sizes = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5]) * lda
    for r in sizes:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        table = np.loadtxt("particle_size_order_csv/Dl_"+str(np.round(2 * r / lda, 2))+".csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)

        ax[0].plot(orders, ecs, marker='.')
        ax[1].plot(orders, f, marker='.')

        ax[0].set_xlabel("order")
        ax[0].set_ylabel("sigma_ex")
        ax[1].set_xlabel("order")
        ax[1].set_ylabel("f")
        fig.suptitle(f"Convergence for D/l = {np.round(2 * r / lda, 2)}")
        plt.show()


# regime()