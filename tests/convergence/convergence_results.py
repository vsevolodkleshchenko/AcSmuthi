import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()  # reset to default
plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')


def orders_spheres():
    ecs_lst, frc_lst = [], []
    for i in np.arange(1, 3):
        table = np.loadtxt(f"n_sph_order_csv/{i}_sph_order.csv", delimiter=",", dtype=str)
        orders = table[1:8, 0].astype(float)
        ecs_lst.append(table[1:8, 1].astype(float))
        frc_lst.append(np.abs(table[1:8, 2].astype(float)))
    ecs_lst1, frc_lst1 = [], []
    for i in np.arange(3, 8) ** 2:
        table = np.loadtxt(f"n_sph_order_csv/{i}_sph_order.csv", delimiter=",", dtype=str)
        orders1 = table[2:, 0].astype(float)
        ecs_lst1.append(table[2:, 1].astype(float))
        frc_lst1.append(np.abs(table[2:, 2].astype(float)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    for i in range(2):
        ax[0].plot(orders, 100 * np.abs(ecs_lst[i] - ecs_lst[i][-1]) / ecs_lst[i][-1], marker='.')
        ax[1].plot(orders, 100 * np.abs(frc_lst[i] - frc_lst[i][-1]) / frc_lst[i][-1], marker='.')

    for i in range(5):
        ax[0].plot(orders1, 100 * np.abs(ecs_lst1[i] - ecs_lst1[i][-1]) / ecs_lst1[i][-1], marker='.')
        ax[1].plot(orders1, 100 * np.abs(frc_lst1[i] - frc_lst1[i][-1]) / frc_lst1[i][-1], marker='.')

    ax[0].legend([1, 4, 9, 16, 25, 36, 49])
    ax[0].set_xlabel("N - order")
    # ax[0].set_ylabel("sigma_ex")
    ax[1].legend([1, 4, 9, 16, 25, 36, 49])
    ax[1].set_xlabel("N - order")
    # ax[1].set_ylabel("f")
    # fig.suptitle("Convergence for d/l ~ 1, D/l ~ 1")
    plt.show()


# orders_spheres()


def dist_2_ord(folder_name, distances, lda):
    for d in distances:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        table = np.loadtxt(folder_name+f"{d/lda}.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)

        ax[0].plot(orders, np.abs(ecs - ecs[10]) / ecs[10], marker='.')
        ax[1].plot(orders, np.abs(f - f[10]) / np.abs(f[10]), marker='.')

        ax[0].set_xlabel("order")
        ax[0].set_ylabel("sigma_ex")
        ax[1].set_xlabel("order")
        ax[1].set_ylabel("f")
        fig.suptitle(f"Convergence for d/l = {np.round(d/lda, 2)}")
        plt.show()


# dist_2_ord("distance2_csv/Dl1/Dl1order2sph", np.linspace(0.022, 0.1, 21), 0.0236)
# dist_2_ord("distance2_csv/Dl0.1/order2sph", np.linspace(0.022, 0.4, 21), 0.236)
# dist_2_ord("distance2_csv/Dl10/order2sph", np.linspace(0.022, 0.1, 21), 0.00236)


def timing():
    sol, frc, ecs = [], [], []
    for i in np.arange(1, 3) ** 2:
        table = np.loadtxt(f"n_sph_order_csv/{i}_sph_order.csv", delimiter=",", dtype=str)
        orders = table[1:8, 0].astype(float)
        sol.append(table[1:8, -3].astype(float))
        ecs.append(table[1:8, -2].astype(float))
        frc.append(table[1:8, -1].astype(float))

    sol1, frc1, ecs1 = [], [], []
    for i in np.arange(3, 8) ** 2:
        table = np.loadtxt(f"n_sph_order_csv/{i}_sph_order.csv", delimiter=",", dtype=str)
        orders1 = table[2:, 0].astype(float)
        sol1.append(table[2:, -3].astype(float))
        ecs1.append(table[2:, -2].astype(float))
        frc1.append(table[2:, -1].astype(float))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i in range(2):
        ax[0].semilogy(orders, sol[i], marker='.')
        # ax[0].plot(orders, ecs[i], marker='.')
        ax[1].plot(orders, frc[i], marker='.')

    for i in range(5):
        ax[0].semilogy(orders1, sol1[i], marker='.')
        # ax[0].plot(orders1, ecs1[i], marker='.')
        ax[1].plot(orders1, frc1[i], marker='.')

    ax[0].set_xlabel("N - order")
    ax[0].set_ylabel("t, c")
    ax[0].legend([1, 4, 9, 16, 25, 36, 49])
    # ax[1].set_xlabel("order")
    # ax[1].set_ylabel("t_cs")
    ax[1].set_xlabel("N - order")
    ax[1].set_ylabel("t, c")
    ax[1].legend([1, 4, 9, 16, 25, 36, 49])
    # fig.suptitle(f"Solving time, cross-sections time, forces time")
    plt.show()


# timing()
