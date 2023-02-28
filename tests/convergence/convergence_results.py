import numpy as np
import matplotlib.pyplot as plt

plt.rcdefaults()  # reset to default
plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')


def orders_spheres():
    table1 = np.loadtxt("n_sph_order_csv/1_sph_order.csv", delimiter=",", dtype=str)
    table2 = np.loadtxt("n_sph_order_csv/2_sph_order.csv", delimiter=",", dtype=str)
    table3 = np.loadtxt("n_sph_order_csv/3_sph_order.csv", delimiter=",", dtype=str)
    table4 = np.loadtxt("n_sph_order_csv/4_sph_order.csv", delimiter=",", dtype=str)
    table5 = np.loadtxt("n_sph_order_csv/5_sph_order.csv", delimiter=",", dtype=str)
    table6 = np.loadtxt("n_sph_order_csv/6_sph_order.csv", delimiter=",", dtype=str)
    table7 = np.loadtxt("n_sph_order_csv/7_sph_order.csv", delimiter=",", dtype=str)

    orders = table1[2:, 0].astype(float)
    ecs1, f1 = table1[2:, 1].astype(float), np.abs(table1[2:, 3].astype(float))
    ecs2, f2 = table2[2:, 1].astype(float), np.abs(table2[2:, 3].astype(float))
    ecs3, f3 = table3[2:, 1].astype(float), np.abs(table3[2:, 8].astype(float))
    ecs4, f4 = table4[2:, 1].astype(float), np.abs(table4[2:, 8].astype(float))
    ecs5, f5 = table5[2:, 1].astype(float), np.abs(table5[2:, 16].astype(float))
    ecs6, f6 = table6[2:, 1].astype(float), np.abs(table6[2:, 8].astype(float))
    ecs7, f7 = table7[2:, 1].astype(float), np.abs(table7[2:, 8].astype(float))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # ax[0].semilogy(orders, ecs1, marker='.')
    # ax[0].semilogy(orders, ecs2, marker='.')
    # ax[0].semilogy(orders, ecs3, marker='.')
    # ax[0].semilogy(orders, ecs4, marker='.')
    ax[0].semilogy(orders, ecs5, marker='.')
    # ax[0].semilogy(orders, ecs6, marker='.')
    # ax[0].plot(orders, ecs7, marker='.')

    # ax[1].plot(orders, f1, marker='.')
    # ax[1].plot(orders, f2, marker='.')
    # ax[1].plot(orders, f3, marker='.')
    # ax[1].plot(orders, f4, marker='.')
    ax[1].plot(orders, f5, marker='.')
    # ax[1].plot(orders, f6, marker='.')
    # ax[1].plot(orders, f7, marker='.')

    ax[0].legend(np.arange(1, 7))
    ax[0].set_xlabel("order")
    ax[0].set_ylabel("sigma_ex")
    ax[1].legend(np.arange(1, 7))
    ax[1].set_xlabel("order")
    ax[1].set_ylabel("f")
    fig.suptitle("Convergence for d/l ~ 1, D/l ~ 1")
    plt.show()


# orders_spheres()


def dist_2_ord_1():
    distances = np.linspace(0.022, 0.1, 21)
    for d in distances:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        table = np.loadtxt(f"distance2_csv/Dl1/Dl1order2sph{d/0.0236}.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)

        ax[0].plot(orders, np.abs(ecs - ecs[10]) / ecs[10], marker='.')
        ax[1].plot(orders, np.abs(f - f[10]) / np.abs(f[10]), marker='.')

        ax[0].set_xlabel("order")
        ax[0].set_ylabel("sigma_ex")
        ax[1].set_xlabel("order")
        ax[1].set_ylabel("f")
        fig.suptitle(f"Convergence for D/l ~ 1 d/l = {np.round(d/0.0236, 2)}")
        plt.show()


dist_2_ord_1()


def dist_2_ord_01():
    distances = np.linspace(0.022, 0.4, 21)
    for d in distances:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        table = np.loadtxt(f"distance2_csv/Dl0.1/order2sph{d/0.236}.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)

        ax[0].plot(orders, ecs, marker='.')
        ax[1].plot(orders, f, marker='.')

        ax[0].set_xlabel("order")
        ax[0].set_ylabel("sigma_ex")
        ax[1].set_xlabel("order")
        ax[1].set_ylabel("f")
        fig.suptitle(f"Convergence for D/l ~ 0.1 d/l = {np.round(d / 0.0236, 2)}")
        plt.show()


# dist_2_ord_01()
