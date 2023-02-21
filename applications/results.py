import numpy as np
import matplotlib.pyplot as plt


def spectrum():
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    table = np.loadtxt("mie_aerogel/spectrum_mltplecs_forces_1aerogelSW.csv", delimiter=",", dtype=str)
    table = np.loadtxt("mie_aerogel/spectrum_mltplecs_1aerogelPW.csv", delimiter=",", dtype=str)
    tp = complex
    dl = table[1:, 0].astype(tp)
    ecs, ecs0, ecs1, ecs2 = table[1:, 1].astype(tp), table[1:, 2].astype(tp), table[1:, 3].astype(tp), table[1:, 4].astype(tp)
    ecs3, ecs4, ecs5 = table[1:, 5].astype(tp), table[1:, 6].astype(tp), table[1:, 7].astype(tp)
    t0, t1, t2 = table[1:, 8].astype(tp), table[1:, 9].astype(tp), table[1:, 10].astype(tp)
    t3, t4, t5 = table[1:, 11].astype(tp), table[1:, 12].astype(tp), table[1:, 13].astype(tp)

    import scipy.signal
    peaks1, _ = scipy.signal.find_peaks(ecs0)
    peaks2, _ = scipy.signal.find_peaks(ecs1)
    t0r, t1r, t2r, t3r = np.real(t0), np.real(t1), np.real(t2), np.real(t3)
    t0i, t1i, t2i, t3i = np.imag(t0), np.imag(t1), np.imag(t2), np.imag(t3)
    fig, ax = plt.subplots(figsize=(10, 6))

    # ax.plot(1/dl, ecs0)
    # ax.plot(1/dl, ecs1)
    # ax.plot(1/dl, ecs2)
    # ax.plot(1/dl, ecs3)
    # ax.plot(1/dl, ecs4)
    # ax.plot(1/dl, ecs5)
    # ax.plot(1/dl[peaks1], ecs0[peaks1], 'x')
    # ax.plot(1/dl[peaks2], ecs1[peaks2], 'x')
    # print(1/dl[peaks1[0]], 1/dl[peaks2[0]])

    # ax.semilogy(1/dl, ecs0)
    # ax.semilogy(1/dl, ecs1)
    # ax.semilogy(1/dl, ecs2)
    # ax.semilogy(1/dl, ecs3)
    # ax.semilogy(1/dl, ecs4)
    # ax.semilogy(1/dl, ecs5)
    # ax.semilogy(1/dl, ecs, '--k')

    ax.plot(1/dl, ecs0)
    ax.plot(1/dl, ecs1)
    ax.plot(1/dl, ecs2)
    # ax.semilogy(1/dl, np.abs(t3))
    # ax.semilogy(1/dl, np.abs(t4))
    # ax.semilogy(1/dl, np.abs(t5))

    # ax.plot(1/dl, np.abs(t0))
    # ax.plot(dl, np.abs(t1))
    # ax.plot(dl, np.abs(t2))
    # ax.plot(dl, np.abs(t3))
    # ax.plot(dl, np.abs(t4))
    # ax.plot(dl, np.abs(t5))

    # ax.plot(1/dl, t0r, 'r')
    # ax.plot(1/dl, t0i, '--r')
    # ax.plot(1/dl, -t1r, 'g')
    # ax.plot(1/dl, -t1i, '--g')
    # ax.plot(1/dl, t1r, 'b')
    # ax.plot(1/dl, t1i, '--b')
    # ax.plot(dl, 2 * t2r, 'g')
    # ax.plot(dl, 2 * t2i, '--g')
    ax.plot(1 / dl, np.zeros_like(dl), '-k')

    # legend = ["n=0", "n=1", "n=2", "n=3", "n=4", "n=5", "total"]
    legend = ["t0", "t1", "t2"]
    # legend = ['monopole', 'dipole']
    ax.legend(legend)
    plt.show()


def forces_dist_na(name):
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    table = np.loadtxt(name, delimiter=",", dtype=str)
    dl = table[1:, 0].astype(float)
    ecs, f1x, f1z = table[1:, 1].astype(float), table[1:, 2].astype(float), table[1:, 4].astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    # ax.plot(dl, ecs)
    ax.plot(dl, f1x)
    ax.plot(dl, f1z)
    ax.legend(["fz", "fx"])
    ax.plot(dl, np.zeros_like(dl), '--r')

    plt.show()


def forces_dist_x():
    table = np.loadtxt("mie_aerogel/forces_x_2aerogelSW.csv", delimiter=",", dtype=str)
    dl = table[1:, 0].astype(float)
    ecs, f1x, f1z = table[1:, 1].astype(float), table[1:, 2].astype(float), table[1:, 4].astype(float)
    fig, ax = plt.subplots(figsize=(4, 4))
    # ax.plot(dl, ecs)
    ax.plot(dl, f1x)
    ax.plot(dl, f1z)
    ax.legend(["fz", "fx"])
    plt.show()


# spectrum()
# forces_dist_na("mie_aerogel/forces_node_2aerogelSW.csv")

# forces_dist_na("new_mie_aerogel/forces_node_2aerogelSW_1l_wide_monom.csv")
# forces_dist_na("new_mie_aerogel/forces_node_2aerogelSW_1r_wide_monom.csv")
# forces_dist_na("new_mie_aerogel/forces_x_2aerogelSW_1l_wide_monom.csv")
# forces_dist_na("new_mie_aerogel/forces_x_2aerogelSW_1r_wide_monom.csv")

# forces_dist_na("new_mie_aerogel/forces_node_2aerogelSW_1l_wide.csv")
# forces_dist_na("new_mie_aerogel/forces_node_2aerogelSW_1r_wide.csv")

# forces_dist_na("new_mie_aerogel/forces_x_2aerogelSW_1r_wide_monom.csv")
# forces_dist_na("mie_aerogel/forces_antinode_2aerogelSW.csv")
# forces_dist_na("mie_aerogel/forces_node_2aerogelSWdl0.6.csv")
# forces_dist_na("mie_aerogel/forces_antinode_2aerogelSW.csv")
# forces_dist_x()


def forces3aerogel(name):
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    table = np.loadtxt(name, delimiter=",", dtype=str)
    dl = table[1:, 0].astype(float)
    ecs, f1x, f2x = table[1:, 1].astype(float), table[1:, 2].astype(float), table[1:, 5].astype(float)
    f1z, f2z, f3z = table[1:, 4].astype(float), table[1:, 7].astype(float), table[1:, 10].astype(float)
    f1, f2, f3 = np.sqrt(f1x ** 2 + f1z ** 2), np.sqrt(f2x ** 2 + f2z ** 2), np.abs(f3z)
    fig, ax = plt.subplots(figsize=(5, 4))
    # ax.plot(dl, ecs)
    ax.plot(dl, f1, 'royalblue')
    ax.plot(dl, f2,  'orange', linestyle='--')
    ax.plot(dl, f3, 'orangered')
    ax.plot(dl, np.zeros_like(dl), 'gray', linestyle='--')
    plt.show()


# forces3aerogel("mie_aerogel/forces_node_3aerogelSW.csv")


def orders_spheres():
    table1 = np.loadtxt("ordersph_csv/order1sph.csv", delimiter=",", dtype=str)
    table2 = np.loadtxt("ordersph_csv/order2sph.csv", delimiter=",", dtype=str)
    table3 = np.loadtxt("ordersph_csv/order3sph.csv", delimiter=",", dtype=str)
    table4 = np.loadtxt("ordersph_csv/order4sph.csv", delimiter=",", dtype=str)
    orders = table1[1:, 0].astype(float)
    ecs1, f1x = table1[1:, 1].astype(float), table1[1:, 2].astype(float)
    ecs2, f2z = table2[1:, 1].astype(float), table2[1:, 5].astype(float)
    ecs3, f3z = table3[1:, 1].astype(float), np.abs(table3[1:, 5].astype(float))
    ecs4, f4z = table4[1:, 1].astype(float), table4[1:, 7].astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # ax[0].scatter(orders, ecs1, marker='.')
    ax[0].scatter(orders, ecs2, marker='.')
    # ax[0].scatter(orders, ecs3, marker='.')
    # ax[0].scatter(orders, ecs4, marker='.')
    # ax[1].scatter(orders, f1x, marker='.')
    ax[1].scatter(orders, f2z, marker='.')
    # ax[1].scatter(orders, f3z, marker='.')
    # ax[1].plot(orders, f4z)
    # ax[0].legend(np.arange(1, 5))
    # ax[1].legend(np.arange(1, 4))
    ax[0].set_xlabel("order")
    ax[1].set_xlabel("order")
    ax[0].set_ylabel("sigma_ex")
    ax[1].set_ylabel("f")
    fig.suptitle("Convergence for d/l~1 D/l ~ 1")
    plt.show()


def distance_2spheres():
    distances = np.linspace(0.022, 0.2, 21)
    for d in distances[:8]:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        table = np.loadtxt(f"distance2_csv/order2sph{d}.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 2].astype(float)
        ax[0].plot(orders, ecs, marker='.')
        ax[1].plot(orders, f, marker='.')
        ax[0].set_xlabel("order")
        ax[1].set_xlabel("order")
        ax[0].set_ylabel("sigma_ex")
        ax[1].set_ylabel("f")
        fig.suptitle(f"Convergence for D/l~1 d/l = {np.round(d/331*12000, 2)}")
        plt.show()


def regime():
    rads = np.logspace(-4, 0, 21)
    for r in rads:
        fig, ax = plt.subplots(figsize=(10, 4))
        table = np.loadtxt(f"regime1_csv/order1sph{r}.csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs = table[1:, 1].astype(float)
        ax.scatter(orders, ecs, marker='.')
        ax.plot(orders, ecs, marker='.')
        ax.set_xlabel("order")
        ax.set_ylabel("sigma_ex")
        fig.suptitle(f"Convergence for D/l~{np.round(2*r/331*12000, 2)}")
        plt.show()


# orders_spheres()
# distance_2spheres()
# regime()

