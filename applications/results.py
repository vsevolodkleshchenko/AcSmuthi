import numpy as np
import matplotlib.pyplot as plt


def spectrum():
    table = np.loadtxt("spectrum_mltplecs_forces_1aerogelSW.csv", delimiter=",", dtype=str)
    dl = table[1:, 0].astype(float)
    ecs, ecs0, ecs1, ecs2 = table[1:, 1].astype(float), table[1:, 2].astype(float), table[1:, 3].astype(float), table[1:, 4].astype(float)
    ecs3, ecs4, ecs5 = table[1:, 5].astype(float), table[1:, 6].astype(float), table[1:, 7].astype(float)
    t0, t1, t2 = table[1:, 8].astype(float), table[1:, 9].astype(float), table[1:, 10].astype(float)
    t3, t4, t5 = table[1:, 11].astype(float), table[1:, 12].astype(float), table[1:, 13].astype(float)
    fig, ax = plt.subplots(figsize=(10, 5))

    # ax.semilogy(dl, ecs0)
    ax.semilogy(dl, ecs)
    ax.semilogy(dl, ecs1)
    ax.semilogy(dl, ecs2)
    ax.semilogy(dl, ecs3)
    ax.semilogy(dl, ecs4)
    ax.semilogy(dl, ecs5)

    # ax.semilogy(dl, t0)
    # ax.semilogy(dl, t1)
    # ax.semilogy(dl, t2)
    # ax.semilogy(dl, t3)
    # ax.semilogy(dl, t4)
    # ax.semilogy(dl, t5)

    # ax.plot(dl, t0)
    # ax.plot(dl, t1)
    # ax.plot(dl, t2)
    # ax.plot(dl, t3)
    # ax.plot(dl, t4)
    # ax.plot(dl, t5)

    legend = ["n=0", "n=1", "n=2", "n=3", "n=4", "n=5"]
    ax.legend(legend)
    ax.set_xlabel("d/lambda")
    ax.set_ylabel("sigma_n")
    ax.set_title("Scattering cross section for aerogel sphere in standing wave (d - diameter)")
    plt.show()


def forces_dist_na(name):
    table = np.loadtxt(name, delimiter=",", dtype=str)
    dl = table[1:, 0].astype(float)
    ecs, f1x, f2x = table[1:, 1].astype(float), table[1:, 2].astype(float), table[1:, 5].astype(float)
    fig, ax = plt.subplots(figsize=(2, 4))
    # ax.plot(dl, ecs)
    ax.plot(dl, f1x)
    ax.plot(dl, f2x)
    plt.show()


def forces_dist_x():
    table = np.loadtxt("forces_x_2aerogelSW.csv", delimiter=",", dtype=str)
    dl = table[1:, 0].astype(float)
    ecs, f1z, f2z = table[1:, 1].astype(float), table[1:, 4].astype(float), table[1:, 7].astype(float)
    fig, ax = plt.subplots(figsize=(2, 4))
    # ax.plot(dl, ecs)
    ax.plot(dl, f1z)
    ax.plot(dl, f2z)
    plt.show()


# spectrum()
# forces_dist_na("forces_node_2aerogelSW.csv")
# forces_dist_na("forces_antinode_2aerogelSW.csv")
# forces_dist_x()