import numpy as np
import seaborn as sns
import matplotlib, matplotlib.pyplot as plt

plt.rcdefaults()
matplotlib.rc('pdf', fonttype=42)
plt.rcParams['axes.formatter.min_exponent'] = 1


def dist_substrate_ord():
    kd = np.array([0.001, 0.01, 0.1, 1, 2, 5, 10])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    colors = sns.color_palette("dark:salmon", len(kd))
    for i, d in enumerate(kd):
        table = np.loadtxt(f"subs_distance_order_csv/freq82/kd{d}"+".csv", delimiter=",", dtype=str)
        orders = table[1:, 0].astype(float)
        ecs, f = table[1:, 1].astype(float), table[1:, 4].astype(float)
        ecs_err, f_err = np.abs((ecs[:-1] - ecs[-1]) / ecs[-1]), np.abs((f[:-1] - f[-1]) / f[-1])

        ax[0].semilogy(orders[:-1], ecs_err, marker='.', color=colors[i])
        ax[1].semilogy(orders[:-1], f_err, marker='.', color=colors[i], label=f"{d}")  # $\delta/\lambda$

    ax[0].set_xticks(orders[:-1])
    ax[1].set_xticks(orders[:-1])
    ax[0].set_yticks(np.logspace(-2, -14, 5))
    ax[1].set_yticks(np.logspace(-2, -14, 5))
    fig.legend(loc='outside lower right')
    plt.show()


# dist_substrate_ord()
