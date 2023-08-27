import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib


plt.rcdefaults()  # reset to default
plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
matplotlib.rc('pdf', fonttype=42)


def draw_spectrum(name):
    colors = sns.color_palette("dark:salmon_r", 4)
    table = np.genfromtxt("spectrums_csv/"+name+"_sil.csv", skip_header=1, delimiter=",", dtype=float)
    table1 = np.genfromtxt("spectrums_csv/"+name+"_sap.csv", skip_header=1, delimiter=",", dtype=float)
    table_comsol = np.genfromtxt("spectrums_csv/"+name+"_comsol.csv", skip_header=5, delimiter=",", dtype=float)

    ka1, ecs1 = table1[:, 0], table1[:, 1]
    fx1, fz1 = table1[:, 2], table1[:, 4]

    ka, ecs = table[:, 0], table[:, 1]
    fx, fz = table[:, 2], table[:, 4]

    ka_comsol = 2 * np.pi * table_comsol[:, 0] / 1403
    scs_comsol, ecs_comsol = table_comsol[:, 4], table_comsol[:, 5]
    fx_comsol, fz_comsol = table_comsol[:, 1], table_comsol[:, 3]

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    ax[0].plot(ka, ecs, color=colors[0])
    ax[0].plot(ka1, ecs1, color=colors[2], linestyle='--')
    ax[0].scatter(ka_comsol, ecs_comsol, marker='x', color=colors[3])

    ax[1].plot(ka, fx, color=colors[0])
    ax[1].plot(ka1, fx1, color=colors[2], linestyle='--')
    ax[1].scatter(ka_comsol, fx_comsol, marker='x', color=colors[3])
    plt.show()


# draw_spectrum("one_golden_water")
