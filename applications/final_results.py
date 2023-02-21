import numpy as np
import matplotlib.pyplot as plt


def spectrum():
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    table1 = np.loadtxt("final_exp/spectrum_mono_aerogelPW_D2_f.csv", delimiter=",", dtype=str)
    table2 = np.loadtxt("final_exp/spectrum_mono_aerogelPW_D2.1_f.csv", delimiter=",", dtype=str)
    tp = complex
    ld = table1[1:, 0].astype(tp)
    ecs1, t01 = table1[1:, 1].astype(tp), table1[1:, 2].astype(tp)
    ecs2, t02 = table2[1:, 1].astype(tp), table2[1:, 2].astype(tp)

    t01r, t01i, t02r, t02i = np.real(t01), np.imag(t01), np.real(t02), np.imag(t02)
    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.plot(ld, t01r)
    # ax.plot(ld, t01i)
    # ax.plot(ld, t02r)
    # ax.plot(ld, t02i)
    ax.plot(ld, t01i)
    ax.plot(ld, np.abs(t01), 'royalblue', linestyle='--')
    # ax.plot(ld, np.abs(t02), 'tab:orange', linestyle='--')
    ax.legend(['t0i', 't0'])

    ax.plot(ld, np.zeros_like(ld), '--r')
    plt.show()


def forces3aerogel():
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    table = np.loadtxt("final_exp/forces_3aerogelSW_l.csv", delimiter=",", dtype=str)
    table1 = np.loadtxt("final_exp/forces_3aerogelSW_m.csv", delimiter=",", dtype=str)
    dl = table[6:40, 0].astype(float) # / 0.02 / 9.27
    dm = table1[6:40, 0].astype(float) # / 0.02 / 9.27
    dr = table[1:, 0].astype(float) / 0.02 / 9.55
    f1x, f2x, f3x = table[6:40, 2].astype(float), table[6:40, 5].astype(float), table[6:40, 8].astype(float)
    f1x1, f2x1, f3x1 = table1[6:40, 2].astype(float), table1[6:40, 5].astype(float), table1[6:40, 8].astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dl, f2x)
    ax.plot(dm, f2x1)
    ax.legend(["l", "m"])
    ax.plot(dl, np.zeros_like(dl), '--r')
    plt.show()


def forces2dist():
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    table = np.loadtxt("new_mie_aerogel/forces_node_2aerogelSW_1l_wide.csv", delimiter=",", dtype=str)
    table1 = np.loadtxt("new_mie_aerogel/forces_node_2aerogelSW_1r_wide.csv", delimiter=",", dtype=str)
    dl = table[5:50, 0].astype(float)
    dl1 = table[5:50, 0].astype(float)
    fx1, fx2 = table[5:50, 2].astype(float), table1[5:50, 2].astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    # ax.plot(dl, ecs)
    ax.plot(dl, fx1)
    ax.plot(dl1, fx2)
    ax.legend(["l", "r"])
    ax.plot(dl, np.zeros_like(dl), '--r')

    plt.show()


# spectrum()
# forces3aerogel()
# forces2dist()


def forces3aerogel_dream():
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    table = np.loadtxt("final_exp/forces_3aerogelSW_l_mo.csv", delimiter=",", dtype=str)
    table1 = np.loadtxt("final_exp/forces_3aerogelSW_m_mo.csv", delimiter=",", dtype=str)
    dl = table[3:, 0].astype(float) # / 0.02 / 9.27
    dm = table1[3:, 0].astype(float) # / 0.02 / 9.27
    dr = table[1:, 0].astype(float) / 0.02 / 9.55
    f1x, f2x, f3x = table[3:, 2].astype(float), table[3:, 5].astype(float), table[3:, 8].astype(float)
    f1x1, f2x1, f3x1 = table1[3:, 2].astype(float), table1[3:, 5].astype(float), table1[3:, 8].astype(float)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(dl, f2x)
    ax.plot(dm, f2x1)
    ax.legend(["l", "m"])
    ax.plot(dl, np.zeros_like(dl), '--r')
    plt.show()


# forces3aerogel_dream()


def spec_3sph():
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    table1 = np.loadtxt("final_exp/spectrum_mltplecs_forces_3aerogelSW.csv", delimiter=",", dtype=str)

    ld = table1[1:, 0].astype(complex)
    ecs, ecs0 = table1[1:, 1].astype(complex), table1[1:, 2].astype(complex)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ld, ecs)
    ax.plot(ld, np.real(ecs0))
    ax.plot(ld, np.imag(ecs0))

    ax.plot(ld, np.zeros_like(ld), '--r')
    plt.show()


# spec_3sph()


def dream():
    plt.rcdefaults()  # reset to default
    # plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    lD = np.linspace(8.90, 9.64, 30)
    # lD = np.linspace(8.9, 12, 30)
    for i, lDi in enumerate(lD):
        table = np.loadtxt(f"final_exp/dream/forces_3aerogelSW_{lDi}_mo.csv", delimiter=",", dtype=str)
        dl, fx = table[1:, 0].astype(float), table[1:, 5].astype(float)
        fx3 = table[1:, 8].astype(float)
        fig, ax = plt.subplots(figsize=(5, 4))
        # ax.plot(dl, fx)
        ax.plot(dl, fx3)
        ax.plot(dl, np.zeros_like(dl), '--r')
        ax.legend([i, lDi])
        plt.show()


# dream()


def dream_dream():
    plt.rcdefaults()  # reset to default
    plt.style.use('https://raw.githubusercontent.com/toftul/plt-styles-phys/main/phys-plots-sans.mplstyle')
    lD = np.linspace(8.90, 9.64, 30)
    first_stable_min = [0.526, 0.529, 0.529, 0.528, 0.533, 0.533, 0.536, 0.536, 0.538, 0.538,
                        0.785, 0.785, 0.788, 0.790, 0.801, 0.809, 0.825, 0.858,
                        0.334, 0.383, 0.441, 0.496, 0.545, 0.551, 0.575, 0.606, 0.636, 0.641, 0.649, 0.657, 0.671]

    point0_stable = [0.192, 0.201, 0.210, 0.221, 0.235, 0.250, 0.274, 0.301, 0.337, 0.384, 0.440, 0.499, 0.546, 0.581]
    ld0_stable = lD[10:24]
    point0_unstable = [0.146, 0.151, 0.153, 0.156, 0.158, 0.161, 0.163, 0.166, 0.174, 0.181]
    ld0_unstable = lD[0:10]

    point1_stable = [0.526, 0.529, 0.529, 0.530, 0.533, 0.535, 0.536, 0.538, 0.539, 0.544]
    ld1_stable = lD[0:10]
    point1_unstable = [0.543, 0.550, 0.553, 0.557, 0.565, 0.574, 0.587, 0.624, 0.658, 0.700, 0.751, 0.802, 0.849, 0.881]
    ld1_unstable = lD[10:24]

    point2_stable = [0.758, 0.767, 0.778, 0.783, 0.794, 0.809, 0.827, 0.850, 0.881, 0.924, 0.975, 1.029, 1.070, 1.106]
    ld2_stable = lD[10:24]
    point2_unstable = [0.743, 0.740, 0.743, 0.745, 0.747, 0.745, 0.749, 0.750, 0.750, 0.751]
    ld2_unstable = lD[0:10]

    fig, ax = plt.subplots()
    # ax.scatter(ld0_stable, point0_stable, color='royalblue')
    # ax.scatter(ld0_unstable, point0_unstable, color='orangered')
    ax.scatter(ld1_stable, point1_stable, color='royalblue')
    ax.scatter(ld1_unstable, point1_unstable, color='orangered')
    ax.scatter(ld2_stable, point2_stable, color='royalblue', marker='^')
    ax.scatter(ld2_unstable, point2_unstable, color='orangered', marker='^')

    tp=complex
    table1 = np.loadtxt("final_exp/spectrum_mono_aerogelPW_D2_f.csv", delimiter=",", dtype=str)
    ld = table1[1:, 0].astype(tp)[105:145]
    t01 = table1[1:, 2].astype(tp)[105:145]
    table2 = np.loadtxt("final_exp/spectrum_mono_aerogelPW_D206_f.csv", delimiter=",", dtype=str)
    ld2 = table2[1:, 0].astype(tp)[135:165]
    t02 = table2[1:, 2].astype(tp)[135:165]
    ax.plot(ld, np.abs(t01), color='grey', linestyle='--', linewidth=2)
    ax.plot(ld2, np.abs(t02), color='grey', linestyle='--', linewidth=2)

    plt.show()


dream_dream()
