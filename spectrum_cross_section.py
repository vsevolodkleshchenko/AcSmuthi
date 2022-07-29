import postprocessing as pp
import numpy as np
import tsystem
import classes as cls
import matplotlib.pyplot as plt
import csv


def spectrum_freq(freqs, ps, order):
    scs = np.zeros(len(freqs), dtype=float)
    ecs = np.zeros(len(freqs), dtype=float)
    for i, freq in zip(range(len(freqs)), freqs):
        ps.incident_field.freq = freq
        scs[i], ecs[i] = pp.cross_section(tsystem.solve_system(ps, order), ps, order)
        print(scs[i], ecs[i])
    return scs, ecs


def spectrum_dist_freq(poses, freqs, ps, order):
    table = np.zeros((len(freqs), len(poses)))
    for j, pos2 in zip(range(len(poses)), poses):
        ps.spheres[0].pos = pos2[0]
        ps.spheres[1].pos = pos2[1]
        table[:, j] = spectrum_freq(freqs, ps, order)[0]
    return table


def spectrum_cs_plot(freqs, scs, ecs):
    fig1, ax = plt.subplots(1, 2)
    ax[0].plot(freqs, scs)
    ax[1].plot(freqs, ecs)
    ax[0].set(title="scs")
    ax[1].set(title="ecs")
    fig2, ax2 = plt.subplots()
    ax2.plot(freqs, scs)
    ax2.scatter(freqs, scs)
    ax2.set(title="Scattering/extinction cross section", xlabel="Frequencies, Hz", ylabel="Cross section, м2")
    plt.show()


def write_csv(data, fieldnames, filename):
    with open(".\\outfiles\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


physical_system = cls.build_ps()
frequencies = np.linspace(6, 160, 78, dtype=float)
coord_z = np.linspace(1, 3.5, 11)
positions = np.zeros((len(coord_z), 2, 3))
for j, z in zip(range(len(coord_z)), coord_z):
    pos1 = np.array([0, 0, z])
    pos2 = [-pos1, pos1]
    positions[j] = pos2
print(coord_z, positions, frequencies)
# spectrum_table = np.zeros((len(frequencies), len(positions) + 1))
# spectrum_table[:, 0] = frequencies
# spectrum_table[:, 1:] = spectrum_dist_freq(positions, frequencies, physical_system, 4)
# header = ["freqs"]+list(coord_z)
# write_csv(spectrum_table, header, "spectrum_dist_freq")


