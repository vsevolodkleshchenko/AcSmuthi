import cross_sections as pp
import numpy as np
import tsystem
import classes as cls
import csv
from multiprocessing import Process, Queue


def freq_proc(ps, order, freq_result):
    solution = tsystem.solve_system(ps, order)
    scs, ecs = pp.cross_section(solution, ps, order)
    freq_result.put((scs, ecs))


def spectrum_freq_proc(freqs, ps, order, spec_result):
    scs = np.zeros(len(freqs), dtype=float)
    ecs = np.zeros(len(freqs), dtype=float)
    for i, freq in zip(range(len(freqs)), freqs):
        ps.incident_field.freq = freq
        queue_res = Queue()
        freq_process = Process(target=freq_proc, args=(ps, order, queue_res,))
        freq_process.start()
        scs[i], ecs[i] = queue_res.get()
        freq_process.join()
    spec_result.put(scs)


def spectrum_dist_freq(poses, freqs, ps, order):
    table = np.zeros((len(freqs), len(poses)))
    for j, pos2 in zip(range(len(poses)), poses):
        ps.spheres[0].pos = pos2[0]
        ps.spheres[1].pos = pos2[1]
        queue_res = Queue()
        dist_process = Process(target=spectrum_freq_proc, args=(freqs, ps, order, queue_res,))
        dist_process.start()
        table[:, j] = queue_res.get()
        dist_process.join()
    return table


def write_csv(data, fieldnames, filename):
    with open(".\\outfiles\\"+filename+".csv", mode="w", encoding='utf-8') as w_file:
        file_writer = csv.writer(w_file, delimiter=",", lineterminator="\r")
        file_writer.writerow(fieldnames)
        file_writer.writerows(data)


if __name__ == '__main__':
    physical_system = cls.build_ps_2s()
    frequencies = np.linspace(6, 160, 78, dtype=float)
    coord_z = np.linspace(1, 3.5, 11)
    positions = np.zeros((len(coord_z), 2, 3))
    for j, z in zip(range(len(coord_z)), coord_z):
        pos1 = np.array([0, 0, z])
        positions[j] = [-pos1, pos1]
    spectrum_table = np.zeros((len(frequencies), len(positions) + 1))
    spectrum_table[:, 0] = frequencies
    spectrum_table[:, 1:] = spectrum_dist_freq(positions, frequencies, physical_system, 9)
    header = ["freqs"]+list(coord_z)
    write_csv(spectrum_table, header, "spectrum_dist_freq")
