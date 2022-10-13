import numpy as np
import time
from acsmuthi.postprocessing import forces, cross_sections as cs, fields, rendering
from acsmuthi.linear_system import LinearSystem


class Simulation:
    def __init__(self, particles, medium, frequency, order,
                 bound=None, number_of_points=None, plane=None, plane_number=None, layer=None):
        self.particles = particles
        self.layer = layer
        self.medium = medium
        self.freq = frequency
        self.order = order
        self.bound = bound
        self.number_of_points = number_of_points
        self.plane = plane
        self.plane_number = plane_number

    def run(self, cross_sections_flag=False, forces_flag=False, plot_flag=False):
        linear_system = LinearSystem(self.particles, self.layer, self.medium, self.freq, self.order)
        t_start = time.process_time()
        t_cs_start = t_cs_finish = t_f_start = t_f_finish = t_sf_start = t_sf_finish = 0
        linear_system.solve()
        t_solution = time.process_time()
        if cross_sections_flag:
            t_cs_start = time.process_time()
            print("Scattering and extinction cs:", *cs.cross_section(self.particles, self.medium, self.freq,
                                                                     self.order, self.layer))
            t_cs_finish = time.process_time()
        if forces_flag:
            t_f_start = time.process_time()
            print("Forces:\n", forces.all_forces(self.particles, self.medium))
            t_f_finish = time.process_time()
        if plot_flag:
            t_sf_start = time.process_time()
            span = rendering.build_discretized_span(self.bound, self.number_of_points)
            x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, self.plane_number, plane=self.plane)
            tot_field = np.real(fields.compute_total_field(self.freq, self.medium, self.particles, x_p, y_p, z_p,
                                                           layer=self.layer))
            rendering.slice_plot(tot_field, span_v, span_h, plane=self.plane)
            t_sf_finish = time.process_time()
        print("Solving the system:", t_solution - t_start, "s")
        print("Counting cross sections:", t_cs_finish - t_cs_start, "s")
        print("Counting forces:", t_f_finish - t_f_start, "s")
        print("Counting fields and drawing the slice plot:", t_sf_finish - t_sf_start, "s")
        print("All process:", time.process_time() - t_start, "s")
