import numpy as np
import time
from acsmuthi.postprocessing import forces, cross_sections as cs, fields, rendering
from acsmuthi.linear_system.linear_system import LinearSystem


class Simulation:
    def __init__(self, particles, medium, initial_field, frequency, order,
                 bound=None, number_of_points=None, plane=None, plane_number=None):
        self.particles = particles
        self.medium = medium
        self.freq = frequency
        self.order = order
        self.bound = bound
        self.number_of_points = number_of_points
        self.plane = plane
        self.plane_number = plane_number
        self.incident_field = initial_field

    def run(self, cross_sections_flag=False, forces_flag=False, plot_flag=False):
        linear_system = LinearSystem(self.particles, self.medium, self.incident_field, self.freq, self.order)
        t_start = time.time()
        t_cs_start = t_cs_finish = t_f_start = t_f_finish = t_sf_start = t_sf_finish = 0
        linear_system.solve()
        t_solution = time.time()
        if cross_sections_flag:
            t_cs_start = time.time()
            print("Scattering and extinction cs:", *cs.cross_section(self.particles, self.medium, self.incident_field,
                                                                     self.freq, self.order))
            t_cs_finish = time.time()
        if forces_flag:
            t_f_start = time.time()
            print("Forces:\n", forces.all_forces(self.particles, self.medium, self.incident_field))
            t_f_finish = time.time()
        if plot_flag:
            t_sf_start = time.time()
            span = rendering.build_discretized_span(self.bound, self.number_of_points)
            x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, self.plane_number, plane=self.plane)
            tot_field = np.abs(fields.compute_total_field(self.particles, self.incident_field, x_p, y_p, z_p)) ** 2
            rendering.slice_plot(tot_field, span_v, span_h, plane=self.plane)
            t_sf_finish = time.time()
        print("Solving the system:", round(t_solution - t_start, 2), "s")
        print("Counting cross sections:", round(t_cs_finish - t_cs_start, 2), "s")
        print("Counting forces:", round(t_f_finish - t_f_start, 2), "s")
        print("Counting fields and drawing the slice plot:", round(t_sf_finish - t_sf_start, 2), "s")
        print("All process:", round(time.time() - t_start, 2), "s")
