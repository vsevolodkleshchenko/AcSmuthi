import numpy as np
import time
from acsmuthi.postprocessing import forces, cross_sections as cs, fields, rendering
from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import InitialField


class Simulation:
    def __init__(self,
                 particles: np.ndarray[Particle],
                 medium: Medium,
                 initial_field: InitialField,
                 frequency: float,
                 order: int,
                 store_t_matrix: bool = False,
                 bound: float = None,
                 number_of_points: int = None,
                 plane: str = None,
                 plane_number: int = None):
        self.particles = particles
        self.medium = medium
        self.freq = frequency
        self.order = order
        self.bound = bound
        self.number_of_points = number_of_points
        self.plane = plane
        self.plane_number = plane_number
        self.incident_field = initial_field
        self.store_t_matrix = store_t_matrix

    def run(self, cross_sections_flag=False, forces_flag=False, plot_flag=False):
        linear_system = LinearSystem(particles=self.particles, medium=self.medium, initial_field=self.incident_field,
                                     frequency=self.freq, order=self.order, store_t_matrix=self.store_t_matrix)
        t_start = time.time()
        t_cs_start = t_cs_finish = t_f_start = t_f_finish = t_sf_start = t_sf_finish = 0
        linear_system.prepare()
        t_preparation = time.time()
        linear_system.solve()
        t_solution = time.time()
        if cross_sections_flag:
            t_cs_start = time.time()
            print("Scattering and extinction cs:", cs.extinction_cs(particles=self.particles,
                                                                    medium=self.medium,
                                                                    incident_field=self.incident_field,
                                                                    freq=self.freq))
            t_cs_finish = time.time()
        if forces_flag:
            t_f_start = time.time()
            print("Forces:\n", forces.all_forces(particles_array=self.particles,
                                                 medium=self.medium,
                                                 incident_field=self.incident_field))
            t_f_finish = time.time()
        if plot_flag:
            t_sf_start = time.time()
            span = rendering.build_discretized_span(bound=self.bound,
                                                    number_of_points=self.number_of_points)
            x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span=span,
                                                                  plane_number=self.plane_number,
                                                                  plane=self.plane)
            tot_field = np.abs(fields.compute_total_field(particles=self.particles,
                                                          incident_field=self.incident_field,
                                                          x=x_p, y=y_p, z=z_p)) ** 2
            rendering.slice_plot(tot_field=tot_field,
                                 span_v=span_v,
                                 span_h=span_h,
                                 plane=self.plane)
            t_sf_finish = time.time()
        print("Preparation of the system:", round(t_preparation - t_start, 3), "s")
        print("Solving the system:", round(t_solution - t_preparation, 3), "s")
        print("Counting cross sections:", round(t_cs_finish - t_cs_start, 3), "s")
        print("Counting forces:", round(t_f_finish - t_f_start, 3), "s")
        print("Counting fields and drawing the slice plot:", round(t_sf_finish - t_sf_start, 3), "s")
        print("All process:", round(time.time() - t_start, 3), "s")
