import numpy as np
import time
from acsmuthi.postprocessing import forces, cross_sections as cs, fields, rendering
from acsmuthi.linear_system.linear_system import LinearSystem
from acsmuthi.particles import Particle
from acsmuthi.medium import Medium
from acsmuthi.initial_field import InitialField


class Simulation:
    def __init__(
            self,
            particles: np.ndarray[Particle],
            medium: Medium,
            initial_field: InitialField,
            frequency: float,
            order: int,
            store_t_matrix: bool = True
    ):
        self.particles = particles
        self.medium = medium
        self.freq = frequency
        self.order = order
        self.incident_field = initial_field
        self.store_t_matrix = store_t_matrix

    def run(self):
        linear_system = LinearSystem(
            particles=self.particles,
            medium=self.medium,
            initial_field=self.incident_field,
            frequency=self.freq,
            order=self.order,
            store_t_matrix=self.store_t_matrix
        )
        t_start = time.time()
        linear_system.prepare()
        t_preparation = time.time() - t_start
        t_start_solving = time.time()
        linear_system.solve()
        t_solution = time.time() - t_start_solving
        return t_preparation, t_solution
