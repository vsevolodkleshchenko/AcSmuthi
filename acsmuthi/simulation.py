import numpy as np
import time

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
            solver: str = 'LU' # 'LU' or 'GMRES'
    ):
        self.particles = particles
        self.medium = medium
        self.freq = frequency
        self.order = order
        self.initial_field = initial_field
        self.solver = solver
        self.linear_system = None

    def run(self):
        self.linear_system = LinearSystem(
            particles=self.particles,
            medium=self.medium,
            initial_field=self.initial_field,
            frequency=self.freq,
            order=self.order,
            solver=self.solver
        )
        t_start = time.time()
        self.linear_system.prepare()
        t_preparation = time.time() - t_start
        t_start_solving = time.time()
        self.linear_system.solve()
        t_solution = time.time() - t_start_solving
        return t_preparation, t_solution
