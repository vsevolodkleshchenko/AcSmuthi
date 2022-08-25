import wavefunctions as wvfs
import tsystem
import numpy as np
import classes as cls
import time
import rendering
import cross_sections as cs
import fields
import forces

# class SphericalWaveExpansion:
#     def __init__(self, amplitude, kind, order):
#         self.ampl = amplitude
#         self.coefficients = None
#         self.kind = kind  # 'regular' or 'outgoing'
#         self.order = order
#
#     def pressure_field(self):
#


class Layer:
    def __init__(self, density, speed_of_sound, a, b, c, d):
        self.rho = density
        self.speed = speed_of_sound
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.reflected_field_exp = None

    def int_dist(self, position):
        r"""Absolute distance between position and interface"""
        return np.abs(self.a * position[0] + self.b * position[1] + self.c * position[2] + self.d) / \
               np.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2)

    @property
    def int_dist0(self):
        r"""Absolute distance between coordinate's origin and interface"""
        return self.int_dist(np.array([0., 0., 0.]))

    @property
    def normal(self):
        r"""Normal unit vector to interface with direction to coordinate's origin"""
        n = np.array(np.array([self.a, self.b, self.c]) / np.sqrt(self.a ** 2 + self.b ** 2 + self.c ** 2))
        n_dist = n * self.int_dist0
        if n_dist[0] * self.a + n_dist[1] * self.b + n_dist[2] * self.c == -self.d:
            n *= -1
        return n


class Particle:
    def __init__(self, position, radius, density, speed_of_sound, order):
        self.pos = position
        self.r = radius
        self.rho = density
        self.speed = speed_of_sound
        self.incident_field_ex = None
        self.scattered_field_ex = None
        self.inner_coefficients = None
        self.reflected_field_ex = None
        self.t_matrix = None
        self.order = order

    def t_matrix(self, sph_number, ps):
        t = np.zeros(((self.order+1)**2, (self.order+1)**2), dtype=complex)
        for i, n, m in enumerate(wvfs.multipoles(self.order)):
            t[i, i] = tsystem.scaled_coefficient(n, sph_number, ps)
        self.t_matrix = t

    def incident_field_decomposition(self, ps):
        d = np.zeros((self.order+1)**2, dtype=complex)
        for i, n, m in enumerate(wvfs.multipoles(self.order)):
            d[i] = wvfs.local_incident_coefficient(m, n, ps.k, ps.incident_field.dir, self.pos, self.order)
        self.incident_field_ex = d


class LinearSystem:
    def __init__(self, physical_system, order):
        self.ps = physical_system
        self.order = order
        self.solution_coefficients = None

    # def compute_t_matrix(self):
    #     self.t_matrix = np.linalg.inv(tsystem.system_matrix(self.ps, self.order))
    #
    # def compute_d_matrix(self):
    #     self.d_matrix = tsystem.d_matrix(self.ps, self.order)
    #
    # def compute_r_matrix(self):
    #     self.r_matix = tsystem.r_matrix(self.ps, self.order)
    #
    # def right_hand_side(self):
    #     self.rhs = tsystem.system_rhs(self.ps, self.order)

    def solve(self):
        if self.ps.interface:
            sol_coefs = tsystem.solve_layer_system(self.ps, self.order)
            incident_coefs, scattered_coefs, inner_coefs, reflected_coefs, local_reflected_coefs = sol_coefs
            for s in range(self.ps.num_sph):
                self.ps.spheres[s].incident_field_ex = incident_coefs[s]
                self.ps.spheres[s].scattered_field_ex = scattered_coefs[s]
                self.ps.spheres[s].inner_field_ex = inner_coefs[s]
                self.ps.spheres[s].reflected_field_ex = local_reflected_coefs[s]
            self.ps.interface.reflected_field_ex = reflected_coefs
        else:
            sol_coefs = tsystem.solve_system(self.ps, self.order)
            incident_coefs, scattered_coefs, inner_coefs = sol_coefs
            for s in range(self.ps.num_sph):
                self.ps.spheres[s].incident_field_ex = incident_coefs[s]
                self.ps.spheres[s].scattered_field_ex = scattered_coefs[s]
                self.ps.spheres[s].inner_field_ex = inner_coefs[s]
        self.solution_coefficients = sol_coefs


def build_new_ps_2s():
    r"""Builds physical system with 2 spheres"""
    # parameters of incident field
    direction = np.array([0.70711, 0, 0.70711])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = cls.PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = cls.Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2])
    pos2 = np.array([0, 0, 2])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 6

    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    spheres = np.array([sphere1, sphere2])

    ps = cls.System(fluid, incident_field, spheres)
    return ps


def build_new_ps_2s_l():
    r"""Builds physical system with 2 spheres and 1 interface(layer)"""
    # parameters of incident field
    direction = np.array([-1, 0, 0])
    freq = 82  # [Hz]
    p0 = 1  # [kg/m/s^2] = [Pa]
    incident_field = cls.PlaneWave(direction, freq, p0)

    # parameters of fluid
    ro_fluid = 1.225  # [kg/m^3]
    c_fluid = 331  # [m/s]
    fluid = cls.Fluid(ro_fluid, c_fluid)

    # parameters of the spheres
    pos1 = np.array([0, 0, -2.5])
    pos2 = np.array([0, 0, 2.5])
    r_sph = 1  # [m]
    ro_sph = 1050  # [kg/m^3]
    c_sph = 1403  # [m/s]

    order = 6

    sphere1 = Particle(pos1, r_sph, ro_sph, c_sph, order)
    sphere2 = Particle(pos2, r_sph, ro_sph, c_sph, order)
    spheres = np.array([sphere1, sphere2])

    # parameters of interface (substrate)
    a, b, c, d = 1, 0, 0, 2
    ro_interface = ro_sph
    c_interface = c_sph
    interface = Layer(ro_interface, c_interface, a, b, c, d)

    ps = cls.System(fluid, incident_field, spheres, interface)
    return ps


def new_compute(physical_system, order, cross_sections_on=False, forces_on=False, slice_field_on=False,
            bound=None, number_of_points=None, plane=None, plane_number=None):
    """Function that present the results and timing"""
    linear_system = LinearSystem(physical_system, order)
    t_start = time.process_time()
    t_cs_start = t_cs_finish = t_f_start = t_f_finish = t_sf_start = t_sf_finish = 0
    linear_system.solve()
    solution_coefficients = linear_system.solution_coefficients
    t_solution = time.process_time()
    if cross_sections_on:
        t_cs_start = time.process_time()
        print("Scattering and extinction cs:", *cs.cross_section(solution_coefficients, physical_system, order))
        t_cs_finish = time.process_time()
    if forces_on:
        t_f_start = time.process_time()
        print("Forces:\n", forces.all_forces(solution_coefficients, physical_system, order))
        t_f_finish = time.process_time()
    if slice_field_on:
        t_sf_start = time.process_time()
        span = rendering.build_discretized_span(bound, number_of_points)
        x_p, y_p, z_p, span_v, span_h = rendering.build_slice(span, plane_number, plane=plane)
        tot_field = np.real(
            fields.total_field(x_p, y_p, z_p, solution_coefficients, physical_system, order, incident_field_on=True))
        rendering.slice_plot(tot_field, x_p, y_p, z_p, span_v, span_h, physical_system, plane=plane)
        t_sf_finish = time.process_time()
    print("Solving the system:", t_solution - t_start, "s")
    print("Counting cross sections:", t_cs_finish - t_cs_start, "s")
    print("Counting forces:", t_f_finish - t_f_start, "s")
    print("Counting fields and drawing the slice plot:", t_sf_finish - t_sf_start, "s")
    print("All process:", time.process_time() - t_start, "s")


def simulation():
    r"""Main simulation function that start computations"""
    bound, number_of_points = 6, 301

    physical_system = build_new_ps_2s()

    order = 6

    plane = 'xz'
    plane_number = int(number_of_points / 2) + 1

    # compute(physical_system, order, cross_sections_on=True, forces_on=True)
    new_compute(physical_system, order, cross_sections_on=True, forces_on=True, slice_field_on=True, bound=bound,
            number_of_points=number_of_points, plane=plane, plane_number=plane_number)


simulation()
