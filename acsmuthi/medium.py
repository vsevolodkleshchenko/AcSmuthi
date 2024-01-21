import numpy as np


class Medium:
    def __init__(
            self,
            density: float,
            pressure_velocity: float,
            hard_substrate: bool = False,
            substrate_density: float | None = None,
            substrate_velocity: float | None = None,
            substrate_velocity_shear: float | None = None
    ):
        self.density = density
        self.cp = pressure_velocity
        if hard_substrate or (substrate_velocity is not None and substrate_density is not None):
            self.is_substrate = True
        else:
            self.is_substrate = False
        self.hard_substrate = hard_substrate
        self.density_sub = substrate_density
        self.cp_sub = substrate_velocity
        self.cs_sub = substrate_velocity_shear

    def k_substrate(self, k_medium):
        omega = k_medium * self.cp
        if not self.is_substrate:
            return None
        elif self.hard_substrate:
            return None
        elif self.cs_sub is None:
            return np.array([omega / self.cp_sub])
        else:
            return np.array([omega / self.cp_sub, omega / self.cs_sub])
