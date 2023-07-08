class Medium:
    def __init__(
            self,
            density: float,
            pressure_velocity: float,
            is_substrate: bool = False
    ):
        self.density = density
        self.cp = pressure_velocity
        self.is_substrate = is_substrate

