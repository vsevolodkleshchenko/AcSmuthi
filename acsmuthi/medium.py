class Medium:
    def __init__(self, density, speed_of_sound, incident_field=None):
        self.rho = density
        self.speed_l = speed_of_sound
        self.incident_field = incident_field
        self.scattered_field = None
        self.reflected_field = None

