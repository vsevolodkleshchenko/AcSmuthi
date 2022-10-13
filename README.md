﻿# AcSmuthi
This is the package for modelling of acoustic scattering on 
particles near the interface. It solves the Helmholtz equation with finding
the spherical waves expansion coefficients in all the system using the 
T-matrix method and calculates all the variables using this coefficients.


## How to use?
The main example of using the package is shown in `run.py`. To do that:
```commandline
poetry run python run.py
```


## How it works?
Before getting the results the `Simulation` must be created, that means
to define the `Particle`s array, `Medium` with 
incident field, it's frequency, determine the presence of 
`Layer`, choose the multipole order and specify the plot 
parameters (if necessary). 

Run the created simulation to get all the results with parameters:
```
simulation.run(cross_sections_flag=True, forces_flag=True, plot_flag=True)
```


## What can it do?
* Compute and render pressure near-fields
* Calculate scattering and extinction cross-sections
* Calculate forces acting on particles

## What realized now?
* Plane wave scattering on spherical fluid and elastic particles in fluid medium


