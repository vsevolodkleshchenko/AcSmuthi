from applications import examples

simulation = examples.one_fluid_sphere_above_interface()
simulation.run(cross_sections_flag=True, forces_flag=True, plot_flag=True)
