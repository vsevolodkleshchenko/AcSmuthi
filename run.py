from applications import examples

simulation = examples.silica_aerogel_sphere_in_standing_wave()
simulation.run(cross_sections_flag=False, forces_flag=True, plot_flag=True)
