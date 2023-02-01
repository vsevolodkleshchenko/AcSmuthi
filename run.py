from applications import examples

simulation = examples.two_silica_aerogel_sphere_in_standing_wave()
simulation.run(cross_sections_flag=True, forces_flag=True, plot_flag=True)
