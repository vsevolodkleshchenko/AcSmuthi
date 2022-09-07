import examples


if __name__ == '__main__':
    simulation = examples.one_elastic_sphere()
    simulation.run(cross_sections_flag=True, forces_flag=True, plot_flag=False)
