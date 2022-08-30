import examples


if __name__ == '__main__':
    sim = examples.one_elastic_sphere()
    sim.run(cross_sections_flag=True, forces_flag=True, plot_flag=True)
