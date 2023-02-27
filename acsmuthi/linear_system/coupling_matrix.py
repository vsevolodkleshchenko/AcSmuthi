
try:
    from acsmuthi.utility.cython_opt import cython_speedups as cysp

    def coupling_block(particle_pos, other_particle_pos, k_medium, order):
        return cysp.coupling_block(particle_pos, other_particle_pos, k_medium, order)


    def translation_block(order, k_medium, distance):
        return cysp.translation_block(order, k_medium, distance)


except:
    import numpy as np

    import acsmuthi.utility.wavefunctions as wvfs

    print("Failed")


    def coupling_block(particle_pos, other_particle_pos, k_medium, order):
        block = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            for mu, nu in wvfs.multipoles(order):
                imunu = nu ** 2 + nu + mu
                distance = particle_pos - other_particle_pos
                block[imn, imunu] = - wvfs.outgoing_separation_coefficient(mu, m, nu, n, k_medium, distance)
        return block


    def translation_block(order, k_medium, distance):
        d = np.zeros(((order + 1) ** 2, (order + 1) ** 2), dtype=complex)
        for m, n in wvfs.multipoles(order):
            imn = n ** 2 + n + m
            for mu, nu, in wvfs.multipoles(order):
                imunu = nu ** 2 + nu + mu
                d[imn, imunu] = wvfs.regular_separation_coefficient(mu, m, nu, n, k_medium, distance)
        return d
