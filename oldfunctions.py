def yz_old(span, plane_number, k, ro, pos, spheres, order):
    r"""
    OLD 2D heat-plot in YZ plane for x[plane_number]
    --->z """
    span_x, span_y, span_z = span[0], span[1], span[2]
    grid = np.vstack(np.meshgrid(span_x, span_y, span_z, indexing='ij')).reshape(3, -1).T
    x, y, z = grid[:, 0], grid[:, 1], grid[:, 2]

    tot_field = np.real(total_field(x, y, z, k, ro, pos, spheres, order))

    for sph in range(len(spheres)):
        x_min = pos[sph, 0] - spheres[sph, 1]
        y_min = pos[sph, 1] - spheres[sph, 1]
        z_min = pos[sph, 2] - spheres[sph, 1]
        x_max = pos[sph, 0] + spheres[sph, 1]
        y_max = pos[sph, 1] + spheres[sph, 1]
        z_max = pos[sph, 2] + spheres[sph, 1]
        tot_field = np.where((x >= x_min) & (x <= x_max) &
                             (y >= y_min) & (y <= y_max) &
                             (z >= z_min) & (z <= z_max), 0, tot_field)

    yz = np.asarray(tot_field[(plane_number - 1) * len(span_y) * len(span_z):
                              (plane_number - 1) * len(span_y) * len(span_z) +
                              len(span_y) * len(span_z)]).reshape(len(span_y), len(span_z))
    fig, ax = plt.subplots()
    plt.xlabel('z axis')
    plt.ylabel('y axis')
    im = ax.imshow(yz, cmap='seismic', origin='lower',
                   extent=[span_z.min(), span_z.max(), span_y.min(), span_y.max()])
    plt.colorbar(im)
    plt.show()


def syst_rhs_old(k, spheres, order):
    r""" build right hand side of system from spheres.pdf"""
    k_abs = dec_to_sph(k[0], k[1], k[2])[0]
    num_of_sph = len(spheres)
    num_of_coef = (order + 1) ** 2
    rhs = np.zeros(num_of_coef * 2 * num_of_sph, dtype=complex)
    for n in range(order + 1):
        idx_1 = np.arange(2 * n ** 2, 2 * (n + 1) ** 2 - 1, 2)
        idx_2 = np.arange(2 * n ** 2 + 1, 2 * (n + 1) ** 2, 2)
        m = np.arange(-n, n + 1)
        for sph in range(num_of_sph):
            rhs[idx_1] = inc_coef(m, n, k) * \
                       scipy.special.spherical_jn(n, k_abs * spheres[sph, 1])
            rhs[idx_2] = inc_coef(m, n, k) * \
                       scipy.special.spherical_jn(n, k_abs * spheres[sph, 1], derivative=True)
            idx_1 += num_of_coef * 2
            idx_2 += num_of_coef * 2
    return rhs