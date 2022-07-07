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


def old_total_field(x, y, z, k, ro, pos, spheres, order):
    r""" counts field outside the spheres """
    coef = syst_solve(k, ro, pos, spheres, order)
    tot_field_array = np.zeros((len(spheres), (order + 1) ** 2, len(x)), dtype=complex)
    tot_field = np.zeros(len(x), dtype=complex)
    for n in range(order + 1):
        for m in range(-n, n + 1):
            for sph in range(len(spheres)):
                tot_field += coef[2 * sph, n ** 2 + n + m] * \
                             outgoing_wvfs(m, n, x - pos[sph][0], y - pos[sph][1], z - pos[sph][2], k)
            # tot_field += inc_coef(m, n, k) * regular_wvfs(m, n, x, y, z, k)
                # tot_field += local_inc_coef(m, n, k) * regular_wvfs(m, n, x, y, z, k)
    return tot_field # np.abs(tot_field - tot_field1)


def old_gaunt_coef(n, m, nu, mu, q):
    r""" Gaunt coefficient: G(n,m;nu,mu;q)
    eq(3.71) in 'Encyclopedia' """
    s = np.sqrt((2 * n + 1) * (2 * nu + 1) * (2 * q + 1) / 4 / np.pi)
    return (-1) ** (m + mu) * s * float(wigner_3j(n, nu, q, 0, 0, 0)) * \
           float(wigner_3j(n, nu, q, m, mu, - m - mu))


def old_local_inc_coef(m, n, k, sph_pos, order):
    r""" Counts local incident coefficients
    d^m_nj - eq(42) in 'Multiple scattering and scattering cross sections P. A. Martin' """
    inccoef = 0
    for nu in range(order + 1):
        for mu in range(-nu, nu + 1):
            inccoef += inc_coef(mu, nu, k) * sepc_matr_coef(mu, m, nu, n, k, sph_pos)
    return inccoef


def old_sepc_matr_coef(m, mu, n, nu, k, dist):
    r""" Coefficient ^S^mmu_nnu(b) of separation matrix
    eq(3.92) and eq(3.74) in 'Encyclopedia' """
    if abs(n - nu) >= abs(m - mu):
        q0 = abs(n - nu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 == 0):
        q0 = abs(m - mu)
    if (abs(n - nu) < abs(m - mu)) and ((n + nu + abs(m - mu)) % 2 != 0):
        q0 = abs(m - mu) + 1
    q_lim = (n + nu - q0) // 2
    sum = 0
    for q in range(0, q_lim + 1, 2):
        sum += (-1) ** q * regular_wvfs(m - mu, q0 + 2 * q, dist[0], dist[1], dist[2], k) * \
               gaunt_coef(n, m, nu, -mu, q0 + 2 * q)
    return 4 * np.pi * (-1) ** (mu + nu + q_lim) * sum