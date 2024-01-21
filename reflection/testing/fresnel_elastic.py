import matplotlib.pyplot as plt
import numpy as np
from acsmuthi.linear_system.coupling.coupling_basics import fresnel_elastic


def check_r_vs_angle():
    ang = np.linspace(3*np.pi/4, 11*np.pi/12, 1050)
    rho, cp, cs = 2780, 6300, 3100  # duralumin
    rho0, c0 = 1000, 1480
    f = 10000
    k0 = 2 * np.pi * f / c0
    kp = -np.sin(ang) * k0
    r = fresnel_elastic(kp, k0, c0, cp, cs, rho0, rho)
    points = np.array([2 * np.pi * f / cp, 2 * np.pi * f / cs, k0])
    print(points)
    r_points = fresnel_elastic(points, k0, c0, cp, cs, rho0, rho)
    plt.plot(ang, np.abs(r))
    # plt.scatter(points, np.abs(r_points))
    plt.show()