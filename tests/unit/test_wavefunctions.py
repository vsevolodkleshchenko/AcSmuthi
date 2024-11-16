import numpy as np
from scipy.special import sph_harm

import acsmuthi.utility.wavefunctions as wvfs


def test_plane_wave_sfe_cfs():
    direction = np.array([np.sqrt(0.5), 0, -np.sqrt(0.5)])
    np.testing.assert_allclose(wvfs.plane_wave_sfe_cfs(direction, 3), np.array([
            3.5449077 +0.j        ,  0.        +3.06998012j,
            0.        -4.34160753j,  0.        -3.06998012j,
           -2.42703239+0.j        ,  4.85406478+0.j        ,
           -1.98166365+0.j        , -4.85406478+0.j        ,
           -2.42703239+0.j        ,  0.        -1.85367661j,
           -0.        +4.54056184j,  0.        -4.30755518j,
            0.        -1.65797876j, -0.        +4.30755518j,
           -0.        +4.54056184j, -0.        +1.85367661j
        ])
    )

