import numpy as np
import scipy.special as ss


def legendres_table(z, order):
    legs_positive_m = np.array([ss.clpmn(order, order, zi, type=2)[0] for zi in z])
    legs_negative_m = np.array([ss.clpmn(-order, order, zi, type=2)[0] for zi in z])
    return np.moveaxis(legs_positive_m, 0, -1), np.moveaxis(legs_negative_m, 0, -1)
