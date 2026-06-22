import numpy as np

def principal_stress(spxx, spyy, spxy):
    left_part = (spxx - spyy) * 0.5
    stress_xy_diff = spxx - spyy
    stress_max = (spxx + spyy) * 0.5 + np.linalg.norm(spxy - left_part)
    stress_min = (spxx + spyy) * 0.5 - np.linalg.norm(spxy - left_part)
    stress_angle = (0.5 * np.degrees(np.arctan((2 * spxy) / stress_xy_diff)))

    return stress_max, stress_min, stress_angle