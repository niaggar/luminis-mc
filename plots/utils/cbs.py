import numpy as np
from scipy.ndimage import uniform_filter1d


def deg_to_mrad(deg):
    return deg * (1000 * np.pi / 180)

def get_sym_slice(enhancement_2d, theta_deg, phi_deg, phi_cut, smooth_size=5):
    """
    Returns (theta_mrad_sym, I_sym) — negative and positive angles,
    X or Y scan, smoothed with 5-point moving average.
    """
    theta_sym, I = get_slice(enhancement_2d, theta_deg, phi_deg, phi_cut=phi_cut)
    I_smooth = uniform_filter1d(I, size=smooth_size)
    return deg_to_mrad(theta_sym), I_smooth

def get_slice(I, theta_degrees, phi_degrees, phi_cut=0.0):
    theta = np.asarray(theta_degrees, float)
    phi   = np.asarray(phi_degrees,   float) % 360

    i_fwd = int(np.argmin(np.abs(phi - (phi_cut       % 360))))
    i_bwd = int(np.argmin(np.abs(phi - ((phi_cut + 180) % 360))))

    theta_sym = np.concatenate([-theta[::-1], theta])
    I_sym     = np.concatenate([np.asarray(I)[::-1, i_bwd], np.asarray(I)[:,    i_fwd]])
    return theta_sym, I_sym

def profile_anisotropy(theta_mrad, I_X, I_Y, angle_limit=None):
    if angle_limit is not None:
        mask = np.abs(theta_mrad) <= angle_limit
        theta_mrad = theta_mrad[mask]
        I_X = I_X[mask]
        I_Y = I_Y[mask]

    # only integrate the cone part (subtract background = 1)
    cone_X = I_X - 1.0
    cone_Y = I_Y - 1.0

    S_X = np.trapezoid(cone_X, theta_mrad)
    S_Y = np.trapezoid(cone_Y, theta_mrad)

    if (S_X + S_Y) == 0:
        return np.nan
    return (S_X - S_Y) / (S_X + S_Y)

