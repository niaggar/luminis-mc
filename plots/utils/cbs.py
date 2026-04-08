import numpy as np
from scipy.ndimage import uniform_filter1d

_FWHM_Q = 0.73

def cbs_fwhm(theta_mrad: np.ndarray, enhancement: np.ndarray) -> float:
    """
    Calcula el FWHM del pico CBS por interpolación lineal a media altura.
 
    El fondo se estima como la mediana del 20% de los bordes del array.
    El pico se ubica en el argmax. Los cruces a media altura se buscan
    hacia afuera desde el pico — robusto aunque el pico esté cerca del borde.
 
    Retorna el FWHM en mrad.
    """
    theta = np.asarray(theta_mrad, dtype=float).ravel()
    y     = np.asarray(enhancement, dtype=float).ravel()
 
    if theta.shape != y.shape:
        raise ValueError(
            f"theta_mrad y enhancement deben tener la misma forma. "
            f"Recibido: theta={theta.shape}, enhancement={y.shape}."
        )
    if len(y) < 5:
        raise ValueError(
            f"enhancement tiene solo {len(y)} elementos "
            f"(shape original: {np.asarray(enhancement).shape}). "
            f"Se necesitan al menos 5."
        )
 
    # Fondo: mediana de los bordes (primer y último 10%)
    n_edge     = max(3, len(y) // 10)
    background = np.median(np.concatenate([y[:n_edge], y[-n_edge:]]))
 
    # Perfil CBS sin fondo y media altura
    cbs      = y - background
    peak_idx = int(np.argmax(cbs))
    half_max = cbs[peak_idx] / 2.0
 
    # Cruce izquierdo: desde el pico hacia el borde izquierdo
    left = None
    for i in range(peak_idx, 0, -1):
        if cbs[i - 1] <= half_max:
            denom = cbs[i] - cbs[i - 1]
            if abs(denom) > 1e-30:
                t    = (half_max - cbs[i - 1]) / denom
                left = theta[i - 1] + t * (theta[i] - theta[i - 1])
            break
 
    # Cruce derecho: desde el pico hacia el borde derecho
    right = None
    for i in range(peak_idx, len(cbs) - 1):
        if cbs[i + 1] <= half_max:
            denom = cbs[i] - cbs[i + 1]
            if abs(denom) > 1e-30:
                t     = (cbs[i] - half_max) / denom
                right = theta[i] + t * (theta[i + 1] - theta[i])
            break
 
    if left is None or right is None:
        missing = []
        if left  is None: missing.append("izquierdo")
        if right is None: missing.append("derecho")
        raise ValueError(
            f"No se encontró el cruce a media altura en el lado {' y '.join(missing)}. "
            f"Pico en theta={theta[peak_idx]:.3f} mrad (índice {peak_idx}/{len(theta)-1}). "
            f"Asegúrate de cubrir suficiente rango angular a ambos lados del pico."
        )
 
    return float(right - left)
 
def _half_crossing(theta_seg, cbs_seg, half_max, side):
    """Interpolación lineal del cruce con `half_max` en un segmento."""
    if side == "left":
        # Busca de derecha a izquierda (hacia el centro desde el borde izq.)
        for i in range(len(cbs_seg) - 1, 0, -1):
            if cbs_seg[i-1] <= half_max <= cbs_seg[i]:
                t = (half_max - cbs_seg[i-1]) / (cbs_seg[i] - cbs_seg[i-1])
                return theta_seg[i-1] + t * (theta_seg[i] - theta_seg[i-1])
    else:
        # Busca de izquierda a derecha (desde el centro hacia el borde der.)
        for i in range(len(cbs_seg) - 1):
            if cbs_seg[i] >= half_max >= cbs_seg[i+1]:
                t = (cbs_seg[i] - half_max) / (cbs_seg[i] - cbs_seg[i+1])
                return theta_seg[i] + t * (theta_seg[i+1] - theta_seg[i])
    return None
 
 
def fwhm_to_mfp(fwhm_mrad: float, wavelength_nm: float, n: float) -> float:
    """
    Convierte el FWHM del pico CBS al camino libre medio de transporte l*.
 
    Parámetros
    ----------
    fwhm_mrad     : FWHM del perfil CBS en mrad.
    wavelength_nm : longitud de onda [nm].
    n              : índice de refracción.
 
    Retorna
    -------
    l* en µm.
    """
    fwhm_rad = fwhm_mrad * 1e-3
    lam_m    = wavelength_nm * 1e-9
    k        = 2.0 * np.pi * n / lam_m
    l_m      = _FWHM_Q / (k * fwhm_rad)
    return l_m * 1e6   # µm


def deg_to_mrad(deg):
    return deg * (1000 * np.pi / 180)

def get_sym_slice(enhancement_2d, theta_deg, phi_deg, phi_cut, smooth_size=7):
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

