
C0 = 0.299792458   # um/fs

def build_time_grid(l_star, n_medium, n_bins=100, t_max_taustar=30, binning="geometric", dt_floor_fs=30.0):
    """Rejilla temporal parametrizada por FISICA (tau*), no por fs absolutos.

    l_star        : puede ser un float (una especie) o lista (toma la mas lenta).
    n_bins        : ~20-30. Resuelve el estrechamiento del cono (1-30 tau*).
    t_max_taustar : cuantos tau* cubrir.
    dt_floor_fs   : chequeo de cordura; avisa si dt cae bajo el piso experimental.
    """
    l_ref = max(l_star) if hasattr(l_star, "__iter__") else l_star
    tau_star_fs = l_ref * n_medium / C0

    t_max_fs = t_max_taustar * tau_star_fs
    dt_fs    = t_max_fs / n_bins                 # = (t_max_taustar/n_bins) * tau*

    if dt_fs < dt_floor_fs:
        print(f"[aviso] dt={dt_fs:.1f} fs < piso {dt_floor_fs} fs "
              f"(no medible con gate SHG; baja n_bins).")

    if binning == "optical":
        dt_sim, t_max_sim = dt_fs * C0, t_max_fs * C0
    else:
        dt_sim, t_max_sim = dt_fs * C0 / n_medium, t_max_fs * C0 / n_medium

    return {"dt_fs": dt_fs, "t_max_fs": t_max_fs, "n_bins": n_bins,
            "dt_sim": dt_sim, "t_max_sim": t_max_sim,
            "tau_star_fs": tau_star_fs, "l_star_ref": l_ref}

def depth_report(grid, l_star, n_medium):
    M = grid["t_max_fs"] / grid["tau_star_fs"]        # = t_max en tau*
    z_bal  = grid["t_max_sim"] / 2                     # um, retorno balistico
    z_diff = (2*M/3)**0.5 * l_star                     # um, difusivo
    print(f"M = {M:.0f} tau*")
    print(f"prof. balistica (retorno): {z_bal:.1f} um = {z_bal/l_star:.1f} l*")
    print(f"prof. difusiva sondeada  : {z_diff:.1f} um = {z_diff/l_star:.1f} l*")
    return z_diff
