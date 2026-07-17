# plots/style.py
import matplotlib as mpl

TEXTWIDTH_IN = 6.33      # <-- el valor real de \the\textwidth / 72.27
DOC_FONTSIZE = 12        # 12pt del \documentclass
COL = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc", "#56463e"]

def apply(width_frac=1.0, aspect=1/1.618, fontsize=DOC_FONTSIZE):
    """
    width_frac: fracción de \textwidth que ocupará la figura (1.0, 0.5, ...)
    aspect:     alto/ancho
    fontsize:   igual al tamaño del documento
    Devuelve (w, h) por si quieres ajustar a mano.
    """
    w = TEXTWIDTH_IN * width_frac
    h = w * aspect
    mpl.rcParams.update({
        "text.usetex":     True,
        "font.family":     "serif",
        "font.serif":      ["Computer Modern Roman"],
        "axes.grid":       False,
        # TODOS los tamaños == documento
        "font.size":       fontsize,
        "axes.titlesize":  fontsize,
        "axes.labelsize":  fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "axes.linewidth":  0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "figure.figsize":  (w, h),
        "figure.dpi":      150,
        "savefig.dpi":     300,
        # constrained_layout en vez de tight_layout / bbox="tight"
        "figure.constrained_layout.use": True,
    })
    return w, h