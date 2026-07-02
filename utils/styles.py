# plots/style.py
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- Sizes (tweak once, used everywhere) ---
# IEEE/Optics Express single column: ~3.5 in, double: ~7.0 in
# Thesis (A4, 1.5cm margins): ~5.9 in full width
SINGLE_COL = 3.5   # inches
DOUBLE_COL = 7.0
THESIS_FULL = 5.9

def apply(context="paper", col="single"):
    """
    context: "paper" | "thesis"
    col:     "single" | "double" | "full"
    """
    width = {"single": SINGLE_COL, "double": DOUBLE_COL, "full": THESIS_FULL}[col]
    height = width / 1.5   # golden ratio-ish, override per figure

    mpl.rcParams.update({
        # Font
        "font.family":       "serif",
        "font.serif":        ["Computer Modern Roman"],  # matches LaTeX
        "text.usetex":       True,
        "font.size":         9 if context == "paper" else 11,
        "axes.titlesize":    9 if context == "paper" else 11,
        "axes.labelsize":    9 if context == "paper" else 11,
        "xtick.labelsize":   8 if context == "paper" else 10,
        "ytick.labelsize":   8 if context == "paper" else 10,
        "legend.fontsize":   8 if context == "paper" else 10,

        # Lines and axes
        "axes.linewidth":    0.8,
        "lines.linewidth":   1.2,
        "lines.markersize":  4,

        # Figure
        "figure.figsize":    (width, height),
        "figure.dpi":        150,   # screen preview
        "savefig.dpi":       300,   # final output
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.05,

    })

# Color palette (colorblind-safe)
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]
CMAP_DIV  = "RdBu_r"   # for Mueller matrix elements
CMAP_SEQ  = "viridis"  # for fluence / absorption maps
