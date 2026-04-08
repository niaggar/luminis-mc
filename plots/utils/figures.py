import numpy as np
import matplotlib.pyplot as plt

def get_label(data, run_name):
    radius = data[run_name]["radius"]
    return f"$a = {radius*1000:.1f}$ nm"

def plot_disk(Icoh, Iinc, labels, theta_deg, phi_data, title="CBS Disk Plot", zoom_theta_deg=0.0, save_path=None):
    plot_data = [
        Icoh, 
        Iinc,   
    ]
    row_labels = ["Coherent", "Incoherent"]
    nrows = 3 if zoom_theta_deg > 0.0 else 2

    fig, axes = plt.subplots(nrows, 3, figsize=(10, 2.5 * nrows), subplot_kw={'projection': 'polar'})
    fig.suptitle(title, fontsize=16)
    
    if nrows == 3:
        plot_data.append(Icoh)
        row_labels.append(f"Zoom (Coherente)\n$\\theta \\leq {zoom_theta_deg}^\\circ$")

    for r in range(nrows):
        for c in range(3):
            ax = axes[r, c] if nrows > 1 else axes[c]
            Z = plot_data[r][c]

            # ── Reseteo de escala para el zoom mediante Slicing ─────────────────
            if r == 2:
                # Crear máscara booleana 1D de los ángulos válidos
                valid_idx = theta_deg <= zoom_theta_deg
                
                # Recortar arreglos espacialmente
                theta_plot = theta_deg[valid_idx]
                Z_plot = Z[valid_idx, :]
                # Validar si phi_data es 2D (meshgrid) o 1D para aplicarle el recorte
                phi_plot = phi_data[valid_idx, :] if phi_data.ndim == 2 else phi_data
            else:
                theta_plot = theta_deg
                Z_plot = Z
                phi_plot = phi_data

            # contourf recibe exclusivamente los datos recortados
            cplot = ax.contourf(phi_plot, theta_plot, Z_plot, levels=100, cmap='jet')
            
            # Configurar el colorbar
            cbar = fig.colorbar(cplot, ax=ax, fraction=0.046, pad=0.08)
            cbar.formatter.set_powerlimits((0, 0)) # Fuerza la notación científica x10^n
            cbar.ax.yaxis.set_offset_position('right')
            cbar.update_ticks()

            # Ajustar límites radiales visuales
            if r == 2:
                ax.set_ylim(0, zoom_theta_deg)
            else:
                ax.set_ylim(0, np.max(theta_deg))

            # Limpiar ejes y mallas para que parezca un disco puro
            ax.grid(False)
            ax.set_xticks([]) # Oculta los ángulos acimutales
            ax.set_yticks([]) # Oculta los anillos radiales
            ax.spines['polar'].set_visible(True) # Mantiene el borde circular

            # Etiquetas
            if r == 0:
                ax.set_title(labels[c], fontsize=14, pad=15)
            if c == 0:
                ax.text(-0.2, 0.5, row_labels[r], transform=ax.transAxes, 
                        va='center', ha='right', fontsize=14, rotation=90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_profiles_left_right(x_axis, left_series, right_series, labels_left, labels_right, title, title_left, title_right, xlabel, ylabel, max_x=None, min_x=None, save_path=None):
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for l, label in zip(left_series, labels_left):
        ax_left.plot(x_axis, l, label=label)

    for r, label in zip(right_series, labels_right):
        ax_right.plot(x_axis, r, label=label)

    ax_left.set_title(title_left)
    ax_right.set_title(title_right)

    for ax in [ax_left, ax_right]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if max_x is not None:
            ax.set_xlim(-max_x, max_x)
        if min_x is not None:
            ax.set_xlim(min_x, max_x)
        ax.legend()
    
    fig.suptitle(title)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    return fig