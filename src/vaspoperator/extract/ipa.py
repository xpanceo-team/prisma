import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

logger = logging.getLogger("IPA Utils")


def eps_to_principal_nk(df: pl.DataFrame) -> pl.DataFrame:
    """
    Converts frequency-dependent dielectric tensor to principal refractive indices (n, k).
    Includes a tracking algorithm to prevent 'branch jumping' of eigenvalues.
    """
    HC_EV_NM = 1239.84193  # Standard conversion constant

    # Calculate wavelength
    df = df.with_columns((HC_EV_NM / pl.col("Energies")).alias("wavelength_nm"))

    # Extract components as numpy for tensor reconstruction
    # VASP order: xx, yy, zz, xy, yz, xz
    re = {
        c: df[f"real_e_{c}"].to_numpy()
        for c in ["xx", "yy", "zz", "xy", "yz", "xz"]
    }
    im = {
        c: df[f"imag_e_{c}"].to_numpy()
        for c in ["xx", "yy", "zz", "xy", "yz", "xz"]
    }

    n_freq = len(df)
    eigvals_tracked = np.zeros((n_freq, 3), dtype=complex)

    # Pre-define permutations for eigenvalue tracking
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

    for i in range(n_freq):
        # Construct symmetric complex dielectric tensor
        eps = np.array(
            [
                [
                    re["xx"][i] + 1j * im["xx"][i],
                    re["xy"][i] + 1j * im["xy"][i],
                    re["xz"][i] + 1j * im["xz"][i],
                ],
                [
                    re["xy"][i] + 1j * im["xy"][i],
                    re["yy"][i] + 1j * im["yy"][i],
                    re["yz"][i] + 1j * im["yz"][i],
                ],
                [
                    re["xz"][i] + 1j * im["xz"][i],
                    re["yz"][i] + 1j * im["yz"][i],
                    re["zz"][i] + 1j * im["zz"][i],
                ],
            ],
            dtype=complex,
        )

        curr_vals = np.linalg.eigvals(eps)

        if i == 0:
            # Initial sort by real part
            eigvals_tracked[i] = curr_vals[np.argsort(curr_vals.real)]
        else:
            # Track eigenvalues by minimizing the Euclidean distance from previous step
            prev = eigvals_tracked[i - 1]
            best_dist = float("inf")
            best_p = curr_vals

            for p in perms:
                permuted = curr_vals[list(p)]
                dist = np.sum(np.abs(prev - permuted) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_p = permuted
            eigvals_tracked[i] = best_p

    # Convert complex eigenvalues (epsilon) to n and k
    # n + ik = sqrt(epsilon_real + i*epsilon_imag)
    # Using the analytical form:
    # n = sqrt((|eps| + eps_re) / 2)
    # k = sqrt((|eps| - eps_re) / 2)

    mod_eps = np.abs(eigvals_tracked)
    re_eps = eigvals_tracked.real

    n_vals = np.sqrt(np.maximum((mod_eps + re_eps) / 2.0, 0.0))
    k_vals = np.sqrt(np.maximum((mod_eps - re_eps) / 2.0, 0.0))

    return df.with_columns(
        [
            pl.Series("n_xx", n_vals[:, 0]),
            pl.Series("n_yy", n_vals[:, 1]),
            pl.Series("n_zz", n_vals[:, 2]),
            pl.Series("k_xx", k_vals[:, 0]),
            pl.Series("k_yy", k_vals[:, 1]),
            pl.Series("k_zz", k_vals[:, 2]),
        ]
    )


def plot_nk_vs_wavelength(
    df: pl.DataFrame,
    save_dir: str | Path,
    id: str,
    filename_prefix: str = "nk_plot",
    plot_components: list[str] = None,
    show_plot: bool = False,
    dpi: int = 300,
    figsize: tuple[int, int] = (10, 6),
    xlim: tuple[float, float] | None = (300, 2000),
    ylim_n: tuple[float, float] | None = None,
    ylim_k: tuple[float, float] | None = None,
    title: str | None = None,
) -> None:
    """Generates a publication-quality plot of n and k vs wavelength."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    plot_components = plot_components or ["xx", "yy", "zz"]
    df_plot = df.sort("wavelength_nm")

    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    ax2 = ax1.twinx()

    colors = {"xx": "#E63946", "yy": "#457B9D", "zz": "#1D3557"}
    n_lines, k_lines = [], []

    for comp in plot_components:
        c = colors.get(comp, "black")

        # Plot n (Refractive Index)
        ln_n = ax1.plot(
            df_plot["wavelength_nm"],
            df_plot[f"n_{comp}"],
            color=c,
            lw=2,
            label=f"$n_{{{comp}}}$",
        )
        n_lines.extend(ln_n)

        # Plot k (Extinction Coefficient)
        ln_k = ax2.plot(
            df_plot["wavelength_nm"],
            df_plot[f"k_{comp}"],
            color=c,
            lw=1.5,
            ls="--",
            alpha=0.7,
            label=f"$k_{{{comp}}}$",
        )
        k_lines.extend(ln_k)

    # Reference line for 1064nm (YAG laser common reference)
    v_ref = ax1.axvline(
        x=1064, color="#6D6875", ls=":", lw=1.5, alpha=0.6, label="1064 nm"
    )

    # Formatting
    ax1.set_xlabel("Wavelength (nm)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Refractive Index ($n$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel(
        "Extinction Coefficient ($k$)",
        fontsize=11,
        fontweight="bold",
        rotation=270,
        labelpad=15,
    )

    if xlim:
        ax1.set_xlim(xlim)
    if ylim_n:
        ax1.set_ylim(ylim_n)
    if ylim_k:
        ax2.set_ylim(ylim_k)

    ax1.grid(True, which="major", linestyle="--", alpha=0.4)
    ax1.set_title(title or f"Optical Constants: {id}", fontsize=13, pad=15)

    # Combined Legend
    all_lns = n_lines + k_lines + [v_ref]
    labs = [lab.get_label() for lab in all_lns]
    ax1.legend(all_lns, labs, loc="center right", frameon=True, fontsize=9)

    plt.tight_layout()

    out_file = save_path / f"{filename_prefix}.png"
    plt.savefig(out_file, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    logger.info(f"Saved NK plot to {out_file}")
