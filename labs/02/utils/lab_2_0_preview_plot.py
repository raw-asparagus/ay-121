import matplotlib.pyplot as plt

from ugradiolab.data import Spectrum

from .plotting import TEXTWIDTH_IN
from .spectrum_plot import plot_spectrum_compare, plot_spectrum_ratio


def plot_dataset_pair(
    dataset_label: str,
    sub_label: str,
    spec_1420: Spectrum,
    spec_1421: Spectrum,
    savgol: dict,
) -> None:
    dataset_tag = f"{dataset_label} {sub_label}"

    R_total = spec_1420.total_power / spec_1421.total_power
    inv_R_total = spec_1421.total_power / spec_1420.total_power

    print(f"R / 1/R summary ({dataset_tag})")
    print("-" * 80)
    print(f"total_power_1420 = {spec_1420.total_power:.10f}")
    print(f"total_power_1421 = {spec_1421.total_power:.10f}")
    print(f"R_total         = {R_total:.10f}")
    print(f"1/R_total       = {inv_R_total:.10f}")

    fig = plt.figure(figsize=(TEXTWIDTH_IN, 6.4 / 8 * TEXTWIDTH_IN), constrained_layout=True)
    gs = fig.add_gridspec(
        nrows=3,
        ncols=1,
        height_ratios=[2, 1, 1],
        hspace=0.05,
    )

    ax_psd = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_psd)
    ax_inv_ratio = fig.add_subplot(gs[2], sharex=ax_psd)

    # Top panel: compare PSDs
    plot_spectrum_compare(
        spec_1420,
        spec_1421,
        ax=ax_psd,
        title=f"{dataset_tag}: LO 1420 vs LO 1421 PSD",
        labels=("LO 1420", "LO 1421"),
        colors=("C0", "C1"),
        smooth_kwargs=savgol,
        mask_dc=True,
        x_mode="baseband",   # must match the ratio plots if x-axis is shared
        yscale="log",
        legend=True,
        title_loc="left",
    )

    # Middle panel: R = LO1420 / LO1421
    plot_spectrum_ratio(
        spec_1420,
        spec_1421,
        ax=ax_ratio,
        title=f"{dataset_tag}: R = LO1420 / LO1421",
        smooth_kwargs=savgol,
        x_mode="baseband",
        raw_label="raw ratio",
        smooth_label="smoothed ratio",
        add_unity_line=True,
        ylabel="Power ratio",
        title_loc="left",
    )

    # Bottom panel: 1/R = LO1421 / LO1420
    plot_spectrum_ratio(
        spec_1421,
        spec_1420,
        ax=ax_inv_ratio,
        title=f"{dataset_tag}: 1/R = LO1421 / LO1420",
        smooth_kwargs=savgol,
        x_mode="baseband",
        raw_label="raw inverse ratio",
        smooth_label="smoothed inverse ratio",
        add_unity_line=True,
        ylabel="Inverse power ratio",
        title_loc="left",
    )

    ax_psd.label_outer()
    ax_ratio.label_outer()

    plt.show()
