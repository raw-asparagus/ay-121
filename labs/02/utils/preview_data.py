from pathlib import Path

import numpy as np

from ugradiolab.data import Record, Spectrum


def choose_zero_files(
    data_dir: Path,
    *,
    lo_freqs_mhz: tuple[int, ...] = (1420, 1421),
    run_index: int = 0,
) -> dict[int, Path]:
    selected: dict[int, Path] = {}
    data_dir = Path(data_dir)
    for lo_mhz in lo_freqs_mhz:
        pattern = f"*-{lo_mhz}-{run_index}_obs_*.npz"
        matches = sorted(data_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
        if not matches:
            raise FileNotFoundError(f"No files matched {pattern!r} in {data_dir}")
        selected[int(lo_mhz)] = matches[-1]
    return selected


def combined_lo_files_exist(
    combined_dir: Path,
    *,
    lo_freqs_mhz: tuple[int, ...] = (1420, 1421),
) -> bool:
    combined_dir = Path(combined_dir)
    if not combined_dir.exists():
        return False
    for lo_mhz in lo_freqs_mhz:
        if not any(combined_dir.glob(f"*-{lo_mhz}_combined.npz")):
            return False
    return True


def load_spectra_cached(data_dir: Path) -> list[tuple[Path, Spectrum]]:
    data_dir = Path(data_dir)
    spectra_dir = data_dir.parent / f"{data_dir.name}_spectra"

    cached = sorted(spectra_dir.glob("*.npz")) if spectra_dir.is_dir() else []
    if cached:
        print(f"Loading pre-processed Spectrum from {spectra_dir}")
        return [(path, Spectrum.load(path)) for path in cached]

    source_files = sorted(data_dir.glob("*.npz"))
    if not source_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    print(f"Generating Spectrum cache: {data_dir} -> {spectra_dir}")
    spectra_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[tuple[Path, Spectrum]] = []
    for path in source_files:
        spectrum = Spectrum.from_data(path)
        cache_path = spectra_dir / path.name
        spectrum.save(cache_path)
        pairs.append((cache_path, spectrum))
    return pairs


def build_preview_rows(paths_by_lo: dict[int, Path]) -> tuple[dict[int, Spectrum], list[dict[str, object]]]:
    pairs_by_lo: dict[int, Spectrum] = {}
    rows: list[dict[str, object]] = []

    for _, path in sorted(paths_by_lo.items()):
        record = Record.load(path)
        spectrum = Spectrum.from_data(path)
        lo_mhz = int(round(spectrum.center_freq / 1e6))
        pairs_by_lo[lo_mhz] = spectrum
        rows.append(_record_preview_row(Path(path), record, spectrum))

    rows.sort(key=lambda row: int(row["LO (MHz)"]))
    return pairs_by_lo, rows


def build_cached_preview_rows(
    spectrum_pairs: list[tuple[Path, Spectrum]],
) -> tuple[dict[int, Spectrum], list[dict[str, object]]]:
    pairs_by_lo: dict[int, Spectrum] = {}
    rows: list[dict[str, object]] = []

    for path, spectrum in sorted(spectrum_pairs, key=lambda pair: pair[1].center_freq):
        lo_mhz = int(round(spectrum.center_freq / 1e6))
        pairs_by_lo[lo_mhz] = spectrum
        rows.append(_spectrum_preview_row(Path(path), spectrum))

    return pairs_by_lo, rows


def _record_preview_row(path: Path, record: Record, spectrum: Spectrum) -> dict[str, object]:
    i_values = record.data.real.ravel()
    q_values = record.data.imag.ravel()
    row = _spectrum_preview_row(path, spectrum)
    row.update(
        {
            "I_max": round(float(i_values.max()), 4),
            "I_min": round(float(i_values.min()), 4),
            "I_rms": round(float(np.sqrt(np.mean(i_values**2))), 4),
            "Q_max": round(float(q_values.max()), 4),
            "Q_min": round(float(q_values.min()), 4),
            "Q_rms": round(float(np.sqrt(np.mean(q_values**2))), 4),
        }
    )
    return row


def _spectrum_preview_row(path: Path, spectrum: Spectrum) -> dict[str, object]:
    return {
        "LO (MHz)": int(round(spectrum.center_freq / 1e6)),
        "filename": Path(path).name,
        "nblocks": spectrum.nblocks,
        "nsamples": spectrum.nsamples,
        "alt (deg)": round(float(spectrum.alt), 2),
        "az (deg)": round(float(spectrum.az), 2),
        "JD": round(float(spectrum.jd), 5),
        "total_power": round(float(spectrum.total_power), 8),
        "sigma": round(float(spectrum.total_power_sigma), 8),
    }
