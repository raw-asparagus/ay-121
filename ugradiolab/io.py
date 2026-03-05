from pathlib import Path
from typing import Callable, Sequence

from .models import Spectrum


def load_spectra_cached(
        data_dir,
        exclude_fn: Callable[[str], bool] | None = None,
) -> list[Spectrum]:
    """Load cached Spectrum files or process raw records on demand."""
    data_path = Path(data_dir)
    lite_dir = data_path.parent / f'{data_path.name}_lite'

    if lite_dir.is_dir():
        lite_files = sorted(lite_dir.glob('*.npz'))
        if lite_files:
            print(f'Loading pre-processed Spectrum from {lite_dir}')
            return [Spectrum.load(path) for path in lite_files]

    print(f'Processing raw data from {data_path} -> {lite_dir}')
    lite_dir.mkdir(parents=True, exist_ok=True)

    spectra = []
    for path in sorted(data_path.glob('*.npz')):
        path_str = str(path)
        if exclude_fn is not None and exclude_fn(path_str):
            continue
        spectrum = Spectrum.from_data(path)
        spectrum.save(lite_dir / path.name)
        spectra.append(spectrum)
    return spectra


def select_spectrum_by_center_freq(
        spectra: Sequence[Spectrum],
        center_freq_hz: float,
        tol_hz: float = 0.5e6,
) -> Spectrum:
    """Return the spectrum whose centre frequency is nearest the target."""
    matches = [
        spectrum for spectrum in spectra
        if abs(spectrum.center_freq - center_freq_hz) < tol_hz
    ]
    if not matches:
        raise ValueError(
            f'No spectrum found near center_freq={center_freq_hz / 1e6:.0f} MHz'
        )
    return min(matches, key=lambda spectrum: abs(spectrum.center_freq - center_freq_hz))


def select_spectra_by_center_freq(
        spectra: Sequence[Spectrum],
        center_freqs_hz: Sequence[float],
        tol_hz: float = 0.5e6,
) -> dict[float, Spectrum]:
    """Return spectra keyed by requested centre frequency."""
    return {
        center_freq_hz: select_spectrum_by_center_freq(
            spectra,
            center_freq_hz,
            tol_hz=tol_hz,
        )
        for center_freq_hz in center_freqs_hz
    }
