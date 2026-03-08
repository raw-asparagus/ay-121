#!/usr/bin/env python3
"""Characterize bandpass filters in the signal chain.

Two modes:
  bw-scan   : sweep signal frequency ν with LO = ν − 0.5 MHz to locate
               filter cut-on/cut-off edges.
  resp-scan : fix LO at 1420 MHz and 1421 MHz, sweep ν within ±1.1 MHz,
               measure in-band frequency response (FIR-corrected).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from ugradio.sdr import SDR

from ugradiolab import SignalGenerator, Spectrum
from ugradiolab.run import CalExperiment

# ---------------------------------------------------------------------------
# Output directories
OUTDIR_BW   = 'data/lab02/bandpass/bw_scan'
OUTDIR_RESP = 'data/lab02/bandpass/resp_scan'

# Bandwidth scan parameters
BW_NU_START_MHZ   = 1415.0
BW_NU_STOP_MHZ    = 1425.0
BW_NU_STEP_MHZ    = 0.1
BW_LO_OFFSET_MHZ  = 0.5     # LO = nu - this; keeps FIR gain constant

# Response scan parameters
RESP_LO_LIST_MHZ      = (1420.0, 1421.0)
RESP_OFFSET_START_MHZ = -1.1
RESP_OFFSET_STOP_MHZ  =  1.1
RESP_OFFSET_STEP_MHZ  =  0.05

# Hardware
SIGGEN_DEVICE    = '/dev/usbtmc0'
SIGGEN_AMP_DBM   = -40.0
SDR_GAIN_DB      =   0.0
NSAMPLES         =  8192
NBLOCKS          =   128
SAMPLE_RATE_HZ   =  2.56e6
SDR_DIRECT       = False

# Analysis thresholds
PEAK_HALF_WIDTH_BINS   = 5
RECOVERED_THRESHOLD_DB = 6.0

# ---------------------------------------------------------------------------
# FIR coefficients from equipment_calibration.ipynb cell 31
_H_FIR = np.array([
    -54, -36, -41, -40, -32, -14,  14,  53,
    101, 156, 215, 273, 327, 372, 404, 421,
    421, 404, 372, 327, 273, 215, 156, 101,
     53,  14, -14, -32, -40, -41, -36, -54,
], dtype=float)

# CSV field definitions
BW_MANIFEST_FIELDS = [
    'point_id', 'nu_mhz', 'lo_mhz', 'bb_offset_mhz',
    'peak_power_db', 'noise_floor_db', 'snr_db',
    'fir_gain_db', 'corrected_peak_db', 'recovered', 'capture_path',
]

RESP_MANIFEST_FIELDS = [
    'point_id', 'lo_mhz', 'nu_mhz', 'bb_offset_mhz',
    'peak_power_db', 'noise_floor_db', 'snr_db',
    'fir_gain_db', 'corrected_peak_db', 'capture_path',
]


# ---------------------------------------------------------------------------
# Helper functions

def _fir_gain_db(bb_offset_hz: float, nsamples: int, sample_rate: float) -> float:
    """FIR power gain (dB) at a baseband frequency offset.

    Normalised so that the peak = 0 dB.
    """
    H = np.fft.rfft(_H_FIR, n=nsamples)
    freqs = np.fft.rfftfreq(nsamples, d=1.0 / sample_rate)
    power = np.abs(H) ** 2
    # interpolate at |bb_offset_hz|; FIR gain is symmetric
    target = abs(bb_offset_hz)
    gain_at_target = float(np.interp(target, freqs, power))
    peak_power = float(np.max(power))
    if peak_power <= 0.0:
        return 0.0
    return 10.0 * math.log10(gain_at_target / peak_power)


def _measure_peak(
    path: str,
    expected_bb_hz: float,
) -> tuple[float, float, float]:
    """Measure peak power, noise floor, and SNR from a captured file.

    Parameters
    ----------
    path : str
        Path to a .npz Record file.
    expected_bb_hz : float
        Expected baseband offset of the tone in Hz (signed).

    Returns
    -------
    (peak_power_db, noise_floor_db, snr_db)
    """
    spec = Spectrum.from_data(path)
    bb_freqs = spec.freqs - spec.center_freq

    # find bin closest to expected tone
    center_bin = int(np.argmin(np.abs(bb_freqs - expected_bb_hz)))
    n = len(spec.psd)

    lo_bin = max(0, center_bin - PEAK_HALF_WIDTH_BINS)
    hi_bin = min(n, center_bin + PEAK_HALF_WIDTH_BINS + 1)

    peak_mask = np.zeros(n, dtype=bool)
    peak_mask[lo_bin:hi_bin] = True

    peak_power = float(np.sum(spec.psd[peak_mask]))
    noise_floor = float(np.median(spec.psd[~peak_mask]))

    peak_power_db = 10.0 * math.log10(peak_power) if peak_power > 0 else math.nan
    noise_floor_db = 10.0 * math.log10(noise_floor) if noise_floor > 0 else math.nan

    if peak_power > 0 and noise_floor > 0:
        snr_db = 10.0 * math.log10(peak_power / noise_floor)
    else:
        snr_db = math.nan

    return peak_power_db, noise_floor_db, snr_db


def _run_capture(
    point_id: int,
    lo_mhz: float,
    nu_mhz: float,
    outdir: str | Path,
    sdr,
    synth,
) -> str:
    """Run one CalExperiment capture and return the file path."""
    prefix = (
        f'BPCHAR-p{point_id:04d}'
        f'-LO{lo_mhz:.3f}'
        f'-NU{nu_mhz:.3f}'
    )
    exp = CalExperiment(
        nsamples=NSAMPLES,
        nblocks=NBLOCKS,
        sample_rate=SAMPLE_RATE_HZ,
        center_freq=lo_mhz * 1e6,
        gain=SDR_GAIN_DB,
        direct=SDR_DIRECT,
        outdir=str(outdir),
        prefix=prefix,
        alt_deg=0.0,
        az_deg=0.0,
        siggen_freq_mhz=nu_mhz,
        siggen_amp_dbm=SIGGEN_AMP_DBM,
    )
    return exp.run(sdr, synth=synth)


def _append_csv_row(
    path: Path,
    fieldnames: list[str],
    row: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open('a', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Bandwidth scan

def run_bw_scan(sdr, synth) -> None:
    outdir = Path(OUTDIR_BW)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir.parent / 'bw_manifest.csv'

    nu_values = np.arange(
        BW_NU_START_MHZ,
        BW_NU_STOP_MHZ + BW_NU_STEP_MHZ / 2,
        BW_NU_STEP_MHZ,
    )
    n_total = len(nu_values)

    # FIR gain at the constant bb offset is the same for every point
    expected_bb_hz = BW_LO_OFFSET_MHZ * 1e6
    fir_gain_db_val = _fir_gain_db(expected_bb_hz, NSAMPLES, SAMPLE_RATE_HZ)

    print('=== Bandwidth scan ===')
    print(f'  ν range : {BW_NU_START_MHZ:.1f} – {BW_NU_STOP_MHZ:.1f} MHz  step={BW_NU_STEP_MHZ:.2f} MHz')
    print(f'  LO offset : {BW_LO_OFFSET_MHZ:.2f} MHz  (bb tone at {BW_LO_OFFSET_MHZ:.2f} MHz)')
    print(f'  FIR gain at bb offset : {fir_gain_db_val:.3f} dB (constant)')
    print(f'  Points : {n_total}')
    print()

    recovered_nus: list[float] = []

    for idx, nu_mhz in enumerate(nu_values, start=1):
        lo_mhz = nu_mhz - BW_LO_OFFSET_MHZ
        print(
            f'  [{idx:03d}/{n_total}] ν={nu_mhz:.3f} MHz  LO={lo_mhz:.3f} MHz',
            end='  ', flush=True,
        )
        path = _run_capture(idx, lo_mhz, nu_mhz, outdir, sdr, synth)
        peak_db, noise_db, snr_db = _measure_peak(path, expected_bb_hz)
        corrected_db = peak_db - fir_gain_db_val
        recovered = bool(not math.isnan(snr_db) and snr_db > RECOVERED_THRESHOLD_DB)
        if recovered:
            recovered_nus.append(nu_mhz)

        print(
            f'SNR={snr_db:+.1f} dB  peak={peak_db:.2f} dB  '
            f'corrected={corrected_db:.2f} dB  '
            f'{"RECOVERED" if recovered else "missed"}'
        )

        row: dict[str, object] = {
            'point_id': idx,
            'nu_mhz': nu_mhz,
            'lo_mhz': lo_mhz,
            'bb_offset_mhz': BW_LO_OFFSET_MHZ,
            'peak_power_db': peak_db,
            'noise_floor_db': noise_db,
            'snr_db': snr_db,
            'fir_gain_db': fir_gain_db_val,
            'corrected_peak_db': corrected_db,
            'recovered': int(recovered),
            'capture_path': path,
        }
        _append_csv_row(manifest_path, BW_MANIFEST_FIELDS, row)

    print()
    if recovered_nus:
        print(f'  Filter passband estimate : {min(recovered_nus):.1f} – {max(recovered_nus):.1f} MHz')
        print(f'  Recovered {len(recovered_nus)} / {n_total} points')
    else:
        print('  No points recovered — check signal chain or thresholds')
    print(f'  Manifest : {manifest_path}')
    print()


# ---------------------------------------------------------------------------
# Response scan

def run_resp_scan(sdr, synth) -> None:
    outdir = Path(OUTDIR_RESP)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = outdir.parent / 'resp_manifest.csv'

    offsets_raw = np.arange(
        RESP_OFFSET_START_MHZ,
        RESP_OFFSET_STOP_MHZ + RESP_OFFSET_STEP_MHZ / 2,
        RESP_OFFSET_STEP_MHZ,
    )
    # skip near-DC offset
    offsets = offsets_raw[np.abs(offsets_raw) > RESP_OFFSET_STEP_MHZ / 2]

    n_per_lo = len(offsets)
    n_total = len(RESP_LO_LIST_MHZ) * n_per_lo

    print('=== Response scan ===')
    print(
        f'  LO list : {", ".join(f"{lo:.0f}" for lo in RESP_LO_LIST_MHZ)} MHz'
    )
    print(
        f'  Offset range : {RESP_OFFSET_START_MHZ:.2f} – {RESP_OFFSET_STOP_MHZ:.2f} MHz  '
        f'step={RESP_OFFSET_STEP_MHZ:.3f} MHz'
    )
    print(f'  Points per LO : {n_per_lo}  (total {n_total})')
    print()

    point_id = 1
    for lo_mhz in RESP_LO_LIST_MHZ:
        print(f'  LO = {lo_mhz:.0f} MHz')
        lo_rows: list[dict[str, object]] = []

        for offset in offsets:
            nu_mhz = lo_mhz + offset
            expected_bb_hz = offset * 1e6
            fir_gain_db_val = _fir_gain_db(expected_bb_hz, NSAMPLES, SAMPLE_RATE_HZ)

            print(
                f'    [{point_id:04d}] offset={offset:+.3f} MHz  ν={nu_mhz:.3f} MHz',
                end='  ', flush=True,
            )
            path = _run_capture(point_id, lo_mhz, nu_mhz, outdir, sdr, synth)
            peak_db, noise_db, snr_db = _measure_peak(path, expected_bb_hz)
            corrected_db = peak_db - fir_gain_db_val

            print(
                f'peak={peak_db:.2f} dB  '
                f'fir={fir_gain_db_val:.2f} dB  '
                f'corrected={corrected_db:.2f} dB  '
                f'SNR={snr_db:+.1f} dB'
            )

            row: dict[str, object] = {
                'point_id': point_id,
                'lo_mhz': lo_mhz,
                'nu_mhz': nu_mhz,
                'bb_offset_mhz': offset,
                'peak_power_db': peak_db,
                'noise_floor_db': noise_db,
                'snr_db': snr_db,
                'fir_gain_db': fir_gain_db_val,
                'corrected_peak_db': corrected_db,
                'capture_path': path,
            }
            _append_csv_row(manifest_path, RESP_MANIFEST_FIELDS, row)
            lo_rows.append(row)
            point_id += 1

        # per-LO summary: 3 dB bandwidth
        corrected_vals = np.array(
            [r['corrected_peak_db'] for r in lo_rows], dtype=float
        )
        valid = np.isfinite(corrected_vals)
        if valid.any():
            peak_val = float(np.nanmax(corrected_vals))
            in_band = np.array(
                [r['nu_mhz'] for r in lo_rows], dtype=float
            )[valid & (corrected_vals >= peak_val - 3.0)]
            print(
                f'    3 dB passband (LO={lo_mhz:.0f}): '
                f'{in_band.min():.3f} – {in_band.max():.3f} MHz'
            )
        print()

    print(f'  Manifest : {manifest_path}')
    print()


# ---------------------------------------------------------------------------
# Main

def main() -> int:
    parser = argparse.ArgumentParser(
        description='Bandpass filter characterization for the 21-cm signal chain.'
    )
    parser.add_argument(
        '--mode',
        choices=['bw-scan', 'resp-scan', 'both'],
        default='both',
        help='Which scan to run (default: both)',
    )
    args = parser.parse_args()

    print('Bandpass characterization')
    print(f'  Mode          : {args.mode}')
    print(f'  Siggen amp    : {SIGGEN_AMP_DBM:.1f} dBm')
    print(
        f'  SDR profile   : sample_rate={SAMPLE_RATE_HZ/1e6:.3f} MHz  '
        f'nsamples={NSAMPLES}  nblocks={NBLOCKS}  '
        f'gain={SDR_GAIN_DB}  direct={SDR_DIRECT}'
    )
    print()

    sdr = None
    synth = None
    try:
        sdr = SDR(
            direct=SDR_DIRECT,
            center_freq=1420e6,
            sample_rate=SAMPLE_RATE_HZ,
            gain=SDR_GAIN_DB,
        )
        synth = SignalGenerator(device=SIGGEN_DEVICE)

        if args.mode in ('bw-scan', 'both'):
            run_bw_scan(sdr, synth)
        if args.mode in ('resp-scan', 'both'):
            run_resp_scan(sdr, synth)
    finally:
        if synth is not None:
            try:
                synth.rf_off()
            except Exception:
                pass
        if sdr is not None:
            sdr.close()

    print('Done.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
