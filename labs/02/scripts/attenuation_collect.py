#!/usr/bin/env python3
"""Collect fixed-tone attenuation data across cable-length sets.

Hardware setup
--------------

Signal path (common trunk):
  Signal generator
    → [N-male to BNC] reference cable (RG58, 50 Ω, variable length)
    → [BNC to SMA] ZFSC-2-372-S+ 2-way splitter (port S)

Splitter output (1) — reference arm:
    → [SMA] Power meter  (read manually before each SDR capture)

Splitter output (2) — SDR arm:
    → [SMA] 6-ft RG58 cable (50 Ω)
    → [SMA] 3 dB attenuator
    → [SMA] SDR

Workflow per set:
1. Enter cable length (m).
2. Turn RF on and enter manual power meter reading (dBm).
3. Turn RF off, switch cable path to SDR, and confirm.
4. Capture all LO frequencies.
5. Print SDR metrics, then choose to save or discard the set.
"""

from pathlib import Path

from ugradio.sdr import SDR

from ugradiolab import SignalGenerator
from ugradiolab.run import CalExperiment

from utils.tools import (
    LO_FREQS_HZ,
    next_id_from_manifest,
    compute_capture_metrics, print_capture_metrics,
    append_manifest_row, remove_manifest_rows_for_paths,
    build_manifest_row, delete_capture_file,
)

# ---------------------------------------------------------------------------
# Session configuration

SIGGEN_FREQ_MHZ = 1420.405751768

OUTDIR        = "data/lab02/attenuation/raw"
MANIFEST_PATH = "data/lab02/attenuation/manifest.csv"
START_SET_ID: int | None = None

COMMON_CAPTURE = dict(
    nsamples=8192,
    nblocks=2048,
    direct=False,
    sample_rate=2.56e6,
    gain=0.0,
    alt_deg=0.0,
    az_deg=0.0,
)


# ---------------------------------------------------------------------------
# Prompt helpers

def prompt_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt).strip())
        except ValueError:
            print("  Invalid number, please try again.")


def prompt_length_or_quit() -> tuple[float | None, bool]:
    while True:
        raw = input("Cable length [m] (or q to quit): ").strip()
        if raw.lower() == "q":
            return None, True
        try:
            value = float(raw)
        except ValueError:
            print("  Invalid number, please try again.")
            continue
        if value < 0:
            print("  Cable length must be non-negative.")
            continue
        return value, False


def prompt_yes_no(prompt: str) -> bool:
    while True:
        raw = input(prompt).strip().lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  Please enter y/yes or n/no.")


# ---------------------------------------------------------------------------
# Capture

def _capture(*, set_id, cable_length_m, lo_hz, siggen_amp_dbm, sdr, synth):
    lo_mhz = int(round(lo_hz / 1e6))
    cable_length_tag = f"{cable_length_m:.3f}".replace(".", "p")
    prefix = f"ATTEN-set{set_id:04d}-LO{lo_mhz}-L{cable_length_tag}m"
    exp = CalExperiment(
        sdr=sdr, synth=synth,
        **COMMON_CAPTURE,
        center_freq=lo_hz,
        outdir=OUTDIR, prefix=prefix,
        siggen_freq_mhz=SIGGEN_FREQ_MHZ,
        siggen_amp_dbm=siggen_amp_dbm,
    )
    return exp.run()


# ---------------------------------------------------------------------------

def main() -> int:
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(MANIFEST_PATH)

    set_id = START_SET_ID if START_SET_ID is not None else next_id_from_manifest(manifest_path, "set_id")

    siggen_amp_dbm = prompt_float("Signal generator amplitude [dBm]: ")

    print("Attenuation collection session")
    print(f"  Siggen frequency: {SIGGEN_FREQ_MHZ:.9f} MHz")
    print(f"  Siggen amplitude: {siggen_amp_dbm:.2f} dBm")
    print(
        "  SDR profile: "
        f"sample_rate={COMMON_CAPTURE['sample_rate']/1e6:.3f} MHz, "
        f"nsamples={COMMON_CAPTURE['nsamples']}, "
        f"nblocks={COMMON_CAPTURE['nblocks']}, "
        f"gain={COMMON_CAPTURE['gain']}, "
        f"direct={COMMON_CAPTURE['direct']}"
    )
    print(f"  Output directory: {outdir}")
    print(f"  Manifest: {manifest_path}")
    print()

    completed = 0
    sdr = synth = None
    try:
        sdr = SDR(
            direct=COMMON_CAPTURE["direct"],
            center_freq=LO_FREQS_HZ[0],
            sample_rate=COMMON_CAPTURE["sample_rate"],
            gain=COMMON_CAPTURE["gain"],
        )
        synth = SignalGenerator()
        synth.set_freq_mhz(SIGGEN_FREQ_MHZ)
        synth.set_ampl_dbm(siggen_amp_dbm)

        while True:
            print(f"Set {set_id:04d}")
            cable_length_m, quit_now = prompt_length_or_quit()
            if quit_now:
                break
            assert cable_length_m is not None

            power_meter_dbm = None
            synth.rf_on()
            try:
                power_meter_dbm = prompt_float("RF is ON. Manual power meter reading [dBm]: ")
            finally:
                try:
                    synth.rf_off()
                except Exception:
                    pass

            switch_confirm = input("RF is OFF. Switch cable path to SDR, then press Enter (or q to quit): ").strip().lower()
            if switch_confirm == "q":
                break

            paths = {}
            metrics = {}
            for lo_hz in LO_FREQS_HZ:
                lo_mhz = int(round(lo_hz / 1e6))
                path = _capture(set_id=set_id, cable_length_m=cable_length_m, lo_hz=lo_hz,
                                siggen_amp_dbm=siggen_amp_dbm, sdr=sdr, synth=synth)
                m = compute_capture_metrics(path)
                print_capture_metrics(lo_mhz, m)
                paths[lo_mhz] = path
                metrics[lo_mhz] = m

            row = build_manifest_row(
                set_id=set_id,
                paths=paths, metrics=metrics,
                cable_length_m=cable_length_m, power_meter_dbm=power_meter_dbm,
                siggen_freq_mhz=SIGGEN_FREQ_MHZ, siggen_amp_dbm=siggen_amp_dbm,
            )

            if prompt_yes_no("Save this run? [y/n]: "):
                append_manifest_row(manifest_path, row)
                completed += 1
                print(f"  saved set {set_id:04d}")
                print()
                set_id += 1
            else:
                for path in paths.values():
                    delete_capture_file(path)
                remove_manifest_rows_for_paths(manifest_path, paths)
                print(f"  discarded set {set_id:04d}; manifest/data removed")
                print()
    finally:
        if synth is not None:
            synth.close()
        if sdr is not None:
            sdr.close()

    print("Session complete")
    print(f"  completed sets: {completed}")
    print(f"  manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
