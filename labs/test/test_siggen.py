#!/usr/bin/env python3
"""Quick hardware smoke test for ugradiolab SignalGenerator query methods."""

import argparse

from ugradiolab.drivers.siggen import SignalGenerator, set_signal


def _check_close(name, got, expected, tol):
    if abs(got - expected) > tol:
        raise AssertionError(
            f"{name} mismatch: got {got}, expected {expected} +/- {tol}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Test SignalGenerator query methods on real hardware."
    )
    parser.add_argument("--device", default="/dev/usbtmc0", help="USBTMC device path")
    parser.add_argument(
        "--freq-mhz",
        type=float,
        default=1421.2058,
        help="Temporary test frequency in MHz",
    )
    parser.add_argument(
        "--amp-dbm",
        type=float,
        default=-35.0,
        help="Temporary test amplitude in dBm",
    )
    parser.add_argument(
        "--freq-tol-hz",
        type=float,
        default=5e3,
        help="Allowed frequency error after set/query (Hz)",
    )
    parser.add_argument(
        "--amp-tol-dbm",
        type=float,
        default=0.2,
        help="Allowed amplitude error after set/query (dBm)",
    )
    args = parser.parse_args()

    synth = SignalGenerator(device=args.device)
    original_freq_hz = None
    original_amp_dbm = None
    original_rf_state = None

    try:
        # Low-level query path check.
        idn = synth._query("*IDN?")
        print(f"[OK] *IDN?: {idn}")
        if "N9310A" not in idn:
            raise AssertionError(f"Unexpected IDN response: {idn}")

        # Read and save current state.
        original_freq_hz = synth.get_freq()
        original_amp_dbm = synth.get_ampl()
        original_rf_state = synth.rf_state()
        print(
            "[OK] Initial state:",
            f"freq={original_freq_hz:.3f} Hz,",
            f"amp={original_amp_dbm:.3f} dBm,",
            f"rf_on={original_rf_state}",
        )

        # Set known values and verify all query functions.
        set_signal(synth, args.freq_mhz, args.amp_dbm, rf_on=True)
        freq_hz = synth.get_freq()
        amp_dbm = synth.get_ampl()
        rf_on = synth.rf_state()
        _check_close("get_freq()", freq_hz, args.freq_mhz * 1e6, args.freq_tol_hz)
        _check_close("get_ampl()", amp_dbm, args.amp_dbm, args.amp_tol_dbm)
        if not rf_on:
            raise AssertionError("rf_state() returned OFF after rf_on command")
        print("[OK] Query round-trip while RF ON")

        # Toggle RF off and verify query.
        synth.rf_off()
        if synth.rf_state():
            raise AssertionError("rf_state() returned ON after rf_off command")
        print("[OK] rf_state() while RF OFF")

        print("PASS: all SignalGenerator query methods are working.")
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        # Restore original hardware state if known.
        try:
            if original_freq_hz is not None:
                synth.set_freq_mhz(original_freq_hz / 1e6)
            if original_amp_dbm is not None:
                synth.set_ampl_dbm(original_amp_dbm)
            if original_rf_state is not None:
                if original_rf_state:
                    synth.rf_on()
                else:
                    synth.rf_off()
        except Exception as restore_exc:
            print(f"WARN: failed to restore original state: {restore_exc}")
        finally:
            synth.close()


if __name__ == "__main__":
    raise SystemExit(main())
