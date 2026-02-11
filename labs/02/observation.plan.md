# Observation Plan - Lab 02 (Bighorn Calibration)

Fixed settings:
- SDR: `direct=False`, `center_freq=1420.4058e6`, `sample_rate=2.56e6` (fallback `2.40e6`), fixed `gain`, `nsamples=4096`, `nblocks=500`
- Signal generator: CW, `50 ohm`, modulation OFF

## 13 Runs

1. `Z-BASE-1`
   `COLD` (temperature reference): Zenith, generator OFF.

2. `Z-BASE-2`
   `COLD` (temperature reference): Zenith, generator OFF (repeat).

3. `H-BASE-1`
   `HOT` (temperature reference): Horizontal, generator OFF.

4. `H-BASE-2`
   `HOT` (temperature reference): Horizontal, generator OFF (repeat).

5. `Z-TONE-PWR1`
   `TONE` (chain/frequency check): Zenith, generator ON, `1421.2058 MHz`, `-44 dBm` (~25 counts).

6. `Z-TONE-PWR2`
   `TONE` (chain/frequency check): Zenith, generator ON, `1421.2058 MHz`, `-41 dBm` (~35 counts).

7. `Z-TONE-PWR3`
   `TONE` (chain/frequency check): Zenith, generator ON, `1421.2058 MHz`, `-38 dBm` (~50 counts, default).

8. `Z-TONE-UP100`
   `TONE` (frequency-tracking): Zenith, generator ON, `1421.3058 MHz`, `-38 dBm`.

9. `Z-TONE-DN100`
   `TONE` (frequency-tracking): Zenith, generator ON, `1421.1058 MHz`, `-38 dBm`.

10. `Z-TONE-LOWER`
    `TONE` (lower-side IF check): Zenith, generator ON, `1419.6058 MHz`, `-38 dBm`.

11. `H-TONE`
    `TONE` (hot-geometry check): Horizontal, generator ON, `1421.2058 MHz`, `-38 dBm`.

12. `H-POST`
    `HOT` (temperature reference): Horizontal, generator OFF.

13. `Z-POST`
    `COLD` (temperature reference): Zenith, generator OFF.

Optional short check:
- `Z-TONE-HIGH`: `TONE` run at `1421.2058 MHz`, `-35 dBm`; stop if peaks approach clipping.

QC rule during runs:
- If `max(|ADC|) > 100`, reduce power by `3 dB`.
- If `max(|ADC|) < 20`, increase power by `3 dB`.

Per-run metadata to log:
- UTC timestamp
- Pointing (zenith/horizontal)
- Generator state/frequency/power
- SDR settings
- Quick notes (`tone seen?`, `clipping?`, `dropouts?`)
