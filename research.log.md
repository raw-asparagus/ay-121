## 2026-02-11 - RTL-SDR Sampling Guidance for `lab_bighorn`

For the 21-cm `lab_bighorn` workflow, use `2.4–2.56 Msps` as the main target.

Repository-backed rationale:
- `lab_bighorn/bighorn.tex:577` uses off-line baseline channels out to about `±1.2 MHz`, requiring roughly `fs >= 2.4 Msps`.
- `lab_bighorn/cal_intensity.tex:182` and `lab_bighorn/cal_intensity.tex:238` discuss `±1.25 MHz` / `2.5 MHz` segments, matching `2.56 Msps` well.
- `ugradio_code/src/sdr.py:74` defaults to `2.2e6`, but that gives only `±1.1 MHz` Nyquist headroom.
- `lab_mixers/allmixers.tex:274` indicates practical operation over `1.0–3.2 MHz`.

Recommended sample-rate choices:
1. `2.56e6` (preferred if stable): best fit to `±1.2 to ±1.25 MHz` analysis windows.
2. `2.40e6` (robust default): workable, but keep baseline windows away from band edges.
3. `2.048e6` or `2.2e6` only if throughput stability requires it.
4. `3.2e6` only after confirming no dropped samples on RPi/USB.

Other parameters to tune:
- Use `direct=False` for HI observations (IQ/tuner mode), not direct-sampling mode (`ugradio_code/src/sdr.py:81`).
- Set LO (`center_freq`) to move HI away from DC spike and use two-frequency switching (`lab_bighorn/bighorn.tex:286`, `lab_bighorn/bighorn.tex:395`).
- Adjust gain by checking sample histogram to avoid clipping and heavy quantization (`lab_bighorn/bighorn.tex:330`).
- Choose `nsamples` / `nblocks` for resolution vs stability, save in chunks, and add exception handling (`lab_bighorn/bighorn.tex:303`, `lab_bighorn/bighorn.tex:403`).
- Use USB-3 and verify sustained streaming (`lab_bighorn/bighorn.tex:296`).

## Observation time

- Consider planning observations ~new moon +- 6 days LST~19h-6h
