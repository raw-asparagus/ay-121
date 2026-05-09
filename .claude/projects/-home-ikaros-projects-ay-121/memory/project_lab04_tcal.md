---
name: Lab 4 noise-diode T_cal values
description: Correct per-pol noise-diode temperatures for the Leuschner 4.5m receiver — earlier docs had pol 0 / pol 1 swapped
type: project
---

Lab 4 (HI survey) noise-diode calibration:

- pol 0: T_cal = 58 K
- pol 1: T_cal = 79 K
- average: 68.5 K

The pipeline uses pol 1 only (`corr11`), so `T_CAL_11 = 79.0` is the load-bearing value in `notebooks/main/scan_load.ipynb`.

**Why:** Earlier `CLAUDE.md` and the notebook had these flipped (pol 0 = 79, pol 1 = 58). User caught the swap on 2026-04-27 while diagnosing a session-level T_sys offset. With the wrong T_cal, gain is mis-scaled by 79/58 ≈ 1.36×, propagating directly into T_B.

**How to apply:** When touching anything that consumes `T_CAL_00` / `T_CAL_11` in lab 4 (calibration utils, notebook config, or any future pol-0 reduction), use 58 K for pol 0 and 79 K for pol 1. Do not reintroduce the swap.
