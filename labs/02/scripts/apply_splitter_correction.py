"""
Apply ZFSC-2-372-S+ splitter and attenuator path corrections to
labs/02/equipment_calibration.ipynb.

Run with:  .venv/bin/python labs/02/scripts/apply_splitter_correction.py
"""

from pathlib import Path
import nbformat

NB_PATH = Path(__file__).parents[1] / 'equipment_calibration.ipynb'


def make_code_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


def make_md_cell(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def main():
    nb = nbformat.read(NB_PATH, as_version=4)
    cells = nb.cells

    # ── Print index/id map for verification ──────────────────────────────────
    for i, c in enumerate(cells):
        print(f'  [{i:2d}] id={c.get("id",""):30s} type={c.cell_type}  '
              f'{c.source[:60].replace(chr(10)," ")!r}')

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Cell 2 (index 2, id=ec01-imports) — append splitter constants
    # ─────────────────────────────────────────────────────────────────────────
    SPLITTER_CONSTANTS = """
# ── ZFSC-2-372-S+ splitter measured insertion losses ─────────────────────────
# Measured at two frequencies; linearly interpolated to SIGGEN_FREQ_MHZ.
SIGGEN_FREQ_MHZ          = 1420.405751768   # 21-cm line (from manifest)

SPLITTER_S1_DB_1200MHZ   = 3.39   # S → Port 1 (power meter branch) at 1200 MHz
SPLITTER_S1_DB_1600MHZ   = 3.36   # S → Port 1                       at 1600 MHz
SPLITTER_S2_DB_1200MHZ   = 3.37   # S → Port 2 (SDR branch)          at 1200 MHz
SPLITTER_S2_DB_1600MHZ   = 3.33   # S → Port 2                       at 1600 MHz

_t = (SIGGEN_FREQ_MHZ - 1200.0) / (1600.0 - 1200.0)
SPLITTER_S1_DB = SPLITTER_S1_DB_1200MHZ + _t * (SPLITTER_S1_DB_1600MHZ - SPLITTER_S1_DB_1200MHZ)
SPLITTER_S2_DB = SPLITTER_S2_DB_1200MHZ + _t * (SPLITTER_S2_DB_1600MHZ - SPLITTER_S2_DB_1200MHZ)
del _t

ATTENUATOR_SDR_DB   = 3.0              # fixed 3 dB attenuator on port 2 → SDR
PORT2_CABLE_LEN_M   = 6 * 0.3048      # 6-ft cable between port 2 and attenuator

print(f'Splitter S→Port1 @ {SIGGEN_FREQ_MHZ:.3f} MHz : {SPLITTER_S1_DB:.4f} dB')
print(f'Splitter S→Port2 @ {SIGGEN_FREQ_MHZ:.3f} MHz : {SPLITTER_S2_DB:.4f} dB')
print(f'SDR attenuator                            : {ATTENUATOR_SDR_DB:.1f} dB')
print(f'Port-2 cable length                       : {PORT2_CABLE_LEN_M:.4f} m ({PORT2_CABLE_LEN_M/0.3048:.0f} ft)')"""

    idx2 = next(i for i, c in enumerate(cells) if c.get('id') == 'ec01-imports')
    cells[idx2].source += SPLITTER_CONSTANTS
    print(f'\n[OK] Appended splitter constants to Cell {idx2} (ec01-imports)')

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Cell 1 (index 1, id=702c15c80fe45119) — update physical assumptions
    # ─────────────────────────────────────────────────────────────────────────
    SPLITTER_MD_BULLETS = (
        '\n- ZFSC-2-372-S+ splitter: measured S→Port1 and S→Port2 insertion losses '
        'at 1200/1600 MHz, linearly interpolated to `SIGGEN_FREQ_MHZ`.\n'
        '- Meter branch: port 1 of splitter (loss = `SPLITTER_S1_DB`).\n'
        '- SDR branch: port 2 of splitter (`SPLITTER_S2_DB`) + 6-ft cable '
        '(`PORT2_CABLE_LEN_M`) + 3 dB attenuator (`ATTENUATOR_SDR_DB`).'
    )

    idx1 = next(i for i, c in enumerate(cells) if c.get('id') == '702c15c80fe45119')
    # Append after the existing bullet list
    cells[idx1].source += SPLITTER_MD_BULLETS
    print(f'[OK] Updated physical assumptions in Cell {idx1} (702c15c80fe45119)')

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Insert new markdown cell after Cell 6 (ec03-data) — path correction rationale
    # ─────────────────────────────────────────────────────────────────────────
    PATH_CORRECTION_MD = """\
## Path correction rationale

### Hardware path (calibration setup)

```
siggen → reference cable → ZFSC-2-372-S+ (port S)
  ├── Port 1 → power meter
  └── Port 2 → 6-ft cable (60-ohm) → 3 dB attenuator → SDR
```

### Correction formulas

All observables are normalised by `P_siggen_dBm` (source amplitude). The splitter
and attenuator add fixed dB losses that offset the intercepts but **leave α unchanged**
(they are cable-length-independent constants):

- **Meter branch:**
  `power_meter_corrected_db = power_meter_norm_db + SPLITTER_S1_DB`

- **SDR branch (fixed part):**
  `y_SDR_corrected = y_SDR_norm + SPLITTER_S2_DB + ATTENUATOR_SDR_DB`

- **SDR branch (deferred — port-2 cable):**
  The 6-ft cable between port 2 and the 3 dB attenuator adds `α × PORT2_CABLE_LEN_M` dB.
  This term is computed and added **after α is known** from the fit.

### Why α is unaffected

α is estimated from the *slope* of power vs. cable length. The splitter/attenuator
losses are constants that shift all observations equally, affecting only the intercepts
B₁₄₂₀, B₁₄₂₁, B_meter — not their mutual differences used to extract α.

### Why L_unknown inference is unaffected

The unknown-length measurement uses the **same hardware path** as the calibration
measurements. The path corrections therefore apply identically to both sides of the
inversion formula:

$$
L = \\frac{B - y^{\\rm obs}}{\\alpha}
$$

and cancel exactly. The inferred L_unknown is numerically identical with or without
the path corrections.
"""

    idx6 = next(i for i, c in enumerate(cells) if c.get('id') == 'ec03-data')
    path_corr_cell = make_md_cell(PATH_CORRECTION_MD)
    cells.insert(idx6 + 1, path_corr_cell)
    print(f'[OK] Inserted path-correction markdown after Cell {idx6} (ec03-data)')

    # After insertion, indices shift by 1 for all cells after idx6.
    # Recompute all subsequent indices from scratch using cell ids.

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Update Cell 6 (ec03-data) — add corrected power columns
    # ─────────────────────────────────────────────────────────────────────────
    PATH_CORRECTION_CODE = """
# ── Path corrections (splitter + attenuator) ──────────────────────────────────
# SDR correction: undo port-2 splitter loss and 3 dB attenuator.
# (The 6-ft port-2 cable is deferred to a post-fit correction using calibrated alpha.)
SDR_FIXED_CORRECTION_DB = SPLITTER_S2_DB + ATTENUATOR_SDR_DB
for df in [df_att_used, df_unk]:
    df['y_lo1420_corrected_db'] = df['y_lo1420_db'] + SDR_FIXED_CORRECTION_DB
    df['y_lo1421_corrected_db'] = df['y_lo1421_db'] + SDR_FIXED_CORRECTION_DB
    df['power_meter_corrected_db'] = df['power_meter_norm_db'] + SPLITTER_S1_DB

# Extract corrected arrays for fitting
y1420_corr = df_att_used['y_lo1420_corrected_db'].values
y1421_corr = df_att_used['y_lo1421_corrected_db'].values
meter_corr = df_att_used['power_meter_corrected_db'].values

# Print correction summary
print(f'SDR path fixed correction  : +{SDR_FIXED_CORRECTION_DB:.4f} dB'
      f'  (splitter S2={SPLITTER_S2_DB:.4f} + att={ATTENUATOR_SDR_DB:.1f})')
print(f'Meter path correction      : +{SPLITTER_S1_DB:.4f} dB  (splitter S1)')
print(f'Port-2 cable correction    : deferred (needs alpha; = alpha × {PORT2_CABLE_LEN_M:.4f} m)')"""

    # ec03-data is still at idx6 (we inserted after it, not before)
    cells[idx6].source += PATH_CORRECTION_CODE
    print(f'[OK] Appended path correction code to Cell {idx6} (ec03-data)')

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Insert post-fit port-2 cable correction cell after ec07-meter (fit_meter)
    # ─────────────────────────────────────────────────────────────────────────
    POST_FIT_CODE = """\
# ── Port-2 cable deferred correction ────────────────────────────────────────
# The 6-ft cable between splitter port 2 and the 3 dB attenuator adds:
PORT2_CABLE_LOSS_DB = fit_lin['alpha'] * PORT2_CABLE_LEN_M
y1420_fully_corrected = y1420_corr + PORT2_CABLE_LOSS_DB
y1421_fully_corrected = y1421_corr + PORT2_CABLE_LOSS_DB
# Note: this doesn't change alpha; the fit intercepts shift by PORT2_CABLE_LOSS_DB.

print(f'alpha (SDR linear, unchanged)       : {fit_lin[\"alpha\"]:.6f} dB/m')
print(f'Port-2 cable deferred correction    : +{PORT2_CABLE_LOSS_DB:.4f} dB'
      f'  ({fit_lin[\"alpha\"]:.6f} dB/m × {PORT2_CABLE_LEN_M:.4f} m)')
print()
print('Corrected intercepts (fully corrected = power at cable output, before splitter):')
B1420_corr = fit_lin['B1420'] + SDR_FIXED_CORRECTION_DB + PORT2_CABLE_LOSS_DB
B1421_corr = fit_lin['B1421'] + SDR_FIXED_CORRECTION_DB + PORT2_CABLE_LOSS_DB
B_meter_corr = fit_meter['B'] + SPLITTER_S1_DB
print(f'  B1420_corrected = {B1420_corr:.4f} dB')
print(f'  B1421_corrected = {B1421_corr:.4f} dB')
print(f'  B_meter_corrected = {B_meter_corr:.4f} dB')
print()
print('Note: L_unknown inference is unchanged because the same path corrections')
print('apply identically to calibration and unknown-length sets, cancelling in')
print('L = (B - y_obs) / alpha.')
"""

    idx_meter = next(i for i, c in enumerate(cells) if c.get('id') == 'ec07-meter')
    post_fit_cell = make_code_cell(POST_FIT_CODE)
    cells.insert(idx_meter + 1, post_fit_cell)
    print(f'[OK] Inserted post-fit port-2 cable correction after Cell {idx_meter} (ec07-meter)')

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Update Cell 18 (533d50f1) — add splitter/attenuator rows to chain_template
    # ─────────────────────────────────────────────────────────────────────────
    CHAIN_NEW_ROWS = """\

# ── ZFSC-2-372-S+ splitter and SDR-path attenuator ────────────────────────────
chain_template = pd.concat([chain_template, pd.DataFrame([
    {'stage': 'ZFSC-2-372-S+ (S→Port1)', 'location': 'lab', 'nominal_db': -3.5,
     'part_number': 'ZFSC-2-372-S+',
     'datasheet_gain_db': -SPLITTER_S1_DB, 'datasheet_url': ''},
    {'stage': 'ZFSC-2-372-S+ (S→Port2)', 'location': 'lab', 'nominal_db': -3.5,
     'part_number': 'ZFSC-2-372-S+',
     'datasheet_gain_db': -SPLITTER_S2_DB, 'datasheet_url': ''},
    {'stage': '3 dB attenuator (SDR path)', 'location': 'lab', 'nominal_db': -3.0,
     'part_number': '', 'datasheet_gain_db': -ATTENUATOR_SDR_DB, 'datasheet_url': ''},
    {'stage': 'Port-2 cable (6 ft)', 'location': 'lab', 'nominal_db': None,
     'part_number': '', 'datasheet_gain_db': np.nan, 'datasheet_url': ''},
])], ignore_index=True)
"""

    idx18 = next(i for i, c in enumerate(cells) if c.get('id') == '533d50f1')
    cells[idx18].source += CHAIN_NEW_ROWS
    print(f'[OK] Appended splitter/attenuator rows to Cell {idx18} (533d50f1)')

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Update Cell 17 (ffd0d8ce) — prepend signal path subsection
    # ─────────────────────────────────────────────────────────────────────────
    SIGNAL_PATH_MD = """\
## Signal path (calibration setup)

```
siggen → reference cable → ZFSC-2-372-S+ (port S)
  ├── Port 1 → power meter       [loss: SPLITTER_S1_DB ≈ 3.37 dB @ 1420 MHz]
  └── Port 2 → 6-ft cable → 3 dB attenuator → SDR
                                  [loss: SPLITTER_S2_DB ≈ 3.35 dB + 3.0 dB = 6.35 dB total fixed]
```

"""

    idx17 = next(i for i, c in enumerate(cells) if c.get('id') == 'ffd0d8ce')
    cells[idx17].source = SIGNAL_PATH_MD + cells[idx17].source
    print(f'[OK] Prepended signal path to Cell {idx17} (ffd0d8ce)')

    # ─────────────────────────────────────────────────────────────────────────
    # Write back
    # ─────────────────────────────────────────────────────────────────────────
    nbformat.write(nb, NB_PATH)
    print(f'\n[DONE] Notebook written to {NB_PATH}')
    print(f'Total cells: {len(cells)}')


if __name__ == '__main__':
    main()
