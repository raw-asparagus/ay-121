# ay-121

Use this repository with `ugradio` and `ugradiolab` installed as packages.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-build-isolation -e src/ugradio/ugradio_code
pip install --no-build-isolation -e .
```

Notes:
- `pip install --no-build-isolation -e src/ugradio/ugradio_code` installs the `ugradio` package.
- `pip install --no-build-isolation -e .` installs the local `ugradiolab` package (from `pyproject.toml`).
- `--no-build-isolation` avoids network fetches for build tools on offline RPi setups.

## Quick Import Check

```bash
python -c "import ugradio, ugradiolab; print('imports OK')"
```

## Running Scripts

From the repository root (or any directory after installation):

```bash
python labs/02/scripts/lab_2_calibration.py --help
python labs/02/scripts/lab_2_cold_cal.py --help
python labs/02/scripts/lab_2_hi_drift.py --help
python labs/test/test_siggen.py --help
```

For real hardware tests, run on the RPi with SDR + signal generator connected and
permissions to access `/dev/usbtmc0`.
