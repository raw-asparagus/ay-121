#!/bin/bash

# Move to directory or fail loudly
cd ~/do-not-touch/ay-121 || { echo "Directory not found"; exit 1; }

echo "Checking for updates..."
git pull

# Run from labs/04/ so OUTPUT_DIR='data/{nps,main}' and artifacts/ paths resolve
cd labs/04 || { echo "labs/04 not found"; exit 1; }

# Stage 1: NPS run, capped to ~1h23m so we hand off to the galactic-plane
# loop just as l=5.37, b=+6 (galactic-centre-ish, dec=-21 deg) rises above
# the 17 deg alt limit at Leuschner.  nps.py runs main() once and exits;
# `timeout` enforces the cap if main() hasn't returned by then.
echo "=== Stage 1: NPS (up to ~1h23m, until l=5.37 b=6 rises) ==="
timeout 5000 env PYTHONPATH=../.. python3 scripts/main/nps.py

# Stage 2: galactic-plane loop (radio.py wraps main() in `while True`).
echo "=== Stage 2: galactic-plane loop ==="
PYTHONPATH=../.. python3 scripts/main/radio.py
