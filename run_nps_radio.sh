#!/bin/bash

# Move to directory or fail loudly
cd ~/do-not-touch/ay-121 || { echo "Directory not found"; exit 1; }

echo "Checking for updates..."
git pull

# Run from labs/04/ so OUTPUT_DIR='data/{nps,main}' and artifacts/ paths resolve
cd labs/04 || { echo "labs/04 not found"; exit 1; }

fmt_elapsed() {
    local s=$1
    printf '%dh%02dm%02ds' $((s/3600)) $(((s%3600)/60)) $((s%60))
}

T0=$(date +%s)
echo "=== Launch: $(date -u +'%Y-%m-%dT%H:%M:%SZ') (t=0) ==="

# Stage 1: NPS run, capped at ~7 h so the galactic-plane loop launches
# when the planner's preferred start has rotated to the low-l edge
# (l ~ -8, b = +2) -- the opposite end of the survey from the typical
# mid-l (l ~ 95) start.  This maximises forecast cell coverage for the
# subsequent main run (~137 cells vs ~109 at the previous 4.5 h cap).
# nps.py runs main() once and exits; `timeout` enforces the cap if main()
# hasn't returned by then.
T1=$(date +%s)
echo "=== Stage 1: NPS (up to ~7 h, until gal-plane low-l edge rises) ==="
echo "  start:  $(date -u -d "@$T1" +'%Y-%m-%dT%H:%M:%SZ')  (t+$(fmt_elapsed $((T1-T0))))"
timeout 25200 env PYTHONPATH=../.. python3 scripts/main/nps.py
T1_END=$(date +%s)
echo "  finish: $(date -u -d "@$T1_END" +'%Y-%m-%dT%H:%M:%SZ')  (t+$(fmt_elapsed $((T1_END-T0))), stage 1 took $(fmt_elapsed $((T1_END-T1))))"

# Stage 2: galactic-plane loop (scripts/main/main wraps main() in `while True`).
T2=$(date +%s)
echo "=== Stage 2: galactic-plane loop ==="
echo "  start:  $(date -u -d "@$T2" +'%Y-%m-%dT%H:%M:%SZ')  (t+$(fmt_elapsed $((T2-T0))))"
PYTHONPATH=../.. python3 scripts/main/main
T2_END=$(date +%s)
echo "  finish: $(date -u -d "@$T2_END" +'%Y-%m-%dT%H:%M:%SZ')  (t+$(fmt_elapsed $((T2_END-T0))), stage 2 took $(fmt_elapsed $((T2_END-T2))))"

echo "=== Total runtime: $(fmt_elapsed $((T2_END-T0))) ==="
