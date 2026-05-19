#!/bin/bash

# Move to directory or fail loudly
cd ~/do-not-touch/ay-121 || { echo "Directory not found"; exit 1; }

echo "Checking for updates..."
git pull

# Run edge-clipped recheck with cwd at labs/04/ so OUTPUT_DIR='data/edges'
# resolves to labs/04/data/edges/ and MANIFEST_PATH resolves to
# labs/04/artifacts/edge_clipped_recheck.json
cd labs/04 || { echo "labs/04 not found"; exit 1; }
PYTHONPATH=../.. python3 scripts/main/edges.py &
OBSERVER_PID=$!
echo "Observer started with PID: $OBSERVER_PID"

while kill -0 $OBSERVER_PID 2>/dev/null; do
    # 1. We search for 'python3' without -x so we find all instances
    # 2. We exclude the Observer itself
    # 3. We exclude the grep process itself
    OTHER_PY_COUNT=$(pgrep -f python3 | grep -v "^$OBSERVER_PID$" | wc -l)

    # Optional: Uncomment the line below to see the count every second
    # echo "Other Python processes detected: $OTHER_PY_COUNT"

    # You wanted "2 or more", so we trigger if count is GREATER THAN 0
    if [ "$OTHER_PY_COUNT" -gt 1 ]; then
        echo "Threshold exceeded ($OTHER_PY_COUNT processes). Shutting down observer..."
        kill $OBSERVER_PID
        exit 0
    fi

    sleep 1
done

echo "Observer process $OBSERVER_PID has ended."
