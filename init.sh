#!/bin/bash
set -e

echo "=== Harness Initialization ==="

echo "=== Checking uv environment ==="
uv run python --version

echo "=== python -m compileall . ==="
uv run python -m compileall . -q

echo "=== python -m pytest ==="
if [ -d "tests" ] || find . -path ./.venv -prune -o -name "test_*.py" -print -o -name "*_test.py" -print 2>/dev/null | grep -q .; then
  uv run python -m pytest
else
  echo "No tests found yet — skipping pytest (add tests/ as features are implemented)"
fi

echo "=== Verification Complete ==="
echo ""
echo "Next steps:"
echo "1. Read feature_list.json to see current feature state"
echo "2. Pick ONE unfinished feature to work on"
echo "3. Implement only that feature"
echo "4. Re-run verification before claiming done"
