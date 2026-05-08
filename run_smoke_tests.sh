#!/bin/bash
# run_smoke_tests.sh
# Run smoke tests for stationarity.py, cointegration.py, signals.py
# Usage: bash run_smoke_tests.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON=".venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "❌ .venv not found. Run: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SMOKE TEST RUNNER"
echo "  StatisticalArbitrage — stats + signals modules"
echo "════════════════════════════════════════════════════════════════"

PASS=0
FAIL=0

run_test() {
    local label="$1"
    local module="$2"
    echo ""
    echo "▶  $label"
    echo "────────────────────────────────────────────────────────────────"
    if $PYTHON -m "$module"; then
        echo "✅  $label: PASSED"
        PASS=$((PASS + 1))
    else
        echo "❌  $label: FAILED"
        FAIL=$((FAIL + 1))
    fi
}

run_test "stationarity.py" "src.stats.stationarity"
run_test "cointegration.py" "src.stats.cointegration"
run_test "signals.py"       "src.signals"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -eq 0 ]; then
    echo "  🎉 All smoke tests passed!"
else
    echo "  ⚠️  Some tests failed — check output above."
fi
echo "════════════════════════════════════════════════════════════════"
echo ""
