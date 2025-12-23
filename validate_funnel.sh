#!/usr/bin/env bash
set -euo pipefail

echo "🔎 Validating funnel invariants..."

FILE="address.html"

grep -q "prize_locked" "$FILE" || {
  echo "❌ prize_locked gate missing"
  exit 1
}

grep -q "sessionStorage.getItem('prize_id')" "$FILE" || {
  echo "❌ prize_id access missing"
  exit 1
}

grep -q "window.location.href.*promo.html" "$FILE" || {
  echo "❌ redirect to promo.html missing"
  exit 1
}

grep -q "Confirm Address" "$FILE" || {
  echo "❌ Confirm CTA missing"
  exit 1
}

echo "✅ Funnel invariants OK"

