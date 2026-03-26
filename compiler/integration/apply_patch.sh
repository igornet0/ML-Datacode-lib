#!/usr/bin/env bash
# Apply datacode_ml_named_args.patch from the root of a DataCode git clone.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PATCH="$ROOT/compiler/integration/datacode_ml_named_args.patch"
DC="${1:-}"
if [[ -z "$DC" ]]; then
  echo "Usage: $0 /path/to/DataCode" >&2
  exit 1
fi
cd "$DC"
if git apply "$PATCH" 2>/dev/null; then
  echo "Applied with git apply."
elif patch -p1 < "$PATCH"; then
  echo "Applied with patch."
else
  echo "Apply failed (already patched?)." >&2
  exit 1
fi
