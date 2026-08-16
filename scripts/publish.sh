#!/usr/bin/env bash
# scripts/publish.sh — publish to crates.io, skipping already-published versions.
#
#   bash scripts/publish.sh
#
# `cargo publish` fails with "crate otter-ai@X.Y.Z already exists" when the
# version was already released. Publishing is idempotent, so this script
# pre-checks crates.io and skips gracefully (exit 0) instead of erroring.
#
# Requires CARGO_REGISTRY_TOKEN (or `cargo login`).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERSION="$(awk -v k=version '
  /^\[package\]/  { in_pkg = 1 ; next }
  /^\[/           { in_pkg = 0 }
  in_pkg && match($0, "^[[:space:]]*" k "[[:space:]]*=[[:space:]]*\"[^\"]+\"") {
    line = $0
    sub(/^[^"]*"/, "", line)
    sub(/"[^"]*$/, "", line)
    print line
    exit
  }
' Cargo.toml 2>/dev/null)"
LATEST="$(curl -sL "https://crates.io/api/v1/crates/otter-ai" -H "User-Agent: otter-ai-publish" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('crate',{}).get('max_version',''))" 2>/dev/null || true)"

if [[ "$VERSION" == "$LATEST" && -n "$LATEST" ]]; then
  echo "✅ otter-ai@$VERSION is already on crates.io; skipping publish."
  exit 0
fi

echo "🚀 publishing otter-ai@$VERSION (latest on crates.io: ${LATEST:-unknown})"
cargo publish --verbose --allow-dirty
