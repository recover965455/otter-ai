#!/usr/bin/env bash
# sync_readme_version.sh — Keep README.md in sync with Cargo.toml metadata.
#
#   • Manually :  bash scripts/sync_readme_version.sh
#   • Check    :  bash scripts/sync_readme_version.sh --check
#   • Hook     :  see `.githooks/pre-commit` for installation
#
# Idempotent (safe to re-run). Zero external dependencies beyond bash +
# python3 + awk (standard on macOS / Linux / GHA runners).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CARGO_FILE="${CARGO_FILE:-Cargo.toml}"
README_FILE="${README_FILE:-README.md}"
MODE="${1:-apply}"

[[ -f "$CARGO_FILE"  ]] || { echo "❌ $CARGO_FILE  missing"  >&2; exit 2; }
[[ -f "$README_FILE" ]] || { echo "❌ $README_FILE missing"  >&2; exit 2; }

# 1) parse package.version / package.rust-version / package.repository from [package] only
read_kv() {
  awk -v k="$1" '
    /^\[package\]/  { in_pkg = 1 ; next }
    /^\[/           { in_pkg = 0 }
    in_pkg && match($0, "^[[:space:]]*" k "[[:space:]]*=[[:space:]]*\"[^\"]+\"") {
      line = $0
      sub(/^[^"]*"/, "", line)
      sub(/"[^"]*$/, "", line)
      print line
      exit
    }
  ' "$CARGO_FILE"
}

CRATE_VERSION="$(read_kv version)"
MSRV="$(          read_kv rust-version)"
REPO_URL="$(      read_kv repository)"
[[ -n "$CRATE_VERSION" && -n "$MSRV" ]] || { echo "❌ Cargo.toml missing version/rust-version" >&2; exit 3; }
CRATE_MAJ_MIN="${CRATE_VERSION%.*}"

echo "📦 Cargo.toml → version=$CRATE_VERSION (prefix=$CRATE_MAJ_MIN)  rust-version=$MSRV  repo=${REPO_URL:-(n/a)}"

# 2) apply rewrites via python3 — single pass, handles all four literal spots
#    AND the marker block at the top (marker block serves as single anchor CI
#    can grep for, the literal rewrites below are belt-and-suspenders for
#    manual readers of the README).
python3 - "$README_FILE" "$CRATE_VERSION" "$CRATE_MAJ_MIN" "$MSRV" "$REPO_URL" <<'PY'
import pathlib, re, sys
path, version, maj_min, msrv, repo = sys.argv[1:6]
text = pathlib.Path(path).read_text(encoding="utf-8")

# (A) Marker block → rendered badge line (idempotent via regex capture)
marker_repl = (
    f"<!-- sync:version:BEGIN -->"
    f" 📦 {version} · 🦀 Rust ≥ {msrv} · Source: {repo} "
    f"<!-- sync:version:END -->"
)
pattern = re.compile(
    r"<!--\s*sync:version:BEGIN\s*-->.*?<!--\s*sync:version:END\s*-->",
    flags=re.DOTALL,
)
text, n_markers = pattern.subn(marker_repl, text)
print(f"   · marker blocks synced: {n_markers}")

# (B) Rust version line:  `Rust **X.YY**  or higher`
text, n_msrv = re.subn(r"Rust \*\*[0-9]+\.[0-9]+\*\*", f"Rust **{msrv}**", text)

# (C) Two dependency forms in code fences:
#       otter-ai = "0.1"           (standalone line)
#       otter-ai = { version = "0.1", default-features = false, ... }
text, n_dep_simple = re.subn(
    r'^(\s*otter-ai\s*=\s*)"[0-9]+\.[0-9]+"\s*$',
    rf'\1"{maj_min}"',
    text,
    flags=re.MULTILINE,
)
text, n_dep_inline = re.subn(
    r'(otter-ai\s*=\s*\{[^}]*?version\s*=\s*)"[0-9]+\.[0-9]+"',
    rf'\1"{maj_min}"',
    text,
)
print(
    f"   · inline rewrites: msrv={n_msrv}  dep-simple={n_dep_simple}  "
    f"dep-inline-struct={n_dep_inline}"
)
pathlib.Path(path).write_text(text, encoding="utf-8")
PY

# 3) --check mode: diff README (must be run against a clean index to be
#    meaningful; in CI it's always clean because we just cloned)
if [[ "$MODE" == "--check" || "$MODE" == "check" ]]; then
  if git diff --quiet -- "$README_FILE" 2>/dev/null; then
    echo "✅ README.md ↔ Cargo.toml: in sync"
    exit 0
  fi
  echo "❌ README.md is OUT OF SYNC with Cargo.toml " \
       "(version=$CRATE_VERSION, rust-version=$MSRV)" >&2
  echo "   Run:    bash scripts/sync_readme_version.sh   then commit README.md." >&2
  git --no-pager diff -- "$README_FILE" || true
  exit 1
fi

echo "✅ Done. (run with --check to assert; install pre-commit hook via README)"
