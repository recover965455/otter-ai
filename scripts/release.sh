#!/usr/bin/env bash
# scripts/release.sh — atomic crate release flow.
#
#   bash scripts/release.sh [patch|minor|major|X.Y.Z]
#
# Runs (in order):
#   1. git worktree clean check
#   2. bump Cargo.toml / Cargo.lock version (next = crates.io latest +1
#      when no argument is given; fails fast if the version was already
#      published)
#   3. sync README version markers (scripts/sync_readme_version.sh)
#   4. cargo fmt --check / clippy -D warnings / cargo test --all-features
#   5. cargo package verify
#   6. commit "chore(release): bump to X.Y.Z" + tag vX.Y.Z
#   7. push commit + tag, then cargo publish
#
# Requires a GitHub PAT (scripts/install_hooks.sh does not cover push
# auth) and CARGO_REGISTRY_TOKEN for crates.io. Both are read from the
# environment when set:
#   GH_TOKEN / GITHUB_TOKEN     — git push auth
#   CARGO_REGISTRY_TOKEN        — cargo publish auth
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# --- helpers ---------------------------------------------------------------
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
  ' "$ROOT/Cargo.toml"
}

CURRENT="$(read_kv version)"
echo "📦 current version: $CURRENT"

# --- 1. worktree clean ------------------------------------------------------
if [[ -n "$(git status --porcelain)" ]]; then
  echo "❌ working tree is not clean; commit or stash first" >&2
  exit 1
fi

# --- 2. next version --------------------------------------------------------
LATEST="$(curl -sL "https://crates.io/api/v1/crates/otter-ai" -H "User-Agent: otter-ai-release" \
  | python3 -c "import json,sys; print(json.load(sys.stdin).get('crate',{}).get('max_version',''))" 2>/dev/null || true)"
echo "🌐 latest on crates.io: ${LATEST:-unknown}"

bump() {
  python3 - "$CURRENT" "$1" <<'PY'
import sys
cur = sys.argv[1]
part = sys.argv[2]
if "." in part:
    print(part)
    sys.exit(0)
maj, minor, patch = (int(x) for x in cur.split(".")[:3])
if part == "major":
    print(f"{maj + 1}.0.0")
elif part == "minor":
    print(f"{maj}.{minor + 1}.0")
else:
    print(f"{maj}.{minor}.{patch + 1}")
PY
}

if [[ "$#" -ge 1 ]]; then
  NEXT="$(bump "" "$1")"
else
  if [[ -z "$LATEST" ]]; then
    echo "❌ cannot determine latest crates.io version; pass the version explicitly" >&2
    exit 1
  fi
  if [[ "$LATEST" == "$CURRENT" ]]; then
    NEXT="$(bump "" patch)"
  else
    NEXT="$CURRENT"   # already ahead of crates.io; keep and publish
  fi
fi

if [[ -z "$NEXT" || ! "$NEXT" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "❌ invalid next version: $NEXT" >&2
  exit 1
fi
if [[ "$NEXT" == "$LATEST" ]]; then
  echo "❌ $NEXT is already published on crates.io; bump further" >&2
  exit 1
fi
echo "🚀 releasing $NEXT"

# --- 3. bump Cargo.toml / Cargo.lock ---------------------------------------
python3 - "$NEXT" <<'PY'
import pathlib, sys
next_ver = sys.argv[1]
for f in ("Cargo.toml", "Cargo.lock"):
    p = pathlib.Path(f)
    t = p.read_text(encoding="utf-8")
    if f == "Cargo.toml":
        import re
        t2 = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{next_ver}"', t, count=1, flags=re.M)
        assert t2 != t, "Cargo.toml version not replaced"
    else:
        t2 = t.replace('name = "otter-ai"\nversion = ', f'name = "otter-ai"\nversion = ', 1)
        import re
        t2 = re.sub(r'(name = "otter-ai"\nversion = ")[^"]+(")', rf'\g<1>{next_ver}\g<2>', t, count=1)
        assert t2 != t, "Cargo.lock version not replaced"
    p.write_text(t2, encoding="utf-8")
print("✅ version files bumped")
PY

# --- 4. README sync + quality gates -----------------------------------------
bash scripts/sync_readme_version.sh
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo package --allow-dirty

# --- 5. commit + tag --------------------------------------------------------
git add Cargo.toml Cargo.lock README.md
git commit -m "chore(release): bump to $NEXT"
git tag "v$NEXT"

# --- 6. push + publish ------------------------------------------------------
REMOTE="origin"
BRANCH="$(git branch --show-current)"
if [[ -n "${GH_TOKEN:-}" || -n "${GITHUB_TOKEN:-}" ]]; then
  TOKEN="${GH_TOKEN:-${GITHUB_TOKEN}}"
  REPO="$(git remote get-url origin | sed -E 's#.*github.com[:/]([^/]+/[^/]+)(\.git)?$#\1#')"
  git -c "credential.helper=!f() { echo username=x-access-token; echo password=$TOKEN; }; f" \
      push "$REMOTE" "HEAD:$BRANCH"
  git -c "credential.helper=!f() { echo username=x-access-token; echo password=$TOKEN; }; f" \
      push "$REMOTE" "v$NEXT"
else
  echo "⚠️  no GH_TOKEN/GITHUB_TOKEN set; push manually:"
  echo "    git push $REMOTE $BRANCH v$NEXT"
fi

if [[ -n "${CARGO_REGISTRY_TOKEN:-}" ]]; then
  CARGO_REGISTRY_TOKEN="$CARGO_REGISTRY_TOKEN" cargo publish --allow-dirty
else
  echo "⚠️  no CARGO_REGISTRY_TOKEN set; publish manually:"
  echo "    cargo publish --allow-dirty"
fi

echo "✅ released $NEXT"
