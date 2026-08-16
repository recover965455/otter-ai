#!/usr/bin/env bash
# scripts/install_hooks.sh — enable the repository git hooks.
#
#   bash scripts/install_hooks.sh
#
# Makes `git commit` run `.githooks/pre-commit` (README sync + fmt +
# clippy gate). Re-run after cloning on a new machine.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

chmod +x .githooks/pre-commit 2>/dev/null || true
git config core.hooksPath .githooks

echo "✅ git hooks installed (core.hooksPath = .githooks)"
echo "   pre-commit: README version sync + cargo fmt --check + clippy -D warnings"
echo "   快速迭代可跳过 clippy： SKIP_CLIPPY=1 git commit"
