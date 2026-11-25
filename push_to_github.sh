#!/usr/bin/env bash
# Optional convenience command. Never rename a branch, change remotes, add files,
# create commits or push a checkout that has not passed the engineering suite.
set -euo pipefail
cd -- "$(dirname -- "$0")"
[[ "$(git rev-parse --show-toplevel)" == "$(pwd -P)" ]] || { echo "Not the repository root" >&2; exit 2; }
[[ "$#" == 0 ]] || { echo "Configure the reviewed repository origin explicitly; no overrides." >&2; exit 2; }
case "$(git remote get-url --push origin)" in
  https://github.com/arivercrabcanwalk/gemini-codex.git|git@github.com:arivercrabcanwalk/gemini-codex.git) ;;
  *) echo "Unexpected push destination" >&2; exit 2;;
esac
[[ -z "$(git status --porcelain --untracked-files=normal)" ]] || { echo "Commit and review changes first" >&2; exit 2; }
branch="$(git symbolic-ref --short HEAD)"
python3 run_acceptance_pack.py
git push origin "HEAD:refs/heads/${branch}"
