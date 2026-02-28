#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_QUERIES="$ROOT_DIR/eval/query.jsonl"
DEFAULT_PROFILE_CONFIG="$ROOT_DIR/config/ablation_profiles.yaml"
DEFAULT_OUTDIR="$ROOT_DIR/eval/runs"
DEFAULT_RUN_ID="eval_$(date -u +%Y%m%dT%H%M%SZ)"
DEFAULT_PROFILES="B0,B1,B2,B3,B4,B5,B6,B7"

print_help() {
  cat <<'EOF'
Usage:
  bash eval/run_full_eval.sh [queries_jsonl] [run_id] [profile_config] [outdir] [extra args...]

Defaults:
  queries_jsonl  = eval/query.jsonl
  run_id         = eval_<UTC timestamp>
  profile_config = config/ablation_profiles.yaml
  outdir         = eval/runs
  profiles       = B0,B1,B2,B3,B4,B5,B6,B7 (override with PROFILES env)

Example:
  bash eval/run_full_eval.sh
  PROFILES=B0,B5,B7 bash eval/run_full_eval.sh eval/query.jsonl myrun
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_help
  exit 0
fi

QUERIES="${1:-$DEFAULT_QUERIES}"
RUN_ID="${2:-$DEFAULT_RUN_ID}"
PROFILE_CONFIG="${3:-$DEFAULT_PROFILE_CONFIG}"
OUTDIR="${4:-$DEFAULT_OUTDIR}"
shift $(( $# > 4 ? 4 : $# ))
EXTRA_ARGS=("$@")

PROFILES="${PROFILES:-$DEFAULT_PROFILES}"

if [[ ! -f "$QUERIES" ]]; then
  echo "[ERROR] Queries file not found: $QUERIES" >&2
  exit 1
fi
if [[ ! -f "$PROFILE_CONFIG" ]]; then
  echo "[ERROR] Profile config not found: $PROFILE_CONFIG" >&2
  exit 1
fi

cd "$ROOT_DIR"
mkdir -p "$OUTDIR"

echo "[Eval] queries=$QUERIES"
echo "[Eval] run_id=$RUN_ID"
echo "[Eval] profile_config=$PROFILE_CONFIG"
echo "[Eval] outdir=$OUTDIR"
echo "[Eval] profiles=$PROFILES"

uv run python src/eval/run_ablation.py \
  --queries "$QUERIES" \
  --profile-config "$PROFILE_CONFIG" \
  --profiles "$PROFILES" \
  --outdir "$OUTDIR" \
  --run-id "$RUN_ID" \
  "${EXTRA_ARGS[@]}"

