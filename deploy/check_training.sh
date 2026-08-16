#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEFAULT_REGION="asia-southeast1"
readonly DEFAULT_JOB="gomoku-zero-production"

usage() {
  cat <<'EOF'
Inspect a Vertex AI production training job without mutating it.

Usage:
  deploy/check_training.sh --project-id PROJECT_ID [options]

Required:
  --project-id ID      Google Cloud project. Never inferred from gcloud config.

Options:
  --region REGION      Vertex region (default: asia-southeast1).
  --job NAME_OR_ID     Resource name/id or display name; latest exact display-name
                       match is used (default: gomoku-zero-production).
  --logs COUNT         Include the newest Cloud Logging entries (default: 30; 0 disables).
  --help               Show this help.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-$DEFAULT_REGION}"
JOB="${JOB:-$DEFAULT_JOB}"
LOG_LIMIT="${LOG_LIMIT:-30}"

while (($#)); do
  case "$1" in
    --project-id) need_value "$@"; PROJECT_ID="$2"; shift 2 ;;
    --region) need_value "$@"; REGION="$2"; shift 2 ;;
    --job) need_value "$@"; JOB="$2"; shift 2 ;;
    --logs) need_value "$@"; LOG_LIMIT="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$PROJECT_ID" ]] || die "--project-id is required"
[[ "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "invalid project id: $PROJECT_ID"
[[ "$REGION" =~ ^[a-z][a-z0-9-]+[a-z0-9]$ ]] || die "invalid region: $REGION"
[[ "$LOG_LIMIT" =~ ^[0-9]+$ ]] && ((LOG_LIMIT <= 1000)) || die "logs must be 0..1000"
command -v gcloud >/dev/null 2>&1 || die "required command not found: gcloud"
command -v python3 >/dev/null 2>&1 || die "required command not found: python3"

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' --limit=1)"
[[ -n "$ACTIVE_ACCOUNT" ]] || die "gcloud is not authenticated; run: gcloud auth login"

if [[ "$JOB" == projects/*/locations/*/customJobs/* ]]; then
  JOB_ID="${JOB##*/}"
elif [[ "$JOB" =~ ^[0-9]+$ ]]; then
  JOB_ID="$JOB"
else
  JOB_RESOURCE="$(gcloud ai custom-jobs list \
    --region="$REGION" \
    --project="$PROJECT_ID" \
    --format=json | python3 -c '
import json, sys
display_name = sys.argv[1]
jobs = [item for item in json.load(sys.stdin) if item.get("displayName") == display_name]
jobs.sort(key=lambda item: item.get("createTime", ""), reverse=True)
print(jobs[0].get("name", "") if jobs else "")
' "$JOB")"
  [[ -n "$JOB_RESOURCE" ]] || die "no Vertex custom job found with display name: $JOB"
  JOB_ID="${JOB_RESOURCE##*/}"
fi

DETAIL_FILE="$(mktemp "${TMPDIR:-/tmp}/gomoku-job.XXXXXX.json")"
trap 'rm -f "$DETAIL_FILE"' EXIT
gcloud ai custom-jobs describe "$JOB_ID" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format=json >"$DETAIL_FILE"

STATE="$(python3 - "$DETAIL_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    job = json.load(handle)
print(job.get("state", "JOB_STATE_UNSPECIFIED"))
PY
)"

python3 - "$DETAIL_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    job = json.load(handle)

spec = job.get("jobSpec", {})
workers = spec.get("workerPoolSpecs", [])
worker = workers[0] if workers else {}
machine = worker.get("machineSpec", {})
disk = worker.get("diskSpec", {})
output = spec.get("baseOutputDirectory", {}).get("outputUriPrefix", "-")
scheduling = spec.get("scheduling", {})

print("Vertex AI custom training status")
print(f"  resource:        {job.get('name', '-')}")
print(f"  display name:    {job.get('displayName', '-')}")
print(f"  state:           {job.get('state', '-')}")
print(f"  created:         {job.get('createTime', '-')}")
print(f"  started:         {job.get('startTime', '-')}")
print(f"  ended:           {job.get('endTime', '-')}")
print(f"  machine:         {machine.get('machineType', '-')}")
if machine.get("acceleratorType"):
    print("  accelerator:     "
          f"{machine.get('acceleratorCount', 0)} x {machine['acceleratorType']}")
else:
    print("  accelerator:     none")
print("  boot disk:       "
      f"{disk.get('bootDiskSizeGb', '-')} GB {disk.get('bootDiskType', '-')}")
print(f"  runtime bound:   {scheduling.get('timeout', '-')}")
print(f"  output prefix:   {output}")
labels = job.get("labels", {})
if labels:
    print("  labels:          " + ", ".join(f"{k}={v}" for k, v in sorted(labels.items())))
error = job.get("error") or {}
if error:
    print(f"  error:           code={error.get('code', '-')} {error.get('message', '-')}")
PY

printf '  console:         https://console.cloud.google.com/vertex-ai/locations/%s/training/%s?project=%s\n' \
  "$REGION" "$JOB_ID" "$PROJECT_ID"

if ((LOG_LIMIT > 0)); then
  printf '\nNewest training logs (up to %s):\n' "$LOG_LIMIT"
  gcloud logging read \
    "resource.type=\"ml_job\" AND resource.labels.job_id=\"${JOB_ID}\"" \
    --project="$PROJECT_ID" \
    --freshness=30d \
    --limit="$LOG_LIMIT" \
    --order=desc \
    --format='table(timestamp,severity,textPayload,jsonPayload.message)' || \
    printf 'No readable training logs were returned.\n' >&2
fi

case "$STATE" in
  JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_EXPIRED)
    exit 1
    ;;
esac
