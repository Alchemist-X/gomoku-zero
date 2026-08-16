#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEFAULT_REGION="asia-southeast1"
readonly DEFAULT_REPOSITORY="gomoku"
readonly DEFAULT_JOB="gomoku-zero-production"
readonly DEFAULT_IMAGE_NAME="gomoku-zero-training"
readonly CPU_TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
readonly DEFAULT_CONFIG_FILE="configs/production.json"

usage() {
  cat <<'EOF'
Build an image and submit bounded production training to Vertex AI Custom Jobs.

Usage:
  deploy/submit_vertex_training.sh --project-id PROJECT_ID (--yes | --plan-only) [options]

Required:
  --project-id ID          Google Cloud project. Never inferred from gcloud config.
  --yes                    Confirm the displayed charge-incurring Vertex workload
                           (not required with --plan-only).
                           VERTEX_TRAINING_CONFIRM=yes is equivalent.

Options:
  --region REGION          Vertex/Artifact Registry region (default: asia-southeast1).
  --repository NAME        Artifact Registry repository (default: gomoku).
  --job NAME               Vertex display name (default: gomoku-zero-production).
  --config PATH            Safe JSON under configs/ (default: configs/production.json).
  --image-name NAME        Built image name (default: gomoku-zero-training).
  --image-tag TAG          Built image tag (default: first 12 chars of clean Git SHA).
  --image-uri URI          Use an already-built image; implies --skip-build.
  --bucket NAME            Output bucket (default: PROJECT_ID-training).
  --output-uri URI         Unique gs:// output prefix (default: bucket/runs/RUN_ID).
  --machine-type TYPE      Worker machine (default: n1-highcpu-32 for CPU).
  --boot-disk-size-gb N    pd-ssd boot disk size (default: 500).
  --max-run-seconds N      Hard runtime bound, 3600..604800 (default: 604800).
  --sync-interval N        Checkpoint upload interval, >=30 seconds (default: 300).
  --resume-uri URI         Optional gs://.../checkpoints/latest.pt to resume from.
  --service-account EMAIL  Optional custom training service account.
  --allow-dirty-source     Explicitly permit a dirty Git worktree (not recommended).
  --plan-only              Print the validated plan and make no changes.
  --skip-build             Do not run Cloud Build (requires --image-uri).
  --yes                    Explicitly authorize build/training charges.
  --help                   Show this help.

CPU is the cost-conscious default. Optional accelerator overrides:
  ACCELERATOR_TYPE=NVIDIA_TESLA_T4
  ACCELERATOR_COUNT=1
  TORCH_INDEX_URL=https://download.pytorch.org/whl/<compatible-cuda-index>

When an accelerator is requested, TORCH_INDEX_URL must be explicitly set to a
CUDA-compatible official PyTorch wheel index. Availability and quota are regional.
EOF
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

need_value() {
  [[ $# -ge 2 && -n "${2:-}" ]] || die "$1 requires a value"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

PROJECT_ID="${PROJECT_ID:-}"
REGION="${REGION:-$DEFAULT_REGION}"
REPOSITORY="${REPOSITORY:-$DEFAULT_REPOSITORY}"
JOB="${JOB:-$DEFAULT_JOB}"
CONFIG_FILE="${CONFIG_FILE:-$DEFAULT_CONFIG_FILE}"
IMAGE_NAME="${IMAGE_NAME:-$DEFAULT_IMAGE_NAME}"
IMAGE_TAG="${IMAGE_TAG:-}"
IMAGE_URI="${IMAGE_URI:-}"
BUCKET="${BUCKET:-}"
OUTPUT_URI="${OUTPUT_URI:-}"
MACHINE_TYPE="${MACHINE_TYPE:-}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-500}"
MAX_RUN_SECONDS="${MAX_RUN_SECONDS:-604800}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"
RESUME_URI="${RESUME_URI:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-${VERTEX_ACCELERATOR_TYPE:-}}"
ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-${VERTEX_ACCELERATOR_COUNT:-}}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"
SKIP_BUILD=0
CONFIRMED=0
ALLOW_DIRTY_SOURCE=0
PLAN_ONLY=0

case "${VERTEX_TRAINING_CONFIRM:-}" in
  yes|YES|true|TRUE|1) CONFIRMED=1 ;;
esac

while (($#)); do
  case "$1" in
    --project-id) need_value "$@"; PROJECT_ID="$2"; shift 2 ;;
    --region) need_value "$@"; REGION="$2"; shift 2 ;;
    --repository) need_value "$@"; REPOSITORY="$2"; shift 2 ;;
    --job) need_value "$@"; JOB="$2"; shift 2 ;;
    --config) need_value "$@"; CONFIG_FILE="$2"; shift 2 ;;
    --image-name) need_value "$@"; IMAGE_NAME="$2"; shift 2 ;;
    --image-tag) need_value "$@"; IMAGE_TAG="$2"; shift 2 ;;
    --image-uri) need_value "$@"; IMAGE_URI="$2"; SKIP_BUILD=1; shift 2 ;;
    --bucket) need_value "$@"; BUCKET="$2"; shift 2 ;;
    --output-uri) need_value "$@"; OUTPUT_URI="$2"; shift 2 ;;
    --machine-type) need_value "$@"; MACHINE_TYPE="$2"; shift 2 ;;
    --boot-disk-size-gb) need_value "$@"; BOOT_DISK_SIZE_GB="$2"; shift 2 ;;
    --max-run-seconds) need_value "$@"; MAX_RUN_SECONDS="$2"; shift 2 ;;
    --sync-interval) need_value "$@"; SYNC_INTERVAL_SECONDS="$2"; shift 2 ;;
    --resume-uri) need_value "$@"; RESUME_URI="$2"; shift 2 ;;
    --service-account) need_value "$@"; SERVICE_ACCOUNT="$2"; shift 2 ;;
    --allow-dirty-source) ALLOW_DIRTY_SOURCE=1; shift ;;
    --plan-only) PLAN_ONLY=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --yes) CONFIRMED=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$PROJECT_ID" ]] || die "--project-id is required"
[[ "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "invalid project id: $PROJECT_ID"
[[ "$REGION" =~ ^[a-z][a-z0-9-]+[a-z0-9]$ ]] || die "invalid region: $REGION"
[[ "$REPOSITORY" =~ ^[a-z][a-z0-9._-]{0,62}$ ]] || die "invalid repository: $REPOSITORY"
[[ "$JOB" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,126}[A-Za-z0-9]$ ]] || die "invalid job display name: $JOB"
[[ "$CONFIG_FILE" =~ ^configs/[a-z0-9][a-z0-9_-]*\.json$ ]] || \
  die "--config must be a lowercase JSON filename directly under configs/"
[[ "$IMAGE_NAME" =~ ^[a-z0-9]+([._-][a-z0-9]+)*$ ]] || die "invalid image name: $IMAGE_NAME"
[[ "$BOOT_DISK_SIZE_GB" =~ ^[0-9]+$ ]] && ((BOOT_DISK_SIZE_GB >= 100 && BOOT_DISK_SIZE_GB <= 64000)) || die "boot disk must be 100..64000 GB"
[[ "$MAX_RUN_SECONDS" =~ ^[0-9]+$ ]] && ((MAX_RUN_SECONDS >= 3600 && MAX_RUN_SECONDS <= 604800)) || die "max runtime must be 3600..604800 seconds"
[[ "$SYNC_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] && ((SYNC_INTERVAL_SECONDS >= 30)) || die "sync interval must be at least 30 seconds"
[[ -z "$RESUME_URI" || "$RESUME_URI" =~ ^gs://[^/]+/.+/checkpoints/latest\.pt$ ]] || \
  die "resume URI must end with /checkpoints/latest.pt"
[[ -z "$SERVICE_ACCOUNT" || "$SERVICE_ACCOUNT" =~ ^[a-z0-9][a-z0-9-]*@[a-z0-9.-]+\.iam\.gserviceaccount\.com$ ]] || die "invalid service account email"
[[ -z "$MACHINE_TYPE" || "$MACHINE_TYPE" =~ ^[a-z0-9-]+$ ]] || die "invalid machine type"

if [[ -n "$ACCELERATOR_TYPE" ]]; then
  [[ "$ACCELERATOR_TYPE" =~ ^[A-Z][A-Z0-9_]+$ ]] || die "invalid accelerator type"
  ACCELERATOR_COUNT="${ACCELERATOR_COUNT:-1}"
  [[ "$ACCELERATOR_COUNT" =~ ^[1-9][0-9]*$ ]] || die "accelerator count must be positive"
  [[ -n "$TORCH_INDEX_URL" ]] || die "set TORCH_INDEX_URL to a CUDA-compatible PyTorch wheel index when using an accelerator"
  [[ "$TORCH_INDEX_URL" != "$CPU_TORCH_INDEX_URL" ]] || die "CPU-only TORCH_INDEX_URL cannot be used with an accelerator"
  MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-16}"
  TRAIN_DEVICE="cuda"
else
  [[ -z "$ACCELERATOR_COUNT" || "$ACCELERATOR_COUNT" == "0" ]] || die "ACCELERATOR_COUNT requires ACCELERATOR_TYPE"
  ACCELERATOR_COUNT=0
  TORCH_INDEX_URL="${TORCH_INDEX_URL:-$CPU_TORCH_INDEX_URL}"
  MACHINE_TYPE="${MACHINE_TYPE:-n1-highcpu-32}"
  TRAIN_DEVICE="cpu"
fi
[[ "$TORCH_INDEX_URL" =~ ^https:// ]] || die "TORCH_INDEX_URL must use https://"

RUN_ID="$(date -u +%Y%m%d-%H%M%S)"
SOURCE_REVISION="${GIT_SHA:-}"
SOURCE_STATE="external"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  SOURCE_REVISION="$(git rev-parse HEAD 2>/dev/null || true)"
  if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    SOURCE_STATE="dirty"
  else
    SOURCE_STATE="clean"
  fi
fi
if [[ ! "$SOURCE_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  if ((PLAN_ONLY)); then
    SOURCE_REVISION="unknown"
    SOURCE_STATE="uncommitted"
  else
    die "formal training requires a committed Git SHA; commit the repository or set GIT_SHA"
  fi
fi
if [[ "$SOURCE_STATE" == "dirty" && "$ALLOW_DIRTY_SOURCE" != "1" ]]; then
  if ((PLAN_ONLY)); then
    printf 'warning: worktree is dirty; an actual submission would be refused\n' >&2
  else
    die "formal training requires a clean Git worktree; commit changes or pass --allow-dirty-source"
  fi
fi
if [[ -z "$IMAGE_TAG" ]]; then
  IMAGE_TAG="${SOURCE_REVISION:0:12}"
  if [[ "$SOURCE_STATE" == "dirty" ]]; then
    IMAGE_TAG+="-dirty-${RUN_ID}"
  fi
fi
[[ "$IMAGE_TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || die "invalid image tag: $IMAGE_TAG"

BUCKET="${BUCKET:-${PROJECT_ID}-training}"
[[ "$BUCKET" =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ ]] || die "invalid bucket name: $BUCKET"
OUTPUT_URI="${OUTPUT_URI:-gs://${BUCKET}/runs/${RUN_ID}}"
OUTPUT_URI="${OUTPUT_URI%/}"
[[ "$OUTPUT_URI" =~ ^gs://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/.+ ]] || die "output URI must be gs://BUCKET/PREFIX"
OUTPUT_BUCKET="${OUTPUT_URI#gs://}"
OUTPUT_BUCKET="${OUTPUT_BUCKET%%/*}"

if ((SKIP_BUILD)); then
  [[ -n "$IMAGE_URI" ]] || die "--skip-build requires --image-uri"
else
  IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

[[ -f "$CONFIG_FILE" ]] || die "run this script from the repository root; missing $CONFIG_FILE"
require_command gcloud
require_command python3

CONFIG_ROOT="$(python3 -c 'from pathlib import Path; print(Path("configs").resolve())')"
CONFIG_REAL="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$CONFIG_FILE")"
[[ "$CONFIG_REAL" == "$CONFIG_ROOT"/* ]] || die "config symlink escapes the repository configs directory"
CONFIG_NAME="$(basename "$CONFIG_FILE" .json)"
CONTAINER_CONFIG="/app/${CONFIG_FILE}"
if [[ "$CONFIG_NAME" != "production" && "$JOB" == "$DEFAULT_JOB" ]]; then
  JOB="gomoku-zero-${CONFIG_NAME}"
fi

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' --limit=1)"
[[ -n "$ACTIVE_ACCOUNT" ]] || die "gcloud is not authenticated; run: gcloud auth login"
gcloud projects describe "$PROJECT_ID" --format='value(projectId)' >/dev/null

MAX_RUN_HOURS=$(((MAX_RUN_SECONDS + 3599) / 3600))
printf '\nVertex AI production training plan\n'
printf '  account:             %s\n' "$ACTIVE_ACCOUNT"
printf '  project / region:    %s / %s\n' "$PROJECT_ID" "$REGION"
printf '  job display name:    %s\n' "$JOB"
printf '  source revision:     %s (%s)\n' "$SOURCE_REVISION" "$SOURCE_STATE"
printf '  image:               %s\n' "$IMAGE_URI"
printf '  config:              %s\n' "$CONTAINER_CONFIG"
printf '  worker:              1 x %s\n' "$MACHINE_TYPE"
REQUESTED_VCPUS="${MACHINE_TYPE##*-}"
if [[ "$REQUESTED_VCPUS" =~ ^[0-9]+$ ]]; then
  printf '  requested capacity:  %s vCPU (regional Vertex training quota must be >= %s)\n' \
    "$REQUESTED_VCPUS" "$REQUESTED_VCPUS"
fi
printf '  boot disk:           %s GB pd-ssd\n' "$BOOT_DISK_SIZE_GB"
if [[ -n "$ACCELERATOR_TYPE" ]]; then
  printf '  accelerator:         %s x %s\n' "$ACCELERATOR_COUNT" "$ACCELERATOR_TYPE"
else
  printf '  accelerator:         none (CPU-optimized image)\n'
fi
printf '  hard runtime bound:  %s hours (%s machine-hours maximum)\n' "$MAX_RUN_HOURS" "$MAX_RUN_HOURS"
if [[ -n "$ACCELERATOR_TYPE" ]]; then
  printf '  accelerator bound:   %s accelerator-hours maximum\n' "$((MAX_RUN_HOURS * ACCELERATOR_COUNT))"
fi
printf '  output/checkpoints:  %s\n' "$OUTPUT_URI"
printf '  checkpoint sync:     every %s seconds and on exit\n' "$SYNC_INTERVAL_SECONDS"
if [[ -n "$RESUME_URI" ]]; then
  printf '  explicit resume:     %s\n' "$RESUME_URI"
else
  printf '  restart recovery:    %s/checkpoints/latest.pt when present\n' "$OUTPUT_URI"
fi

python3 - "$CONFIG_FILE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = json.load(handle)
model = cfg["model"]
train = cfg["training"]
evaluation = cfg["evaluation"]
print("  model depth:         "
      f"{model['channels']} channels, {model['residual_blocks']} residual blocks")
print("  training workload:   "
      f"{train['iterations']} iterations x {train['self_play_games_per_iteration']} "
      f"self-play games = {train['iterations'] * train['self_play_games_per_iteration']:,} games, "
      f"{train['self_play_actors']} actors")
print("  search depth:        "
      f"{train['mcts_simulations']} MCTS simulations/move during self-play")
print("  optimization:        "
      f"{train['iterations'] * train['training_steps_per_iteration']:,} total gradient steps, "
      f"batch {train['batch_size']}")
promotion_games = (
    train['iterations'] // train['promotion_every'] * train['promotion_games']
)
print("  promotion workload:  "
      f"{promotion_games:,} games at "
      f"{train['promotion_mcts_simulations']:,} simulations/move")
print("  frozen evaluation:   "
      f"{evaluation['games']:,} games at {evaluation['mcts_simulations']:,} simulations/move")
PY

printf '\nThis operation creates billable Cloud Build, storage, and Vertex AI resources.\n'
if ((PLAN_ONLY)); then
  printf 'Plan-only mode complete; no APIs, buckets, images, IAM policies, or jobs were changed.\n'
  exit 0
fi
if ((CONFIRMED == 0)); then
  die "submission not confirmed; re-run with --yes or VERTEX_TRAINING_CONFIRM=yes"
fi

printf 'Enabling required APIs...\n'
gcloud services enable \
  aiplatform.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com \
  serviceusage.googleapis.com \
  storage.googleapis.com \
  --project="$PROJECT_ID" \
  --quiet

if ! gcloud artifacts repositories describe "$REPOSITORY" \
  --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1; then
  printf 'Creating Artifact Registry repository %s in %s...\n' "$REPOSITORY" "$REGION"
  gcloud artifacts repositories create "$REPOSITORY" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Gomoku Zero application and training images" \
    --project="$PROJECT_ID" \
    --quiet
fi

if ! gcloud storage buckets describe "gs://${OUTPUT_BUCKET}" \
  --project="$PROJECT_ID" >/dev/null 2>&1; then
  printf 'Creating private output bucket gs://%s in %s...\n' "$OUTPUT_BUCKET" "$REGION"
  gcloud storage buckets create "gs://${OUTPUT_BUCKET}" \
    --project="$PROJECT_ID" \
    --location="$REGION" \
    --uniform-bucket-level-access \
    --public-access-prevention \
    --quiet
fi

if [[ -n "$SERVICE_ACCOUNT" ]]; then
  gcloud iam service-accounts describe "$SERVICE_ACCOUNT" \
    --project="$PROJECT_ID" >/dev/null
  gcloud storage buckets add-iam-policy-binding "gs://${OUTPUT_BUCKET}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role=roles/storage.objectAdmin \
    --project="$PROJECT_ID" \
    --quiet >/dev/null
fi

if ((SKIP_BUILD == 0)); then
  printf 'Building training image %s...\n' "$IMAGE_URI"
  gcloud builds submit . \
    --config=cloudbuild.yaml \
    --substitutions="_REGION=${REGION},_REPOSITORY=${REPOSITORY},_IMAGE_NAME=${IMAGE_NAME},_TAG=${IMAGE_TAG},_TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
    --project="$PROJECT_ID" \
    --quiet
else
  printf 'Skipping build; using %s\n' "$IMAGE_URI"
fi

PINNED_IMAGE_URI="$(gcloud artifacts docker images describe "$IMAGE_URI" \
  --project="$PROJECT_ID" \
  --format='value(image_summary.fully_qualified_digest)' 2>/dev/null || true)"
if [[ -z "$PINNED_IMAGE_URI" ]]; then
  if ((SKIP_BUILD == 0)); then
    die "built image digest could not be resolved: $IMAGE_URI"
  fi
  PINNED_IMAGE_URI="$IMAGE_URI"
  printf 'warning: existing image could not be digest-pinned; using %s\n' "$IMAGE_URI" >&2
else
  printf 'Pinned training image: %s\n' "$PINNED_IMAGE_URI"
fi

SPEC_FILE="$(mktemp "${TMPDIR:-/tmp}/gomoku-vertex-spec.XXXXXX.json")"
trap 'rm -f "$SPEC_FILE"' EXIT

python3 - "$SPEC_FILE" "$MACHINE_TYPE" "$BOOT_DISK_SIZE_GB" "$PINNED_IMAGE_URI" \
  "$OUTPUT_URI" "$MAX_RUN_SECONDS" "$TRAIN_DEVICE" "$SYNC_INTERVAL_SECONDS" \
  "$RESUME_URI" "$SERVICE_ACCOUNT" "$ACCELERATOR_TYPE" "$ACCELERATOR_COUNT" "$RUN_ID" \
  "$SOURCE_REVISION" "$CONTAINER_CONFIG" <<'PY'
import json
import sys

(
    destination,
    machine_type,
    disk_size,
    image_uri,
    output_uri,
    timeout_seconds,
    train_device,
    sync_seconds,
    resume_uri,
    service_account,
    accelerator_type,
    accelerator_count,
    run_id,
    source_revision,
    container_config,
) = sys.argv[1:]

machine_spec = {"machineType": machine_type}
if accelerator_type:
    machine_spec.update(
        acceleratorType=accelerator_type,
        acceleratorCount=int(accelerator_count),
    )

environment = [
    {"name": "TRAIN_CONFIG", "value": container_config},
    {"name": "GCS_OUTPUT_URI", "value": output_uri},
    {"name": "TRAIN_DEVICE", "value": train_device},
    {"name": "SYNC_INTERVAL_SECONDS", "value": sync_seconds},
    {"name": "GOMOKU_RUN_ID", "value": run_id},
    {"name": "ALLOW_SLOW_PRODUCTION", "value": "1"},
    {"name": "GIT_SHA", "value": source_revision},
]
if resume_uri:
    environment.append({"name": "RESUME_CHECKPOINT_URI", "value": resume_uri})

spec = {
    "workerPoolSpecs": [
        {
            "machineSpec": machine_spec,
            "diskSpec": {
                "bootDiskType": "pd-ssd",
                "bootDiskSizeGb": int(disk_size),
            },
            "replicaCount": 1,
            "containerSpec": {
                "imageUri": image_uri,
                "command": ["/app/deploy/vertex_entrypoint.sh"],
                "env": environment,
            },
        }
    ],
    "baseOutputDirectory": {"outputUriPrefix": output_uri},
    "scheduling": {
        "timeout": f"{timeout_seconds}s",
        "restartJobOnWorkerRestart": True,
    },
}
if service_account:
    spec["serviceAccount"] = service_account

with open(destination, "w", encoding="utf-8") as handle:
    json.dump(spec, handle, indent=2)
    handle.write("\n")
PY

printf 'Submitting Vertex AI Custom Job...\n'
JOB_RESOURCE="$(gcloud ai custom-jobs create \
  --display-name="$JOB" \
  --config="$SPEC_FILE" \
  --labels="app=gomoku-zero,workload=training,config=${CONFIG_NAME},run_id=${RUN_ID},source_revision=${SOURCE_REVISION},source_state=${SOURCE_STATE}" \
  --region="$REGION" \
  --project="$PROJECT_ID" \
  --format='value(name)' \
  --quiet)"

[[ -n "$JOB_RESOURCE" ]] || die "Vertex returned no job resource name"
printf '\nTraining submitted.\n'
printf 'Job resource: %s\n' "$JOB_RESOURCE"
printf 'Output prefix: %s\n' "$OUTPUT_URI"
printf 'Check status: deploy/check_training.sh --project-id %s --region %s --job %s\n' \
  "$PROJECT_ID" "$REGION" "$JOB_RESOURCE"
