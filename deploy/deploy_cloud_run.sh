#!/usr/bin/env bash
set -Eeuo pipefail

readonly DEFAULT_REGION="asia-southeast1"
readonly DEFAULT_REPOSITORY="gomoku"
readonly DEFAULT_SERVICE="gomoku-zero"
readonly DEFAULT_IMAGE_NAME="gomoku-zero"
readonly DEFAULT_TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

usage() {
  cat <<'EOF'
Build the CPU image with Cloud Build and deploy the public API/UI to Cloud Run.

Usage:
  deploy/deploy_cloud_run.sh --project-id PROJECT_ID [options]

Required:
  --project-id ID          Google Cloud project. Never inferred from gcloud config.

Options:
  --region REGION          Deployment region (default: asia-southeast1).
  --repository NAME        Artifact Registry repository (default: gomoku).
  --service NAME           Cloud Run service name (default: gomoku-zero).
  --image-name NAME        Container image name (default: gomoku-zero).
  --image-tag TAG          Image tag (default: UTC timestamp, or git SHA when committed).
  --checkpoint-uri URI     Optional gs:// URI exposed as GOMOKU_CHECKPOINT_URI.
  --service-account EMAIL  Optional Cloud Run runtime service account.
  --cpu COUNT              vCPUs per instance (default: 4).
  --memory SIZE            Memory per instance (default: 8Gi).
  --timeout SECONDS        Request timeout, at most 3600 (default: 3600).
  --concurrency COUNT      Requests per instance (default: 1).
  --min-instances COUNT    Minimum instances (default: 0).
  --max-instances COUNT    Maximum instances (default: 5).
  --help                   Show this help.

Environment overrides use the uppercase option names. TORCH_INDEX_URL may be
set for a non-default PyTorch wheel index; the default is the official CPU index.
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
SERVICE="${SERVICE:-$DEFAULT_SERVICE}"
IMAGE_NAME="${IMAGE_NAME:-$DEFAULT_IMAGE_NAME}"
IMAGE_TAG="${IMAGE_TAG:-}"
CHECKPOINT_URI="${CHECKPOINT_URI:-}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-}"
CPU="${CPU:-4}"
MEMORY="${MEMORY:-8Gi}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
CONCURRENCY="${CONCURRENCY:-1}"
MIN_INSTANCES="${MIN_INSTANCES:-0}"
MAX_INSTANCES="${MAX_INSTANCES:-5}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-$DEFAULT_TORCH_INDEX_URL}"

while (($#)); do
  case "$1" in
    --project-id) need_value "$@"; PROJECT_ID="$2"; shift 2 ;;
    --region) need_value "$@"; REGION="$2"; shift 2 ;;
    --repository) need_value "$@"; REPOSITORY="$2"; shift 2 ;;
    --service) need_value "$@"; SERVICE="$2"; shift 2 ;;
    --image-name) need_value "$@"; IMAGE_NAME="$2"; shift 2 ;;
    --image-tag) need_value "$@"; IMAGE_TAG="$2"; shift 2 ;;
    --checkpoint-uri) need_value "$@"; CHECKPOINT_URI="$2"; shift 2 ;;
    --service-account) need_value "$@"; SERVICE_ACCOUNT="$2"; shift 2 ;;
    --cpu) need_value "$@"; CPU="$2"; shift 2 ;;
    --memory) need_value "$@"; MEMORY="$2"; shift 2 ;;
    --timeout) need_value "$@"; REQUEST_TIMEOUT="$2"; shift 2 ;;
    --concurrency) need_value "$@"; CONCURRENCY="$2"; shift 2 ;;
    --min-instances) need_value "$@"; MIN_INSTANCES="$2"; shift 2 ;;
    --max-instances) need_value "$@"; MAX_INSTANCES="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$PROJECT_ID" ]] || die "--project-id is required"
[[ "$PROJECT_ID" =~ ^[a-z][a-z0-9-]{4,28}[a-z0-9]$ ]] || die "invalid project id: $PROJECT_ID"
[[ "$REGION" =~ ^[a-z][a-z0-9-]+[a-z0-9]$ ]] || die "invalid region: $REGION"
[[ "$REPOSITORY" =~ ^[a-z][a-z0-9._-]{0,62}$ ]] || die "invalid repository: $REPOSITORY"
[[ "$SERVICE" =~ ^[a-z]([a-z0-9-]{0,61}[a-z0-9])?$ ]] || die "invalid service: $SERVICE"
[[ "$IMAGE_NAME" =~ ^[a-z0-9]+([._-][a-z0-9]+)*$ ]] || die "invalid image name: $IMAGE_NAME"
[[ -z "$CHECKPOINT_URI" || "$CHECKPOINT_URI" =~ ^gs://[^/]+/.+ ]] || die "checkpoint URI must be gs://BUCKET/OBJECT"
[[ -z "$SERVICE_ACCOUNT" || "$SERVICE_ACCOUNT" =~ ^[a-z0-9][a-z0-9-]*@[a-z0-9.-]+\.iam\.gserviceaccount\.com$ ]] || die "invalid service account email"
[[ "$CPU" =~ ^[1-9][0-9]*$ ]] || die "cpu must be a positive integer"
[[ "$MEMORY" =~ ^[1-9][0-9]*(Mi|Gi)$ ]] || die "memory must look like 4096Mi or 8Gi"
[[ "$REQUEST_TIMEOUT" =~ ^[0-9]+$ ]] && ((REQUEST_TIMEOUT >= 1 && REQUEST_TIMEOUT <= 3600)) || die "timeout must be 1..3600 seconds"
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] && ((CONCURRENCY >= 1 && CONCURRENCY <= 1000)) || die "concurrency must be 1..1000"
[[ "$MIN_INSTANCES" =~ ^[0-9]+$ ]] || die "min-instances must be a non-negative integer"
[[ "$MAX_INSTANCES" =~ ^[0-9]+$ ]] && ((MAX_INSTANCES >= 1)) || die "max-instances must be positive"
((MIN_INSTANCES <= MAX_INSTANCES)) || die "min-instances cannot exceed max-instances"
[[ "$TORCH_INDEX_URL" =~ ^https:// ]] || die "TORCH_INDEX_URL must use https://"

require_command gcloud
if [[ -z "$IMAGE_TAG" ]]; then
  if command -v git >/dev/null 2>&1 \
    && git rev-parse --verify HEAD >/dev/null 2>&1 \
    && [[ -z "$(git status --porcelain --untracked-files=normal)" ]]; then
    IMAGE_TAG="$(git rev-parse --short=12 HEAD)"
  else
    IMAGE_TAG="$(date -u +%Y%m%d-%H%M%S)"
  fi
fi
[[ "$IMAGE_TAG" =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$ ]] || die "invalid image tag: $IMAGE_TAG"

ACTIVE_ACCOUNT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' --limit=1)"
[[ -n "$ACTIVE_ACCOUNT" ]] || die "gcloud is not authenticated; run: gcloud auth login"
gcloud projects describe "$PROJECT_ID" --format='value(projectId)' >/dev/null

printf 'Using account: %s\n' "$ACTIVE_ACCOUNT"
printf 'Enabling Cloud Run build/deploy APIs in %s...\n' "$PROJECT_ID"
gcloud services enable \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
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

if [[ -n "$CHECKPOINT_URI" ]]; then
  printf 'Validating configured checkpoint object %s...\n' "$CHECKPOINT_URI"
  gcloud storage objects describe "$CHECKPOINT_URI" \
    --project="$PROJECT_ID" >/dev/null
  if [[ -n "$SERVICE_ACCOUNT" ]]; then
    CHECKPOINT_BUCKET="${CHECKPOINT_URI#gs://}"
    CHECKPOINT_BUCKET="${CHECKPOINT_BUCKET%%/*}"
    gcloud storage buckets add-iam-policy-binding "gs://${CHECKPOINT_BUCKET}" \
      --member="serviceAccount:${SERVICE_ACCOUNT}" \
      --role=roles/storage.objectViewer \
      --project="$PROJECT_ID" \
      --quiet >/dev/null
  fi
fi

IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
printf 'Building %s with Cloud Build...\n' "$IMAGE_URI"
gcloud builds submit . \
  --config=cloudbuild.yaml \
  --substitutions="_REGION=${REGION},_REPOSITORY=${REPOSITORY},_IMAGE_NAME=${IMAGE_NAME},_TAG=${IMAGE_TAG},_TORCH_INDEX_URL=${TORCH_INDEX_URL}" \
  --project="$PROJECT_ID" \
  --quiet

ENV_VARS="PYTHONUNBUFFERED=1,GOMOKU_DEFAULT_CONFIG=/app/configs/production.json,GOMOKU_STATIC_DIR=/app/static"
if [[ -n "$CHECKPOINT_URI" ]]; then
  ENV_VARS+=",GOMOKU_CHECKPOINT_URI=${CHECKPOINT_URI}"
fi

DEPLOY_ARGS=(
  run deploy "$SERVICE"
  --image="$IMAGE_URI"
  --region="$REGION"
  --project="$PROJECT_ID"
  --platform=managed
  --execution-environment=gen2
  --port=8080
  --cpu="$CPU"
  --memory="$MEMORY"
  --timeout="${REQUEST_TIMEOUT}s"
  --concurrency="$CONCURRENCY"
  --min-instances="$MIN_INSTANCES"
  --max-instances="$MAX_INSTANCES"
  --cpu-boost
  --ingress=all
  --allow-unauthenticated
  --set-env-vars="$ENV_VARS"
  --labels="app=gomoku-zero,component=web"
  --quiet
)
if [[ -n "$SERVICE_ACCOUNT" ]]; then
  DEPLOY_ARGS+=(--service-account="$SERVICE_ACCOUNT")
fi

printf 'Deploying public Cloud Run service %s...\n' "$SERVICE"
gcloud "${DEPLOY_ARGS[@]}"

SERVICE_URL="$(gcloud run services describe "$SERVICE" \
  --region="$REGION" --project="$PROJECT_ID" --format='value(status.url)')"
printf '\nDeployment complete.\nImage: %s\nPublic URL: %s\n' "$IMAGE_URI" "$SERVICE_URL"
