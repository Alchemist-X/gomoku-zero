#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

TRAIN_CONFIG="${TRAIN_CONFIG:-/app/configs/production.json}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-/tmp/gomoku-zero-output}"
GCS_OUTPUT_URI="${GCS_OUTPUT_URI:-}"
TRAIN_DEVICE="${TRAIN_DEVICE:-cpu}"
SYNC_INTERVAL_SECONDS="${SYNC_INTERVAL_SECONDS:-300}"
RESUME_CHECKPOINT_URI="${RESUME_CHECKPOINT_URI:-}"
ALLOW_SLOW_PRODUCTION="${ALLOW_SLOW_PRODUCTION:-0}"
INITIAL_MODEL_URI="${INITIAL_MODEL_URI:-}"
INITIAL_MODEL_SHA256="${INITIAL_MODEL_SHA256:-}"
RUN_TOURNAMENT="${RUN_TOURNAMENT:-0}"
TOURNAMENT_GAMES_PER_PAIR="${TOURNAMENT_GAMES_PER_PAIR:-8}"
TOURNAMENT_SIMULATIONS="${TOURNAMENT_SIMULATIONS:-128}"
TOURNAMENT_OPENING_PLIES="${TOURNAMENT_OPENING_PLIES:-4}"
TOURNAMENT_SEED="${TOURNAMENT_SEED:-20260817}"
TOURNAMENT_MAX_BATCH_SIZE="${TOURNAMENT_MAX_BATCH_SIZE:-32}"
TOURNAMENT_BOOTSTRAP_SAMPLES="${TOURNAMENT_BOOTSTRAP_SAMPLES:-1000}"

if [[ -z "$GCS_OUTPUT_URI" && -n "${AIP_MODEL_DIR:-}" ]]; then
  GCS_OUTPUT_URI="${AIP_MODEL_DIR%/}"
  GCS_OUTPUT_URI="${GCS_OUTPUT_URI%/model}"
fi

[[ -f "$TRAIN_CONFIG" ]] || die "training config not found: $TRAIN_CONFIG"
[[ "$LOCAL_OUTPUT_DIR" == /tmp/* ]] || die "LOCAL_OUTPUT_DIR must be an absolute path below /tmp"
[[ "$GCS_OUTPUT_URI" =~ ^gs://[^/]+/.+ ]] || die "GCS_OUTPUT_URI must be gs://BUCKET/PREFIX"
[[ "$TRAIN_DEVICE" == "cpu" || "$TRAIN_DEVICE" == "cuda" ]] || die "TRAIN_DEVICE must be cpu or cuda"
[[ "$ALLOW_SLOW_PRODUCTION" == "1" ]] || die "production training requires ALLOW_SLOW_PRODUCTION=1 from the confirmed submitter"
[[ "$SYNC_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] && ((SYNC_INTERVAL_SECONDS >= 30)) || die "SYNC_INTERVAL_SECONDS must be at least 30"
[[ -z "$RESUME_CHECKPOINT_URI" || "$RESUME_CHECKPOINT_URI" =~ ^gs://[^/]+/.+ ]] || die "RESUME_CHECKPOINT_URI must be gs://BUCKET/OBJECT"
[[ -z "$INITIAL_MODEL_URI" || "$INITIAL_MODEL_URI" =~ ^gs://[^/]+/.+ ]] || die "INITIAL_MODEL_URI must be gs://BUCKET/OBJECT[#GENERATION]"
[[ -z "$INITIAL_MODEL_SHA256" || "$INITIAL_MODEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || die "INITIAL_MODEL_SHA256 must be lowercase SHA-256"
[[ "$RUN_TOURNAMENT" == "0" || "$RUN_TOURNAMENT" == "1" ]] || die "RUN_TOURNAMENT must be 0 or 1"
for value in "$TOURNAMENT_GAMES_PER_PAIR" "$TOURNAMENT_SIMULATIONS" \
  "$TOURNAMENT_OPENING_PLIES" "$TOURNAMENT_SEED" "$TOURNAMENT_MAX_BATCH_SIZE" \
  "$TOURNAMENT_BOOTSTRAP_SAMPLES"; do
  [[ "$value" =~ ^[0-9]+$ ]] || die "tournament settings must be non-negative integers"
done
((TOURNAMENT_GAMES_PER_PAIR > 0 && TOURNAMENT_GAMES_PER_PAIR % 2 == 0)) || die "TOURNAMENT_GAMES_PER_PAIR must be positive and even"
((TOURNAMENT_SIMULATIONS > 0 && TOURNAMENT_MAX_BATCH_SIZE > 0)) || die "tournament simulations and batch size must be positive"

mkdir -p "$LOCAL_OUTPUT_DIR"
cp "$TRAIN_CONFIG" "$LOCAL_OUTPUT_DIR/config.production.json"

INITIAL_MODEL_PATH=""
if [[ -n "$INITIAL_MODEL_URI" ]]; then
  FROZEN_DIR="${LOCAL_OUTPUT_DIR}/frozen"
  mkdir -p "$FROZEN_DIR"
  INITIAL_MODEL_PATH="${FROZEN_DIR}/v0000-base.pt"
  python - "$INITIAL_MODEL_URI" "$INITIAL_MODEL_PATH" "$INITIAL_MODEL_SHA256" <<'PY'
import hashlib
import pathlib
import sys

from google.cloud import storage

uri, destination, expected_digest = sys.argv[1:]
object_uri, marker, generation_text = uri.partition("#")
if marker and (not generation_text.isdigit() or int(generation_text) <= 0):
    raise SystemExit("INITIAL_MODEL_URI generation must be a positive integer")
bucket_name, separator, object_name = object_uri[5:].partition("/")
if not separator or not object_name:
    raise SystemExit(f"invalid initial model URI: {uri}")
client = storage.Client()
generation = int(generation_text) if marker else None
blob = client.bucket(bucket_name).blob(object_name, generation=generation)
target = pathlib.Path(destination)
blob.download_to_filename(
    str(target),
    checksum="auto",
    if_generation_match=generation,
)
digest = hashlib.sha256(target.read_bytes()).hexdigest()
if expected_digest and digest != expected_digest:
    raise SystemExit(
        f"initial model SHA-256 mismatch: expected {expected_digest}, got {digest}"
    )
print(f"Downloaded frozen initial model {uri} ({digest})", flush=True)
PY
fi

RESUME_DIR="${LOCAL_OUTPUT_DIR}/checkpoints"
mkdir -p "$RESUME_DIR"
RESUME_PATH=""
AUTO_RESUME_URI="${GCS_OUTPUT_URI%/}/checkpoints/latest.pt"
DOWNLOAD_URI="${RESUME_CHECKPOINT_URI:-$AUTO_RESUME_URI}"
REQUIRE_RESUME=0
if [[ -n "$RESUME_CHECKPOINT_URI" ]]; then
  REQUIRE_RESUME=1
fi

RESUME_PATH="$(python - "$DOWNLOAD_URI" "$RESUME_DIR" "$REQUIRE_RESUME" <<'PY'
import hashlib
import json
import pathlib
import posixpath
import sys

import torch
from google.cloud import storage

uri, destination_dir, required = sys.argv[1:]
bucket_name, separator, object_name = uri[5:].partition("/")
if not separator or not object_name:
    raise SystemExit(f"invalid Cloud Storage object URI: {uri}")

client = storage.Client()
blob = client.bucket(bucket_name).blob(object_name)
if blob.exists(client=client):
    target_dir = pathlib.Path(destination_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / pathlib.PurePosixPath(object_name).name
    blob.download_to_filename(str(target), checksum="auto")

    try:
        checkpoint = torch.load(target, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(target, map_location="cpu")
    replay_name = checkpoint.get("replay_path", "")
    replay_digest = checkpoint.get("replay_sha256", "")
    if not replay_name or pathlib.PurePosixPath(replay_name).name != replay_name:
        raise SystemExit("resume checkpoint contains an unsafe or missing replay_path")
    if not replay_digest:
        raise SystemExit("resume checkpoint is missing replay_sha256")

    remote_parent = posixpath.dirname(object_name)
    replay_remote = posixpath.join(remote_parent, replay_name)
    metadata_name = str(pathlib.PurePosixPath(replay_name).with_suffix(".json"))
    metadata_remote = posixpath.join(remote_parent, metadata_name)
    for remote_name, local_name in (
        (replay_remote, replay_name),
        (metadata_remote, metadata_name),
    ):
        sidecar = client.bucket(bucket_name).blob(remote_name)
        if not sidecar.exists(client=client):
            raise SystemExit(f"resume bundle is incomplete; missing gs://{bucket_name}/{remote_name}")
        sidecar.download_to_filename(str(target_dir / local_name), checksum="auto")

    replay_path = target_dir / replay_name
    actual_digest = hashlib.sha256(replay_path.read_bytes()).hexdigest()
    if actual_digest != replay_digest:
        raise SystemExit("resume replay digest does not match checkpoint")
    metadata = json.loads((target_dir / metadata_name).read_text(encoding="utf-8"))
    if metadata.get("sha256") != replay_digest:
        raise SystemExit("resume replay metadata digest does not match checkpoint")

    # Keep append-only metrics intact when Vertex replaces the worker VM.
    run_prefix = posixpath.dirname(remote_parent)
    metrics_remote = posixpath.join(run_prefix, "metrics.jsonl")
    metrics_blob = client.bucket(bucket_name).blob(metrics_remote)
    if metrics_blob.exists(client=client):
        metrics_blob.download_to_filename(
            str(target_dir.parent / "metrics.jsonl"),
            checksum="auto",
        )
    print(f"Downloaded coherent resume bundle from {uri}", file=sys.stderr, flush=True)
    print(target)
elif required == "1":
    raise SystemExit(f"explicit resume checkpoint does not exist: {uri}")
else:
    print("No prior checkpoint found; starting a new production run.", file=sys.stderr, flush=True)
PY
)"

UPLOAD_CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gomoku-upload.XXXXXX")"
UPLOAD_DONE_FILE="${UPLOAD_CONTROL_DIR}/done"

python - "$LOCAL_OUTPUT_DIR" "$GCS_OUTPUT_URI" "$SYNC_INTERVAL_SECONDS" "$UPLOAD_DONE_FILE" <<'PY' &
import os
import pathlib
import sys
import time
import traceback

from google.cloud import storage

local_root = pathlib.Path(sys.argv[1])
uri = sys.argv[2].rstrip("/")
interval = int(sys.argv[3])
done_file = pathlib.Path(sys.argv[4])

bucket_name, separator, prefix = uri[5:].partition("/")
if not separator or not prefix:
    raise SystemExit(f"invalid Cloud Storage prefix: {uri}")
prefix = prefix.strip("/")
client = storage.Client()
bucket = client.bucket(bucket_name)
uploaded = {}


def changed_files():
    paths = [path for path in local_root.rglob("*") if path.is_file()]
    # latest.pt is the commit record for a resumable checkpoint. Upload every
    # referenced replay/metadata artifact first, then publish latest.pt last.
    paths.sort(key=lambda path: (path.relative_to(local_root).as_posix() == "checkpoints/latest.pt", path.as_posix()))
    for path in paths:
        stat = path.stat()
        fingerprint = (stat.st_size, stat.st_mtime_ns)
        if uploaded.get(path) != fingerprint:
            yield path, fingerprint


def upload_incremental(final=False):
    count = 0
    for path, fingerprint in changed_files():
        relative = path.relative_to(local_root).as_posix()
        object_name = f"{prefix}/{relative}"
        bucket.blob(object_name).upload_from_filename(
            str(path),
            checksum="auto",
            timeout=600,
        )
        uploaded[path] = fingerprint
        count += 1
        print(f"Uploaded gs://{bucket_name}/{object_name}", flush=True)

    if final:
        latest = local_root / "checkpoints" / "latest.pt"
        if latest.is_file():
            model_object = f"{prefix}/model/model.pt"
            bucket.blob(model_object).upload_from_filename(
                str(latest),
                checksum="auto",
                timeout=600,
            )
            print(f"Published final model gs://{bucket_name}/{model_object}", flush=True)
    return count


while not done_file.exists():
    try:
        upload_incremental()
    except Exception:
        print("Checkpoint sync failed; retrying on the next interval.", file=sys.stderr, flush=True)
        traceback.print_exc()
    for _ in range(interval):
        if done_file.exists():
            break
        time.sleep(1)

last_error = None
for attempt in range(1, 6):
    try:
        upload_incremental(final=True)
        print("Final checkpoint sync complete.", flush=True)
        raise SystemExit(0)
    except Exception as error:
        last_error = error
        print(f"Final checkpoint sync attempt {attempt}/5 failed: {error}", file=sys.stderr, flush=True)
        if attempt < 5:
            time.sleep(min(30, attempt * 5))
raise SystemExit(f"final checkpoint sync failed: {last_error}")
PY
UPLOADER_PID=$!

TRAIN_COMMAND=(
  python -m gomoku_zero.training
  --config "$TRAIN_CONFIG"
  --output-dir "$LOCAL_OUTPUT_DIR"
  --device "$TRAIN_DEVICE"
  --allow-slow-production
)
if [[ -n "$RESUME_PATH" && -f "$RESUME_PATH" ]]; then
  TRAIN_COMMAND+=(--resume "$RESUME_PATH")
elif [[ -n "$INITIAL_MODEL_PATH" && -f "$INITIAL_MODEL_PATH" ]]; then
  TRAIN_COMMAND+=(--initialize-from "$INITIAL_MODEL_PATH")
fi

TRAIN_PID=""
RECEIVED_SIGNAL=0
forward_signal() {
  RECEIVED_SIGNAL=1
  if [[ -n "$TRAIN_PID" ]] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    kill -TERM "$TRAIN_PID"
  fi
}
trap forward_signal INT TERM

printf 'Starting production training with output staged at %s\n' "$LOCAL_OUTPUT_DIR"
"${TRAIN_COMMAND[@]}" &
TRAIN_PID=$!

set +e
wait "$TRAIN_PID"
TRAIN_STATUS=$?
set -e
if ((RECEIVED_SIGNAL)) && ((TRAIN_STATUS == 0)); then
  TRAIN_STATUS=143
fi

TOURNAMENT_STATUS=0
if ((TRAIN_STATUS == 0)) && [[ "$RUN_TOURNAMENT" == "1" ]]; then
  TOURNAMENT_COMMAND=(
    python -m gomoku_zero.tournament
    --output-dir "${LOCAL_OUTPUT_DIR}/tournament"
    --games-per-pair "$TOURNAMENT_GAMES_PER_PAIR"
    --simulations "$TOURNAMENT_SIMULATIONS"
    --opening-plies "$TOURNAMENT_OPENING_PLIES"
    --seed "$TOURNAMENT_SEED"
    --device "$TRAIN_DEVICE"
    --max-batch-size "$TOURNAMENT_MAX_BATCH_SIZE"
    --bootstrap-samples "$TOURNAMENT_BOOTSTRAP_SAMPLES"
  )
  if [[ -n "$INITIAL_MODEL_PATH" && -f "$INITIAL_MODEL_PATH" ]]; then
    TOURNAMENT_COMMAND+=(--model "v0000-base=${INITIAL_MODEL_PATH}" --anchor v0000-base)
  fi
  for checkpoint in "${LOCAL_OUTPUT_DIR}"/checkpoints/iteration-*.pt; do
    [[ -f "$checkpoint" ]] || continue
    stem="$(basename "$checkpoint" .pt)"
    iteration="${stem#iteration-}"
    TOURNAMENT_COMMAND+=(--model "v${iteration}=${checkpoint}")
  done
  printf 'Starting fixed-checkpoint round-robin tournament.\n'
  set +e
  "${TOURNAMENT_COMMAND[@]}" &
  TRAIN_PID=$!
  wait "$TRAIN_PID"
  TOURNAMENT_STATUS=$?
  set -e
  if ((RECEIVED_SIGNAL)) && ((TOURNAMENT_STATUS == 0)); then
    TOURNAMENT_STATUS=143
  fi
fi

touch "$UPLOAD_DONE_FILE"
set +e
wait "$UPLOADER_PID"
UPLOAD_STATUS=$?
set -e

if ((TRAIN_STATUS != 0)); then
  printf 'Training exited with status %s; partial checkpoints were synced.\n' "$TRAIN_STATUS" >&2
  exit "$TRAIN_STATUS"
fi
if ((TOURNAMENT_STATUS != 0)); then
  printf 'Tournament exited with status %s; completed games were synced.\n' "$TOURNAMENT_STATUS" >&2
  exit "$TOURNAMENT_STATUS"
fi
if ((UPLOAD_STATUS != 0)); then
  printf 'Training completed, but final GCS sync exited with status %s.\n' "$UPLOAD_STATUS" >&2
  exit "$UPLOAD_STATUS"
fi

printf 'Production training, optional tournament, and GCS publication completed successfully.\n'
