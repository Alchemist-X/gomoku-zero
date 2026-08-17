# Self-play scaling: from batch 1 to batched GPU inference

## The two unrelated batch sizes

This project now names the two batches separately because they solve different
problems:

| Setting | Tensor/work unit | What it changes |
|---|---|---|
| `training.batch_size` | replay samples per optimizer step | GPU utilization, gradient noise, optimizer memory, and number of parameter updates |
| `training.inference_batch_size` | MCTS leaf positions per network forward | number of model calls during self-play and inference throughput |

The completed historical eight-hour run used optimizer `batch_size=256`, but
each CPU actor evaluated one MCTS leaf at a time with model input
`[1, 3, 15, 15]`. Increasing the optimizer batch does not fix that inference
bottleneck.

## Phase 1 implemented here

The `batched` self-play backend keeps multiple independent games in lockstep:

1. Every active game owns its own PUCT tree and RNG.
2. Each tree selects exactly one non-terminal leaf.
3. Those independent leaves are stacked into `[B, 3, 15, 15]`.
4. One model forward produces `[B, 225]` policy logits and `[B, 3]` WDL logits.
5. Each result is returned to its original tree and backed up before that tree
   may select another leaf.

Root positions are batched as well. A tree never has two outstanding leaves,
so this phase does not require virtual loss, tree locks, or an approximation to
serial PUCT. Stable lane order and per-game seeds preserve reproducibility.

Policy masking is still per position and happens before softmax. The batched
legal mask has shape `[B, 225]`; occupied points receive exactly zero
probability and each row is normalized only over that board's legal moves.
Terminal positions bypass the network and use their exact WDL target.

## Configuration and runnable GPU profile

`configs/gpu-batched.json` is the explicit Phase 1 profile:

```json
{
  "self_play_backend": "batched",
  "self_play_lanes": 48,
  "inference_batch_size": 48,
  "batch_size": 256
}
```

The profile has 48 games per iteration, so its real inference batch can never
exceed 48. Near the end of an iteration it will be smaller as games finish.
`configs/eight-hour.json` remains unchanged because it is the provenance and
strict-resume configuration for the already completed CPU run.

Run the profile only with a CUDA-capable image and worker:

```bash
uv run gomoku-train \
  --config configs/gpu-batched.json \
  --output-dir /tmp/gomoku-gpu-run \
  --device cuda
```

The CLI refuses the batched backend on CPU by default. The
`--allow-batched-cpu` escape hatch exists for deliberate correctness benchmarks,
not production: a single Python process then owns all PUCT selection work and
can be slower than the legacy multi-process CPU backend.

For Vertex AI, submit the same config with an accelerator and a compatible
official CUDA PyTorch wheel index:

```bash
ACCELERATOR_TYPE=NVIDIA_L4 \
ACCELERATOR_COUNT=1 \
TORCH_INDEX_URL=https://download.pytorch.org/whl/<compatible-cuda-index> \
deploy/submit_vertex_training.sh \
  --project-id PROJECT_ID \
  --config configs/gpu-batched.json \
  --machine-type g2-standard-16 \
  --yes
```

Accelerator availability, quota, machine compatibility, and the exact PyTorch
CUDA index must be verified before submission. Large runs still require the
existing explicit cost acknowledgement; batching is not treated as permission
to start an expensive job.

## Measurements, not promises

Every batched self-play iteration writes `self_play_inference` telemetry into
`metrics.jsonl`:

- `inference_calls`
- `positions_evaluated`
- `mean_effective_batch_size`
- `max_effective_batch_size`
- `inference_seconds`
- `inference_positions_per_second`
- `self_play_seconds`
- `end_to_end_positions_per_second`

`positions_evaluated` includes non-terminal roots and non-terminal simulation
leaves. It is not the same as the requested simulation count because terminal
leaves are resolved without the network. CUDA timing includes the final
device-to-host synchronization.

Do not infer an end-to-end speedup from forward-only throughput. Compare a
fixed seed/config on the target worker and report games/hour, total
self-play seconds, mean effective batch, and GPU utilization. The Python PUCT
selection loop can become the next bottleneck once neural inference is fast.

## Tuning order

1. Start with lanes and inference batch at 32 or 48 on one GPU.
2. Check `mean_effective_batch_size`; increasing the configured maximum does
   nothing if there are fewer active lanes.
3. Check GPU memory and utilization before moving to 64 or 128 lanes.
4. Compare end-to-end games/hour, not only positions/second.
5. Tune optimizer `batch_size` independently according to training memory and
   convergence; it does not control MCTS inference batching.

## Known Phase 1 limits and Phase 2

Phase 1 batches the dominant self-play network calls, but it deliberately does
not change promotion or frozen evaluation. Those paths still search serially at
batch 1, so an iteration with a promotion gate can remain much slower.

It also moves PUCT selection for the active lanes into one Python process. A
larger deployment should use a Phase 2 architecture:

- multiple CPU actor processes, each owning one or more independent game trees;
- one central GPU inference service with a short dynamic batching window;
- shared-memory request/result slots rather than pickling 225-value policies;
- grouping requests by model identity during candidate/champion evaluation;
- backpressure, timeouts, and batch/queue latency telemetry.

That design restores CPU parallelism while retaining large GPU batches. It is
more complex and should follow an end-to-end Phase 1 benchmark, not be mixed
into the first correctness upgrade.
