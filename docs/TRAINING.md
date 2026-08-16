# Training and evaluation

This guide treats self-play generation, neural-network optimization, model promotion, independent
evaluation, and per-position convergence as different jobs. Keeping their budgets separate is
essential both statistically and financially.

## Before spending compute

Install the development environment, run lint and unit tests, then complete the smoke profile:

```bash
uv sync --frozen --python 3.11 --extra dev
uv run ruff check .
uv run pytest
uv run gomoku-train --config configs/smoke.json --output-dir runs/smoke
uv run gomoku-evaluate --config configs/smoke.json \
  --checkpoint runs/smoke/checkpoints/latest.pt \
  --output-dir runs/smoke/evaluation
uv run gomoku-convergence --config configs/smoke.json \
  --checkpoint runs/smoke/checkpoints/latest.pt \
  --positions examples/convergence-positions.json \
  --output runs/smoke/convergence.json
```

The smoke result proves only that data generation, optimization, checkpointing, loading, and
evaluation connect correctly. Its convergence ladder is only `2 → 4 → 8`; two self-play games and
a tiny network provide no useful strength or convergence evidence.

Before a production run, record the code revision, complete configuration, ruleset, dependency
lock, device type, accelerator count, seed, intended checkpoint destination, storage retention,
and a maximum compute/currency budget. Estimate one small iteration on the intended hardware and
extrapolate from measured positions per second; theoretical FLOP estimates routinely miss MCTS
and actor bottlenecks.

## Three meanings of “6,000”

| Work item | Unit | What it can answer | What it cannot answer |
|---|---|---|---|
| 6,000 MCTS simulations | leaf expansions for one position | whether the current model's leading moves are stable under a larger search budget | whether the model or its win rate is correct |
| 6,000 complete self-play games | generated training games | whether the replay set is large enough for an improved bootstrap experiment | an unbiased final win rate |
| 6,000 independent evaluation games | complete games of frozen policies | the W/D/L rate for the exact frozen policies, openings, rules, and budgets | the game-theoretic value of Gomoku |

MCTS leaves are adaptive and share paths; they are not independent Bernoulli trials. Therefore a
confidence interval for 6,000 independent games cannot be attached to 6,000 simulations in one
tree.

The expensive knobs multiply. Evaluating 6,000 games while searching 6,000 simulations on every
move can require billions of leaf evaluations once game length is included. The production plan
instead fixes formal evaluation at 800 simulations per move and reserves budgets through 12,000
for a small set of convergence positions.

## Bounded, resumable phase-1 target

`configs/production.json` is the formal phase-1 profile. It is intentionally bounded by explicit
game/search counts, preflight gates, cloud runtime, and persistent checkpoints; “formal” means the
protocol is recorded and resumable, not that its future model is already strong or calibrated.
It currently describes:

- 100 iterations × 200 self-play games = **20,000 complete self-play games**;
- 28 self-play actors and 800 MCTS simulations per move;
- a 1,000,000-position replay capacity;
- 1,000 optimizer steps per iteration, batch size 256, gradient clipping, and mixed precision;
- 128 trunk channels with 10 residual blocks;
- 200 candidate-versus-champion promotion games every ten iterations at 200 simulations per move
  (ten gates and 2,000 matches in the full run), with a configured promotion threshold;
- a separate 6,000-game evaluation at 800 simulations per move;
- selected-position convergence budgets `128 → 512 → 2,048 → 6,000 → 12,000`; and
- public serving stages `40 → 200 → 1,000 → 3,000 → 6,000`, capped at 6,000.

Start it only after the smoke run and budget gate succeed:

```bash
uv run gomoku-train --config configs/production.json \
  --output-dir runs/production \
  --preflight-only \
  --allow-slow-production

uv run gomoku-train --config configs/production.json \
  --output-dir runs/production \
  --allow-slow-production
```

The trainer intentionally refuses a run of 1,000 or more self-play games unless
`--allow-slow-production` is present. That option acknowledges the printed worst-case leaf count;
it does not make the job fast. The current search evaluates neural-network leaves one at a time,
so measure the smoke throughput before authorizing the production target. The printed training
preflight reports self-play and promotion separately. At the configured maxima, rough upper
bounds are 3.6 billion leaf simulations for 20,000 self-play games, another 90 million for 2,000
promotion matches at the smaller promotion budget, and 1.08 billion for the later 6,000-game
evaluation. Actual games often end before 225 plies, but root evaluations and operational
overhead also cost time; use measurements rather than treating these bounds as a quote.

Twenty thousand games are a production target for this project, not a promise of expert strength
and not proof that training has converged. Inspect replay diversity, loss curves, gradient norms,
candidate-versus-champion results, color balance, game length, resignation behavior if added, and
illegal-action assertions throughout the run.

## Vertex AI production job

The submission wrapper requires a clean committed Git revision, creates a private regional output
bucket when needed, prints the complete billable plan, applies a hard runtime bound, and refuses
to submit without `--yes` (or the equivalent confirmation environment variable):

```bash
bash deploy/submit_vertex_training.sh \
  --project-id YOUR_PROJECT_ID \
  --repository gomoku \
  --yes
```

Its cost-conscious default is one `n1-highcpu-32` CPU worker, a 500 GB SSD boot disk, and a
seven-day maximum. Those are upper bounds, not a promise that batch-one MCTS will finish the full
configuration. Optional accelerators require an explicitly compatible CUDA PyTorch wheel index;
do not select a GPU until a measured smoke benchmark shows that the input pipeline can use it.
Cloud Build tags the image with the committed Git SHA, and the submitted job is pinned to the
resolved image digest. `--allow-dirty-source` exists as an explicit escape hatch for experiments,
but a dirty-source job is not a formal reproducible run.

The worker stages output under `/tmp`, uploads changed artifacts to the unique private GCS run
prefix every five minutes by default, publishes `latest.pt` only after its replay bundle, retries
final upload, and auto-resumes from that run prefix after a worker restart. An explicit
`--resume-uri gs://.../checkpoints/latest.pt` must point to a coherent checkpoint/replay bundle.
The stored replay digest and metadata are checked before resuming.

The submission command prints the Vertex job resource. Monitor it without mutating the job:

```bash
bash deploy/check_training.sh \
  --project-id YOUR_PROJECT_ID \
  --job projects/PROJECT_NUMBER/locations/REGION/customJobs/JOB_ID
```

The script submits training only. Candidate promotion occurs inside the loop, while the formal
6,000-game evaluation remains a separately budgeted frozen-policy job. Keep the bucket private,
set lifecycle retention deliberately, and configure billing alerts before confirmation.

## Checkpoint discipline and recovery

The trainer writes `metrics.jsonl`, versioned `checkpoints/iteration-NNNN.pt` model snapshots,
`checkpoints/latest.pt`, and a paired `replay-iteration-NNNN.npz` plus JSON metadata. The
`latest.pt` payload contains model/champion, optimizer, scheduler, AMP scaler, RNG, configuration,
global step, runtime mode, replay name/digest, and available code revision. Writes are atomic. A
new replay is made durable before `latest.pt` changes; stale replay snapshots are removed only
afterward. Therefore only `latest.pt` and its referenced replay/JSON pair form the supported exact
resume bundle. Iteration `.pt` files remain frozen models for evaluation or serving.

Store each run under an immutable run ID. Calculate an artifact digest, record the code revision,
then copy the checkpoint/replay set to durable object storage. Keep at least:

- the last known good promoted checkpoint;
- several recent resumable checkpoints;
- milestone checkpoints for regression and calibration; and
- the exact checkpoint used by each evaluation report and deployment.

Do not overwrite the promoted model with `latest.pt`. Resume with
`--resume RUN/checkpoints/latest.pt`; seed, `--skip-promotion` mode, and all training-dynamics
settings must match, although the iteration count may be extended. Replay digest, iteration, and
sample count are verified on load. Test a resume in the smoke profile before depending on it for
a costly job. After a crash, restore the matching checkpoint and replay artifacts into a new run
directory, verify the recorded configuration and digest, run a short evaluation, and only then
continue generation.
Loading an untrusted model artifact is unsafe; consume only artifacts produced by the controlled
pipeline. This code requests PyTorch's weights-only loader and pins a checkpoint schema, but
provenance and digest checks remain necessary.

Parallel actors and accelerators can make exact bitwise replay impossible even with a fixed seed.
The seed and deterministic settings are reproducibility inputs, not a guarantee that distributed
runs will be byte-identical.

## Candidate promotion

Promotion games compare a candidate against the current champion. Alternate colors and, when
openings are injected, pair each opening with colors swapped. Freeze network weights and search
settings for the match. Record wins, draws, losses, score convention, confidence method, early
stopping rule, and all search parameters.

A promotion threshold is an operational gate, not an independent final evaluation. Repeatedly
testing candidates against the same champion introduces selection effects; attach formal metrics
only after evaluating the selected frozen checkpoint on the predeclared evaluation suite.

## Independent evaluation

For the 6,000-game report:

1. freeze and hash the candidate and opponent checkpoints;
2. freeze the rules, encoder schema, 800-simulation search budget, PUCT parameters, and hardware
   inference mode;
3. use a fixed opening suite, with paired color-swapped games where configured;
4. disable self-play exploration noise and training-time sampling unless the evaluated policy
   explicitly includes them;
5. save every game record and aggregate black wins, draws, and white wins separately; and
6. report uncertainty with its assumptions, plus failures/timeouts as first-class outcomes.

Games that share openings or random seeds are not fully independent, so use paired or clustered
uncertainty when appropriate. A simple binomial interval is only an approximation for genuinely
independent games under one fixed outcome definition.

For scale only, suppose a future set of 6,000 genuinely independent games measured a black win
rate of `0.536`. Treating “black win” as a binary outcome would give
`SE ≈ sqrt(0.536 × 0.464 / 6000) ≈ 0.0064`, or a rough 95% interval of about ±1.3 percentage
points. This is an **illustrative calculation, not a result from this repository**. A real W/D/L
report is multinomial, paired openings introduce dependence, and the interval says nothing about
the game-theoretic result.

Run the evaluator with the exact promoted checkpoint:

```bash
uv run gomoku-evaluate --config configs/production.json \
  --checkpoint runs/production/checkpoints/PROMOTED_CHECKPOINT.pt \
  --output-dir runs/production/evaluation \
  --preflight-only \
  --allow-slow-evaluation

uv run gomoku-evaluate --config configs/production.json \
  --checkpoint runs/production/checkpoints/PROMOTED_CHECKPOINT.pt \
  --output-dir runs/production/evaluation \
  --allow-slow-evaluation
```

As with production training, the evaluator has a deliberate safety gate: 1,000 or more global
games require `--allow-slow-evaluation`, and a worst-case estimate above the configured ceiling
requires either sharding or a separate `--allow-large-run` acknowledgement. Use `--num-shards`
and a distinct `--shard-index` on each worker; each shard appends its own JSONL and writes its own
summary. `--resume` continues a partial shard without replaying completed global game IDs.

With no `--opponent-checkpoint`, the same frozen model plays both colors, every game receives an
independently seeded opening, and the report estimates black/draw/white rates for that fixed
policy. Supplying an opponent alternates the candidate's color and uses color-swapped opening
pairs; each pair stays on one shard and the report additionally gives candidate wins, draws,
losses, score, and Wilson intervals. Because games within a pair are correlated, the report labels
that limitation and counts independent opening units separately. A per-shard manifest binds the
checkpoint/opponent hashes, full config hash, game range, search budget, sharding, and pairing;
`--resume` refuses to append if any of them changed.

The historical empty-board figures `53.6% black / 12.2% draw / 34.2% white` and 600-game
self-play counts `368 black / 55 draw / 177 white` are **unverified bootstrap-era claims**. They
are included only to prevent accidental reuse as current results. They were not produced by a
versioned checkpoint and frozen evaluation protocol in this repository, so they must never appear
in a current scorecard. Even a validated W/D/L distribution would describe a fixed model/search
matchup, not the mathematical truth of a deterministic solved position.

## Selected-position convergence

The dedicated runner accepts a JSON object, a list of position objects, or
`{"positions": [...]}`. A position must contain exactly one of `moves` (row-major integers or
coordinates such as `H8`) or `cells`/`board` (225 flat values or a 15×15 array); give cells an
explicit `to_play` when counts do not determine it. For example:

```json
{
  "positions": [
    {"id": "empty", "moves": []},
    {"id": "opening-a", "moves": ["H8", "H9", "G8", "I8"]}
  ]
}
```

Run the same promoted checkpoint at cumulative budgets:

```bash
uv run gomoku-convergence \
  --config configs/production.json \
  --checkpoint runs/production/checkpoints/PROMOTED_CHECKPOINT.pt \
  --positions positions.json \
  --output runs/production/convergence.json
```

The configured ladder is:

```text
128 → 512 → 2,048 → 6,000 → 12,000
```

One tree is reused per position, so each rung adds only the missing simulations. The atomic JSON
report records checkpoint/config/positions hashes, code revision and dirty state, seed, C-PUCT,
device, model shape, generated time, and elapsed timing. Every rung records its requested budget,
simulations actually executed, legal Top-1/Top-3 move/coordinate/visit share/Q/network-prior
statistics, and absolute black/draw/white W/D/L. Production compares 6,000 versus 12,000 (shorter
custom ladders compare their last two rungs). The comparison reports Top-1 and Top-3-set equality,
per-outcome and maximum absolute W/D/L deltas, the strict threshold result, and one combined
`converged` flag.

Each non-terminal position executes up to 12,000 simulations; this command has no bulk-evaluation
meaning and should be limited to a small, predeclared position set after a cost/memory check.

“6,000 is enough” is supported only for a specific model and position when the 6,000- and
12,000-stage results satisfy those predeclared checks. This establishes search stability relative
to the current evaluator, not model correctness.

## Cost and public-service safety

- Training and bulk evaluation belong on private workers with explicit quotas and budget alerts.
- Do not expose arbitrary simulation counts on an unauthenticated endpoint. Clamp to the serving
  maximum and cap request concurrency, queue depth, body size, and wall time.
- Keep the default public path fast; make deep analysis asynchronous or otherwise bounded.
- Treat checkpoint paths and object-store locations as server configuration, never request input.
- Collect aggregate latency/error/cost telemetry without logging private credentials or
  unnecessary user positions.
- Stop a run on non-finite loss, illegal policy mass, corrupted checkpoint digest, runaway queue,
  or a breached cost ceiling. “More training” is not a reason to ignore a failed invariant.
