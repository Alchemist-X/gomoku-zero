# Model card

## Summary

Gomoku Zero is an AlphaZero-style policy-and-value system for a 15×15 Gomoku board. A residual
neural network supplies 225 raw policy logits and a black/draw/white value estimate; legal-action
masking and PUCT turn those predictions into moves and analysis. This repository primarily ships
training and evaluation infrastructure. A code release or an untrained checkpoint is not, by
itself, a validated playing model.

## Model details

- **Version:** repository version 0.1.0; checkpoint version must be recorded separately.
- **Rules:** 15×15 freestyle Gomoku; five or more in a line wins; no Renju forbidden moves.
- **Inputs:** three 15×15 planes: black stones, white stones, and black-to-move.
- **Outputs:** 225 raw policy logits plus raw W/D/L logits in `[black, draw, white]` order.
- **Decision layer:** rules-derived mask before softmax, then legal-only PUCT search.
- **Configured production architecture:** 128 channels, 10 residual blocks.
- **Training source:** generated self-play positions and completed-game outcomes; no external human
  game corpus is claimed.
- **License:** MIT for repository code; checkpoint provenance must be documented per release.

Model size and search depth are configuration facts, not evidence of strength.

No trained or promoted checkpoint is bundled with this model card. When the service starts
without a trusted `GOMOKU_CHECKPOINT`, it uses a deterministic legal heuristic, marks provenance
as `bootstrap-untrained`, and labels estimates as `deterministic-heuristic`. Those W/D/L-shaped
values are not neural-network predictions and must never be recorded as model evaluation.

The repository also bundles no calibration report and no validated INT8 model. Search-budget
convergence is not probability calibration, and ordinary PyTorch inference is not evidence for
quantized accuracy. Calibration or INT8 claims require separately versioned artifacts and
evaluation under the exact deployed runtime.

## Intended uses

- research and education about self-play, masking, PUCT, calibration, and reproducible evaluation;
- bounded interactive analysis of legal moves on the supported board/ruleset;
- a baseline for controlled engine-vs-engine experiments; and
- self-hosted demonstration through the API.

## Out-of-scope uses

- claiming that Gomoku is solved or that a displayed probability is its mathematical truth;
- using policy or W/D/L output as a guaranteed optimal move;
- comparing checkpoints without fixed rules, openings, colors, search budgets, and uncertainty;
- unbounded public compute, wagering, or other high-stakes decisions; and
- loading checkpoints from untrusted sources.

## Current evidence status

No current strength or initial-position W/D/L number should be claimed unless it is tied to an
immutable checkpoint digest, code revision, ruleset, opponent distribution, opening suite, search
configuration, game records, and an evaluation report produced by this repository.

The following numbers originated in historical/bootstrap discussion and are retained here only as
a warning against mislabeling them:

| Historical claim | Reported value | Evidence status |
|---|---:|---|
| Empty-board black/draw/white estimate | 53.6% / 12.2% / 34.2% | **Unverified; not a current model result** |
| 600 self-play black/draw/white counts | 368 / 55 / 177 | **Unverified; training games are not independent evaluation** |
| Same 600-game percentages | 61.3% / 9.2% / 29.5% | **Derived from the unverified counts; not a scorecard** |

These figures must not be copied into UI labels, release notes, or deployment metadata as current
truth. A deterministic position has a game-theoretic result—win, loss, or draw under fixed
rules—not an intrinsic `53.6%` result. A probability reflects a policy matchup, search budget,
calibration method, and uncertainty.

## Evaluation requirements

The production evidence target is distinct from the training target:

- generate at least **20,000 self-play games** for the planned training run;
- freeze a promoted checkpoint;
- play approximately **6,000 separately budgeted evaluation games** with frozen search
  parameters: independently seeded openings for a single-model color estimate, or color-swapped
  opening pairs for a declared opponent matchup; and
- evaluate selected positions at `128 → 512 → 2,048 → 6,000 → 12,000` MCTS simulations.

The formal 6,000-game profile uses 800 simulations per move. Setting both games and simulations
per move to 6,000 multiplies into billions of leaf evaluations for realistic game lengths and is
not the default evidence plan. The position ladder tests whether search results stabilize relative
to one frozen evaluator; it does not prove those results correct.

Reports should publish black wins, draws, white wins, color-swapped paired results, illegal-move
or timeout counts, game-length distribution, the uncertainty method, and calibration metrics.
Results are valid only for the recorded model/search pairing. Web 40-simulation output and server
6,000-simulation output require separate calibration checks.

## Limitations and failure modes

- Self-play reinforces the model's own blind spots; rare tactics and unfamiliar openings can be
  out of distribution.
- More MCTS reduces search-budget error but cannot remove network bias, data bias, or value
  miscalibration.
- W/D/L probabilities can be overconfident and can move when search budget, opponent, temperature,
  quantization, or hardware inference changes.
- A Top-3 set may be stable while all candidates are strategically wrong.
- Parallel self-play creates correlated positions; evaluation games can also be correlated through
  shared openings and seeds.
- The production game and search budgets are substantial. Partial, interrupted, or silently
  resumed runs are not equivalent to the declared protocol.
- The 20,000-game data target excludes 2,000 configured promotion matches and the final
  6,000-game suite; current search performs batch-one leaf inference, so nominal game counts
  understate operational cost.
- Public unauthenticated deep-search endpoints are susceptible to cost exhaustion and denial of
  service without server-side limits.

## Legal-action safety

Occupied points are removed from the action set; they are not visually hidden. Masking occurs
before policy normalization, MCTS creates only legal children, replay targets may contain no
illegal mass, and move application checks legality again. Terminal positions return no legal
policy even if unused points remain. See [Architecture](ARCHITECTURE.md) for equations and test
invariants.

This defense-in-depth prevents normal illegal recommendations but does not make arbitrary model
files or malformed network responses safe. Output shapes and finite values must also be validated;
the server should fail closed on evaluator errors.

## Reproducibility and release checklist

A releasable checkpoint should include or reference:

- immutable weight digest and safe serving format;
- repository revision, dependency lock, and complete validated configuration;
- rules and encoder schema versions;
- training seed, hardware, actor count, elapsed compute, self-play/replay statistics, and resume
  history;
- promotion record and independent evaluation game records;
- selected-position convergence and calibration reports; and
- known regressions, operating limits, and rollback checkpoint.

Promote an immutable checkpoint only after these artifacts pass review. Keep `latest` as a training
convenience, never as the public model identity.
