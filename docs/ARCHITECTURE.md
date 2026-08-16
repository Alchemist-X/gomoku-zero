# Architecture

Gomoku Zero keeps game rules, policy masking, search, learning, evaluation, and serving as
separate layers. The main contract across every layer is stronger than “the UI does not show
occupied points”: an illegal action must never enter a policy denominator, MCTS child set,
training target, sampled move, visit distribution, or API recommendation.

## System flow

```text
Board state + side to move
          │
          ├── rules ──> terminal result + legal mask [225]
          │
          └── encoder ──> neural network
                             ├── raw policy logits [225]
                             └── W/D/L value [3]
                                      │
                 legal mask ──────────┤
                                      v
                         legal-only policy prior
                                      │
                                      v
                    PUCT tree (legal children only)
                         │                    │
                  visit policy         search W/D/L
                         │                    │
                         ├── self-play samples
                         └── analysis API
```

The implemented rules are 15×15 freestyle Gomoku: black moves first, five or more contiguous
stones win, and there are no Renju forbidden moves. Actions use row-major indexing,
`action = row * 15 + column`, for 225 board points. The three input planes are black stones,
white stones, and an all-one/all-zero black-to-move plane. Training W/D/L targets are stored in
the absolute order `[black win, draw, white win]`. Geometric D4 augmentation transforms the board
planes, policy target, legal mask, and visit counts together; it does not change the absolute game
result.

## Legal-action masking

Let the network output raw logits $z \in \mathbb{R}^{225}$, and let $A(s)$ be the legal
action set produced by the rules for state $s$. For a non-terminal position:

$$
z'_a =
\begin{cases}
z_a, & a \in A(s) \\
-\infty, & a \notin A(s)
\end{cases}
\qquad
P(a\mid s) = \frac{\exp(z'_a)}{\sum_b \exp(z'_b)}.
$$

The inference helper validates exact mask shape and boolean type, requires at least one legal
action in every normalized row, rejects non-finite legal logits, applies literal negative infinity
with `masked_fill`, and explicitly zeros illegal entries after softmax. Non-finite values on an
illegal entry are discarded before the reduction and cannot poison legal probabilities. A
terminal or otherwise all-false mask is a separate control-flow case: it returns no move and an
all-zero policy; it must never be sent through softmax.

For numerical stability, the equivalent legal-only calculation is:

$$
m=\max_{b\in A(s)} z_b,\quad
P(a\mid s)=
\begin{cases}
\dfrac{\exp(z_a-m)}{\sum_{b\in A(s)}\exp(z_b-m)},&a\in A(s)\\
0,&a\notin A(s).
\end{cases}
$$

This is “mask before softmax.” It gives four required invariants:

1. every illegal action has exactly zero probability;
2. legal probabilities sum to one;
3. changing an illegal logit cannot change any legal probability or the policy loss; and
4. the gradient of every illegal logit is exactly zero.

Doing a 225-way softmax first and then assigning zero to occupied points is not equivalent. Its
denominator still contains illegal logits, the surviving probabilities do not sum to one over
the legal set, and policy gradients still spend capacity suppressing moves that the rules already
forbid. A post-hoc renormalization fixes only the displayed probabilities, not that gradient.

The legal mask is a rules result, not merely a rendering mask. In particular, after a win or draw
the game is terminal even if empty points remain, so the legal set is empty. Defensive checks at
move application, MCTS expansion, self-play sampling, and API serialization make a corrupted
policy fail closed.

## Model and loss

The configured network is a convolutional residual trunk with separate policy and W/D/L heads.
`configs/smoke.json` uses a deliberately tiny network for integration testing;
`configs/production.json` uses 128 channels and 10 residual blocks. Architecture size is not a
strength claim: playing strength depends on the promoted weights, self-play distribution, search
budget, and calibration evidence.

For an active training row with legal visit target $\pi$, the policy objective is computed only
over legal actions:

$$
\mathcal{L}_{policy}=-\sum_{a\in A(s)}\pi_a\log P(a\mid s).
$$

Targets with meaningful mass on an illegal point are rejected instead of silently repaired.
Terminal states do not produce a policy target. Replay samples retain the per-position legal mask,
so the same invariant is checked again at training time. The W/D/L head uses the completed game
outcome; combined optimization may also include regularization configured by the trainer.

## PUCT search

PUCT consumes masked priors and creates children only for legal actions. Root Dirichlet noise,
temperature sampling, visit-count targets, Top-K selection, and zero-visit fallbacks all operate
on that same child set. A final legality assertion remains necessary even when every upstream
component is correct.

The evaluator and search carry absolute W/D/L in `[black, draw, white]` order. After simulations,
`root_wdl` follows the currently selected most-visited move, while `root_value` and each move's
Q value are expressed from the root side-to-move's perspective. Coordinates include the `I`
column (`A1`, `H8`, `O15`); consumers should not apply Go-style letter skipping.

The analysis service can expose cumulative stages `40 → 200 → 1,000 → 3,000 → 6,000`. These
numbers are MCTS simulations for one position, not training games. A stage should extend the same
tree; it is not five unrelated searches. Clients must bind results to the analyzed position and
ignore stale stages after a move, reset, or rules/model change.

## Boundaries and artifacts

- Configuration is validated centrally before expensive work begins. Seeds, model shape,
  training budget, evaluation budget, convergence thresholds, and serving caps belong in the
  recorded run configuration.
- Replay artifacts contain pre-move encoded states, legal visit policies, absolute W/D/L targets,
  and packed legal masks. They are training data, not evaluation evidence.
- A training checkpoint is resumable internal state. A promoted inference checkpoint is an
  immutable, explicitly selected serving artifact. Treating “latest” as “best” couples public
  behavior to an unfinished run and is discouraged.
- Evaluation freezes the candidate, opponent, rules, openings, and search settings. Its reports
  are versioned evidence attached to a checkpoint; they are not inferred from training loss.
- The public API is a bounded inference surface. Training, checkpoint writes, and cloud project
  credentials stay outside the public container.

## Analysis API contract

`GET /api/health` publishes service readiness, model provenance, and configured limits;
`GET /api/model` returns checkpoint/heuristic provenance; and `POST /api/analyze` returns one
NDJSON frame per cumulative budget. Requests contain exactly 225 stone values and optional
strictly increasing budgets. The API derives side to move from valid alternating stone counts,
rejects terminal positions, caps the final budget, and constructs one `SearchSession` per request.
`mode: "instant"` is the default and emits only the first configured rung (normally 40);
`mode: "deep"` emits the full `40 → 200 → 1,000 → 3,000 → 6,000` ladder unless explicit budgets
are supplied.

Every analysis frame carries a request/analysis ID, sequence, position hash, cumulative
simulation count, completion flag, absolute black/draw/white outcome, model status, and up to
three legal moves sorted by visits, root-perspective Q, then action. Clients use the analysis ID,
position hash, and monotonic sequence together; cancellation is a compute optimization, not a
substitute for rejecting stale frames.

No trained checkpoint is bundled with the repository. If `GOMOKU_CHECKPOINT` is absent, the
service intentionally uses a deterministic legal heuristic and labels every response
`bootstrap-untrained` / `deterministic-heuristic`. Its W/D/L-shaped output is a display heuristic,
not a trained prediction, calibration result, or evaluation statistic. A configured checkpoint
is loaded lazily, identified by SHA-256, and reported through model provenance. Cloud Run can
materialize an explicitly configured `gs://` object into a private atomic cache. If an explicit
local or GCS checkpoint is missing, unreadable, or invalid, analysis fails closed with unavailable
provenance/503 rather than silently falling back to the bootstrap heuristic.

## Verification priorities

Tests should cover an occupied point with the highest raw logit, a single legal action, a terminal
board with empty points, extreme logits, illegal-target rejection, exact zero illegal gradients,
D4 mask/target alignment, MCTS child legality, and API recommendations after the formerly best
point has been occupied. A selected-position budget study must run
`128 → 512 → 2,048 → 6,000 → 12,000` and report Top-1, Top-3 set, visit proportions, and W/D/L
drift rather than declaring a budget sufficient by inspection.
