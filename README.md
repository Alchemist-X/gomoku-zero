# Gomoku Zero / 五子棋 Zero

An AlphaZero-style 15×15 Gomoku engine with legal-action masking, PUCT search,
self-play training, independent evaluation, and a public analysis API.

一个可复现的 AlphaZero 风格 15×15 五子棋引擎，包含合法动作 masking、PUCT 搜索、
self-play 训练、独立评估和可公开部署的分析 API。

> Status / 状态：early training infrastructure, not a solved-game oracle. The historical
> `53.6% / 12.2% / 34.2%` empty-board estimate and `368 / 55 / 177` results from 600
> self-play games are **unverified bootstrap-era claims**, not measurements of the model in
> this repository and not the game-theoretic truth of Gomoku.
>
> 当前项目仍处于训练基础设施阶段，并不是“已解出五子棋”的预言机。历史材料中的空盘
> `53.6% / 12.2% / 34.2%` 与 600 局 self-play 的 `368 / 55 / 177` 均为
> **未经本仓库验证的 bootstrap 旧说法**，不能当作当前模型指标或五子棋的数学真值。

No model weights, calibration report, or validated INT8 artifact is bundled in Git. Deployments
remain explicitly `bootstrap-untrained` until a trusted checkpoint is configured. The reference
deployment currently loads the promoted checkpoint from the bounded 2026-08-17 run; its exact
metrics, provenance, plots, and limitations are recorded in
[the experiment report](docs/experiments/eight-hour-20260817/README.md).

仓库不会把模型权重、校准报告或未经验证的 INT8 产物直接提交到 Git；未配置可信
checkpoint 的部署仍会明确标为 `bootstrap-untrained`。当前参考部署已加载 2026-08-17
受限训练中晋级的 checkpoint，完整指标、来源、图表与限制见
[实验报告](docs/experiments/eight-hour-20260817/README.md)。

## Public reference deployment / 线上参考部署

- App / 在线应用：<https://gomoku-zero-313049501255.asia-southeast1.run.app>
- Cloud Run revision: `gomoku-zero-00003-v6j`
- Serving source: `1aff963ac66893f057c30368f560e13cc74aa2c5`
- Model: training step `1000`, SHA-256
  `b82cb09cb163ef32986fe602ae3c2026a08b76e25ca3229534fc218c90b86137`

This checkpoint came from a **5 h 50 min bounded experiment**: four iterations, 192 self-play
games, 9,371 replay samples, and a 20-game promotion gate. It is a working trained artifact, not
the planned 20,000-game production run or 6,000-game independent evaluation. Its displayed W/D/L
values are model-and-search estimates, not the mathematical truth of Gomoku.

该 checkpoint 来自一次 **5 小时 50 分的时间盒实验**：4 个训练迭代、192 局 self-play、
9,371 个 replay 样本和 20 局晋级赛。它是可运行的训练产物，但不是计划中的 20,000 局
正式训练，也未完成 6,000 局独立评估；页面中的胜/和/负仅代表该模型与搜索预算的估计。

## Quick start / 快速开始

Requires Python 3.11 or 3.12. The locked `uv` path is recommended:

```bash
cd gomoku-zero
uv sync --frozen --python 3.11 --extra dev
uv run ruff check .
uv run pytest
```

Run the API locally / 本地运行 API：

```bash
uv run gomoku-api
```

Then open [http://127.0.0.1:8080](http://127.0.0.1:8080). No trained checkpoint is bundled. With
no `GOMOKU_CHECKPOINT`, the API clearly reports `bootstrap-untrained` and uses a deterministic
legal heuristic whose W/D/L-shaped values are not measured win rates. Load a trusted local model
explicitly:

```bash
GOMOKU_CHECKPOINT=runs/production/checkpoints/PROMOTED_CHECKPOINT.pt \
  uv run gomoku-api
curl http://127.0.0.1:8080/api/model
```

The API is intended for analysis and demonstration; public deployments should enforce request
size, search-budget, concurrency, and timeout limits.

The production-like CPU container listens on port 8080 / CPU 容器监听 8080 端口：

```bash
docker build \
  --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  -t gomoku-zero .
docker run --rm -p 8080:8080 -e PORT=8080 gomoku-zero
```

## Train and evaluate / 训练与评估

Use the smoke profile before starting an expensive run / 先用 smoke 配置验证完整链路：

```bash
uv run gomoku-train --config configs/smoke.json --output-dir runs/smoke
uv run gomoku-evaluate --config configs/smoke.json \
  --checkpoint runs/smoke/checkpoints/latest.pt \
  --output-dir runs/smoke/evaluation
uv run gomoku-convergence --config configs/smoke.json \
  --checkpoint runs/smoke/checkpoints/latest.pt \
  --positions examples/convergence-positions.json \
  --output runs/smoke/convergence.json
```

The bounded, resumable phase-1 production profile targets **20,000 self-play games** (`100 × 200`)
and a separate
**6,000-game independent evaluation**. These are different from MCTS simulations per move.
Do not combine 6,000 evaluation games with 6,000 simulations on every move without a cost
estimate: that can require billions of neural-network leaf evaluations. A bounded formal
evaluation can use 6,000 games at a fixed 800 simulations per move, while selected positions
use the convergence ladder `128 → 512 → 2,048 → 6,000 → 12,000`.

The configured promotion gate additionally plays 200 candidate-versus-champion games every ten
iterations at 200 simulations per move—2,000 matches across ten gates. Training preflight reports
self-play and promotion work separately; the later evaluation is still a separate budget.

有界、可恢复的第一阶段生产配置目标是 **20,000 局 self-play**（`100 × 200`），之后另做 **6,000 局
独立评估**。这与“每步 6,000 次 MCTS simulation”是两个完全不同的参数。若把两者同时
开到 6,000 而不先估算成本，可能产生数十亿次神经网络叶节点推理。正式评估可固定为
每步 800 次搜索、共 6,000 局；仅对选定局面使用
`128 → 512 → 2,048 → 6,000 → 12,000` 的预算收敛实验。
当前晋级门槛每十轮额外运行 200 局、每步搜索 200 次，100 轮共十次、合计 2,000 局
候选模型对战；训练 preflight 会分别报告 self-play 与晋级对局，最终评估仍需单独预算。

Run the tiny smoke ladder on selected positions without restarting the tree at each budget. Swap
in `configs/production.json` only when you deliberately want the full
`128 → 512 → 2,048 → 6,000 → 12,000` ladder / 对选定局面复用同一搜索树运行预算收敛实验：

```bash
uv run gomoku-convergence --config configs/smoke.json \
  --checkpoint runs/smoke/checkpoints/latest.pt \
  --positions examples/convergence-positions.json \
  --output runs/smoke/convergence.json
```

The report records legal Top 1/Top 3 visit shares and absolute black/draw/white WDL at every rung.
The production profile checks 6,000 against 12,000 simulations; smaller profiles compare their
last two rungs for Top-1/Top-3 stability and a sub-one-percentage-point maximum WDL change.
Terminal positions are returned immediately with exact WDL and no search work.

```bash
uv run gomoku-train --config configs/production.json \
  --output-dir runs/production \
  --preflight-only \
  --allow-slow-production

# Starts the acknowledged expensive job; omit --skip-promotion in a formal run.
uv run gomoku-train --config configs/production.json \
  --output-dir runs/production \
  --allow-slow-production

uv run gomoku-evaluate --config configs/production.json \
  --checkpoint runs/production/checkpoints/PROMOTED_CHECKPOINT.pt \
  --output-dir runs/production/evaluation \
  --preflight-only \
  --allow-slow-evaluation

# Starts the acknowledged expensive evaluation.
uv run gomoku-evaluate --config configs/production.json \
  --checkpoint runs/production/checkpoints/PROMOTED_CHECKPOINT.pt \
  --output-dir runs/production/evaluation \
  --allow-slow-evaluation

# Selected positions only: cumulative 128→512→2048→6000→12000 search.
uv run gomoku-convergence --config configs/production.json \
  --checkpoint runs/production/checkpoints/PROMOTED_CHECKPOINT.pt \
  --positions positions.json \
  --output runs/production/convergence.json
```

Training writes checkpoints below the chosen run directory. Resume with
`--resume runs/production/checkpoints/latest.pt`. Keep checkpoints and run metadata in durable
object storage, test resume with the smoke profile, and never overwrite the last known good
checkpoint. See [Training](docs/TRAINING.md) and [Model card](docs/MODEL_CARD.md).

The GPU-oriented Phase 1 profile at `configs/gpu-batched.json` replaces per-leaf
`[1, 3, 15, 15]` forwards with real cross-game batches. Its topology, telemetry,
submission guardrails, and remaining single-process PUCT limitation are documented in
[Self-play scaling](docs/SCALING.md). The historical `configs/eight-hour.json` remains unchanged
as exact run provenance.

训练会把 checkpoint 写入指定运行目录。正式训练前应把 checkpoint 与运行元数据同步到
持久对象存储，用 smoke 配置验证断点恢复，并保留最后一个已知可用版本。详见
[训练说明](docs/TRAINING.md)和[模型卡](docs/MODEL_CARD.md)。

面向 GPU 的 `configs/gpu-batched.json` 会把逐叶 `[1, 3, 15, 15]` 前向改为真实的跨对局
批量推理；拓扑、遥测、提交保护和当前单进程 PUCT 的限制见
[自博弈扩展说明](docs/SCALING.md)。历史 `configs/eight-hour.json` 保持不变，作为本次实验
可复现来源。

## Deploy publicly on Google Cloud / 公开部署到 Google Cloud

The checked-in deployment script enables the required APIs, creates a private Artifact Registry
repository when needed, builds the CPU image with Cloud Build, and makes the Cloud Run URL
publicly callable:

```bash
gcloud auth login
bash deploy/deploy_cloud_run.sh \
  --project-id YOUR_PROJECT_ID \
  --repository gomoku \
  --service gomoku-zero

# Load a deliberately promoted model from a private GCS object:
bash deploy/deploy_cloud_run.sh \
  --project-id YOUR_PROJECT_ID \
  --repository gomoku \
  --service-account RUNTIME_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --checkpoint-uri gs://YOUR_PRIVATE_BUCKET/models/PROMOTED_CHECKPOINT.pt
```

`--allow-unauthenticated` is an explicit public-access decision. Do not bake cloud credentials,
training data, or private checkpoints into the image. Prefer a deliberately promoted inference
checkpoint; keep training workers private. Public deployment instructions make the software
deployable by anyone, but do not grant strangers access to your Google Cloud project.

`--allow-unauthenticated` 会明确开放公网访问。不要把云凭据、训练数据或私有 checkpoint
打进镜像；API 应只加载经过明确晋级的推理 checkpoint，训练 worker 保持私有。公开部署
文档意味着任何人都能部署自己的副本，并不意味着向陌生人开放你的 Google Cloud 项目。

Production training is a separate, billable Vertex AI job. The script prints its bounded resource
plan and refuses submission without explicit confirmation. A formal submission also requires a
clean committed Git revision; Cloud Build tags it by Git SHA and the Vertex job uses the resolved
image digest so the executed code cannot drift after submission:

```bash
bash deploy/submit_vertex_training.sh \
  --project-id YOUR_PROJECT_ID \
  --region asia-southeast1 \
  --repository gomoku \
  --job gomoku-zero-production \
  --bucket YOUR_PROJECT_ID-training \
  --service-account TRAINER_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com \
  --yes

bash deploy/check_training.sh \
  --project-id YOUR_PROJECT_ID \
  --region asia-southeast1 \
  --job gomoku-zero-production
```

When `--checkpoint-uri` is used, run Cloud Run under a least-privilege service account that can
read that one private object; do not make the bucket public. See [Training](docs/TRAINING.md) for
Vertex checkpoint/resume and monitoring details.

## Why mask before softmax? / 为什么必须先 mask 再 softmax？

The network always emits 225 policy logits. Occupied points and all actions after a terminal
position are illegal. The engine builds the legal set first, excludes illegal logits from the
softmax denominator, and normalizes only over legal actions. Consequently illegal policy mass,
MCTS children, visits, and recommendations are zero. Zeroing probabilities after a 225-way
softmax is incorrect because legal probabilities are no longer normalized over the legal set and
illegal logits still affect the training gradient.

网络固定输出 225 个 policy logits。引擎先生成合法动作集合，把已占点（以及终局后的所有
动作）从 softmax 分母中彻底排除，再只对合法动作归一化。因此非法动作的概率、MCTS
子节点、访问次数和推荐结果都必须为零。若先做 225 维 softmax 再把非法概率清零，合法
概率并未在合法集合上归一化，非法 logit 仍会污染训练梯度，这是错误实现。

See [Architecture](docs/ARCHITECTURE.md) for the exact equations and terminal-state handling.

## Repository guide / 仓库导航

- `src/gomoku_zero/`: game rules, model, PUCT, training, evaluation, and API.
- `configs/smoke.json`: tiny end-to-end validation profile; not a useful playing model.
- `configs/production.json`: costly training target; review its budget before running.
- `docs/ARCHITECTURE.md`: data flow and masking invariants.
- `docs/TRAINING.md`: staged training, evaluation, cost, and checkpoint operations.
- `docs/MODEL_CARD.md`: intended use, limitations, and evidence requirements.

## License

MIT
