# 五子棋高吞吐云训练与小时级 Checkpoint 计划

日期：2026-08-17  
状态：等待批准，尚未启动任何训练或计费资源  
项目：`gomoku-zero`  
推荐平台：Google Cloud Vertex AI，`asia-southeast1`

## Goal

在不降低每手 800 次 MCTS 搜索预算的前提下，把 self-play 从历史的神经网络推理 `batch=1` 升级为跨对局 GPU batching，并用更多完整对局和参数更新训练一个可验证的新模型。

这次实验同时追求四个结果：

1. **吞吐**：端到端 self-play 至少达到历史 CPU 基线的 2 倍，目标 3 倍。
2. **训练量**：先把完整 self-play 从 192 局提高到至少 2,400 局；通过中期门槛后再扩到 6,400 局。
3. **可靠性**：每个 UTC 小时生成一个真正可恢复、不可覆盖的 checkpoint bundle，最坏恢复损失不超过 75 分钟。
4. **棋力证据**：冻结最终模型后，通过成对换色评估、配对置信区间、战术回归和搜索收敛检查，再决定是否替换线上模型。

本计划不承诺模型必然变强。更多数据和更高吞吐只提高成功概率；是否变强必须由冻结评估证明。

## Current Baseline

上一轮已完成的 CPU 实验是本次唯一基线：

| 指标 | 历史值 |
|---|---:|
| 机器 | `n1-highcpu-32`，无 GPU |
| 模型 | 128 channels × 10 residual blocks |
| self-play | 192 局 |
| 原始训练位置 | 9,371 |
| optimizer updates | 1,000 |
| 每手 MCTS | 800 simulations |
| self-play 推理 batch | 1 |
| 纯 self-play 吞吐 | 约 39.5 局/小时 |
| 完整任务吞吐 | 约 32.9 局/小时 |
| 总墙钟 | 约 5 小时 50 分钟 |

当前代码已经有第一阶段的跨独立对局 lockstep batching：每棵树一次只准备一个叶子，把多个树的叶局面合成 `[B, 3, 15, 15]` 后一次送入 GPU。根节点也会合批，合法动作仍逐局执行 mask-before-softmax。

这解决了网络 `batch=1` 的主要浪费，但还存在两个已知上限：

- PUCT 选树仍在一个 Python 进程内，GPU 变快后可能转为 CPU 单核瓶颈。
- promotion 和冻结评估当前仍是串行 `batch=1`，正式评估前也要升级或分片。

## Recommended Outcome

### 最低可接受结果

- 目标硬件上达到 **≥80 self-play 局/小时**。
- `inference_batch_size=64` 时，平均有效 batch **≥32**；目标 **≥48**。
- 2,400 局训练完成，至少约为历史训练量的 12.5 倍。
- 12,500 次左右 optimizer updates，至少约为历史的 12.5 倍。
- checkpoint 强制中断恢复测试通过，训练记录无重复、无缺口。
- 候选模型不出现非法动作、NaN/Inf、critical tactical regression。

### 目标结果

- 达到 **≥120 局/小时**，即历史 CPU self-play 吞吐约 3 倍。
- 扩展到 6,400 局、32,000 optimizer updates。
- 对当前线上模型的 6,000 局正式主赛中，候选总 score ≥52%，且 pair-cluster 95% CI 下界 >50%。
- 6,000 与 12,000 次搜索的 Top-1、Top-3 和 WDL 漂移达到预声明收敛门槛。

## Hardware Choice

### 推荐：先在现有 GCP 做受控 A/B

2026-08-17 的 live quota 结果：

| Vertex Custom Training GPU | Regular | Spot / Flex quota |
|---|---:|---:|
| A100 40GB | 0 | 8 |
| T4 | 0 | 1 |
| L4 | 0 | 0 |
| A100 80GB | 0 | 0 |

因此不能直接提交 Regular GPU job。推荐比较：

1. **低成本对照**：`n1-standard-16 + 1×T4`，Spot。
2. **首选正式候选**：`a2-highgpu-1g + 1×A100 40GB`，Flex Start。

L4 对这个 3.2M 参数小网络通常更合适，但当前项目没有 L4 quota；可以并行申请额度，不把它作为本轮启动阻塞。

Vertex Flex Start 使用动态工作负载调度，排队时不计计算费，作业开始后不会像 Spot 一样被主动抢占；Spot 可能被回收，因此两条路径都必须依赖可验证的 checkpoint/resume。官方参考：[Flex Start](https://cloud.google.com/vertex-ai/docs/training/schedule-jobs-dws)、[Spot training](https://cloud.google.com/vertex-ai/docs/training/use-spot-vms)、[GPU 与机器兼容表](https://cloud.google.com/vertex-ai/docs/training/configure-compute)。

### 为什么不直接租最强 GPU

当前网络很小，显存不是瓶颈。若单进程 Python PUCT 已经占满 CPU，A100 会等待输入，H100 只会更贵。硬件必须按以下指标选择：

- 端到端 games/hour；
- 网络 positions/second；
- 平均真实 batch；
- GPU utilization；
- CPU PUCT 时间占比；
- 每 1,000 局成本。

### VPS 备选

如果之后提供固定 GPU VPS 账号，可在 RTX 4090/L40S 级机器上复用同一容器和 GCS checkpoint 协议。优点是固定价格和不被抢占；代价是驱动、守护进程、磁盘、网络和故障恢复由我们自己维护。当前 GCP 已有 bucket、IAM、镜像仓库和审计链，因此本轮优先 Vertex。

## Experiment Design

### Phase 0 — 启动前工程门槛

不运行计费训练，先完成：

1. 构建固定 CUDA/PyTorch 版本的训练镜像；记录 Git SHA、镜像 digest 和依赖锁。
2. 提交器增加 `FLEX_START` / `SPOT`、`maxWaitDuration` 和 GPU/config fail-fast；当前提交器只能创建 Regular 调度，不能消费现有 GPU quota。
3. 实现下文的 checkpoint v2，并通过本地和小型云恢复演练。
4. 为 throughput、recovery 和 formal run 分别建立固定配置，禁止在任务启动后静默修改。
5. 正式评估前补齐跨 shard 聚合、pair-cluster CI 和 batched evaluation；不把逐局 Wilson 区间冒充配对置信区间。

验收：完整测试通过；提交器 `--plan-only` 显示正确机器、GPU、调度策略、时间上限、bucket、service account、Git SHA 和镜像 digest。

### Phase 1 — 硬件与 batch 吞吐 A/B

目标：回答“GPU 真的提升了完整 self-play，还是瓶颈已转到 Python PUCT”。

保持不变：

- 同一个冻结模型 checkpoint；
- 128×10 网络；
- 每手 800 MCTS simulations；
- 相同 96 个 game seed；
- 不训练、不 promotion；
- 相同温度、Dirichlet、规则和随机种子协议。

步骤：

1. 在 T4 Spot 与 A100 Flex 上各跑一次 `lanes=48 / inference batch=48`。
2. 在吞吐/美元更好的硬件上继续跑 `32/32`、`64/64`、`96/96`。
3. 只有 64→96 仍提升 ≥15% 时才测 128；否则选择更小 batch。

预计：约 4–8 小时，pilot 总预算硬上限 US$15。价格会随区域和调度模式变化，提交前以 Google 当时报价重新计算；官方价格入口：[DWS pricing](https://cloud.google.com/products/dws/pricing)、[Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing)。

进入下一阶段的门槛：

- 绝对下限：≥80 局/小时；目标：≥120 局/小时。
- batch 64 的平均有效 batch ≥32，目标 ≥48。
- 无 OOM、NaN/Inf、非法动作、非法 policy mass。
- GPU 显存峰值 <80%。
- 同 seed 的规则结果、合法性和训练目标与串行路径一致。

若低于 80 局/小时，不开始长跑。先进入 Phase 2 扩展改造：多 CPU actor 各自持有搜索树，中央 GPU 动态 batcher，使用共享内存返回 policy/WDL；promotion/evaluation 按模型身份合批。

### Phase 2 — 8 小时恢复性试跑

目标：验证新吞吐路径在真实训练、保存和恢复条件下稳定，而不把 pilot 成功直接等同于正式可运行。

推荐 workload：

- 8 iterations × 64 games = 512 self-play games；
- 800 simulations/move；
- 320 optimizer steps/iteration，共 2,560 steps；
- optimizer minibatch 256；
- 使用 Phase 1 胜出的 lanes/inference batch；
- 跳过正式 promotion，只做最多 40 局的非发布 smoke arena。

恢复演练：

1. 生成至少一个 committed hourly bundle。
2. 在 self-play 或 optimizer 安全块之间主动终止 worker。
3. 创建新 worker，从 GCS 自动恢复。
4. 验证模型、optimizer、scheduler、RNG、replay、metrics 和 game IDs 连续。
5. 继续至少一个完整 iteration。

预计 3–8 小时，硬上限 10 小时。恢复后的吞吐必须 ≥恢复前基准的 90%。

### Phase 3A — 推荐的 48 小时正式段

先做一个有界但明显大于历史实验的正式段：

- 目标约 2,400 self-play games；
- 800 simulations/move；
- 约 12,500 optimizer updates；
- minibatch 256；
- replay capacity 1,000,000；
- 从当前线上 immutable champion 权重 warm-start；建立新的 run ID，不能伪装成旧 run 的 strict resume；
- 旧 replay 只有通过 schema、config 和 digest 检查后才可作为 seed data；
- 每新增 2,000–5,000 局才进行 promotion，小时 checkpoint 不触发晋级。

这相对历史 192 局/1,000 updates 约增加 12.5 倍。A100 Flex 48 小时按当前公开价格量级约 US$125，含构建、磁盘、存储、日志和 20% 缓冲后，推荐总硬上限 US$160；以提交前的实时价格为准。

48 小时结束后必须先冻结、评估和复盘，不自动继续烧预算。

### Phase 3B — 条件扩展到 6,400 局

只有同时满足以下条件才从 3A 的 committed checkpoint 延长：

- 完成吞吐 ≥120 局/小时，或实测 ETA 能在累计 96 小时内完成；
- 最近 6 小时吞吐下降 <20%；
- checkpoint 恢复与哈希验证持续正常；
- 至少一次 promotion 通过；
- 没有 critical regression 或数值异常；
- 用户批准新增预算。

扩展目标：

- 100 iterations × 64 games = 6,400 self-play games；
- 320 optimizer steps/iteration = 32,000 updates；
- learning-rate milestones 50 / 80 / 95；
- 累计训练硬上限 96 小时。

按实测吞吐估算：

| Self-play 吞吐 | 6,400 局完整训练粗估 |
|---:|---:|
| 80 局/小时 | 90–106 小时 |
| 120 局/小时 | 65–75 小时 |
| 200 局/小时 | 40–48 小时 |

不直接启动 20,000 局。单 GPU、Phase 1 batching 下预计仍需约 5–11 天；应先完成多 actor 中央推理架构，或把 self-play 分布式化。

### Phase 4 — Promotion、冻结评估与搜索收敛

#### Promotion

- 每新增 2,000–5,000 局 self-play 才运行；
- 800 局 / 400 个 opening pairs；
- 候选在同 opening 下分别执黑、执白；
- 200–400 simulations/move；
- score ≥55%，且单侧 95% pair-cluster CI 下界 >50%；
- critical tactical regression 零失败。

Promotion suite 与最终 evaluation suite 必须隔离。

#### 正式主赛

- 最终 checkpoint 只冻结一次；
- 对当前线上模型 6,000 局 / 3,000 opening pairs；
- 800 simulations/move；
- 24 shards，每 shard 125 pairs；
- 以 opening pair 为 cluster 做固定 seed、10,000 次分层 bootstrap；
- 发布门槛：candidate score ≥52%，双侧 95% CI 下界 >50%。

3,000 个 opening pairs 在最坏 50% 附近的 95% 半宽约 1.8 个百分点。不能错误地把 6,000 个相关对局当作 6,000 个独立样本。

#### Search convergence

使用 64 个冻结局面、每个 3 个独立 MCTS seed，复用搜索树记录：

`40 → 128 → 200 → 512 → 800 → 2048 → 6000 → 12000`

发布门槛：

- 排除预声明的近似并列局面后，Top-1 稳定率 ≥95%；
- Top-3 集合稳定率 ≥90%；
- 6000 vs 12000 的 WDL 最大绝对差 p95 ≤1 个百分点；
- critical tactical positions 在三个 seed 下全部通过。

这验证的是“当前模型的搜索是否稳定”，不是五子棋数学真值，也不是胜率已经校准。未完成 Brier/ECE 校准前，不声明网页 WDL 是校准概率。

## Checkpoint v2

### 为什么现有机制不够

当前 `checkpoint_every=1` 表示“每个 iteration 保存”，不是每小时。`SYNC_INTERVAL_SECONDS=300` 只是每五分钟把已经存在的文件同步到 GCS，不会创建新的训练状态。上一轮保存间隔曾超过两小时。

### 时间与恢复目标

- 每个 UTC 小时在下一个安全块生成一个完整 bundle。
- self-play、optimizer 和 promotion 拆成 10–15 分钟以内的恢复块。
- 不序列化正在搜索的 MCTS 树；未完成对局按固定 seed 重跑。
- RPO：≤75 分钟；RTO：≤15 分钟。
- promotion 前后、学习率里程碑、正常结束和终止信号时额外保存。
- checkpoint 超过 75 分钟未提交，或连续两次上传/校验失败，暂停训练。

### Bundle 内容

每个不可变目录包含：

- `training.pt`：learner、champion、optimizer、scheduler、scaler、RNG、phase/global step；
- `replay.npz` 与 replay metadata；
- committed metrics 和 self-play game index；
- `manifest.json`。

Manifest 必须记录：

- 所有文件的 SHA-256、大小和 GCS generation；
- Git SHA、image digest、config digest、run ID；
- iteration、phase、global step、replay sample count；
- learner/champion digest；
- 上一个 manifest hash。

提交顺序：数据文件 → manifest commit marker → 小型 `latest.json`。更新 `latest.json` 使用 GCS generation precondition，旧 uploader 不能覆盖新 pointer。恢复时固定 generation 下载并逐个校验；缺文件、metrics 超前或哈希错误一律 fail closed。

### Retention

推荐默认：

- hourly：最近 72 个；
- daily：每天一个，保留 30 天；
- milestone：promotion、LR 变化、正式评估、部署和 final，永久保留；
- 当前线上与上一线上模型永久保留以便回滚。

清理由每日 retention job 完成，先保护 `latest` 和 milestone，再删除无引用 bundle；不能用一个简单 GCS lifecycle 规则直接删除目录内容。

按当前压缩率估算，1M replay 的一个 bundle 约 168 MB；72 hourly + 30 daily + 20 milestones 约 20.5 GB。启动前必须用目标 replay 做一次实际保存时长和压缩率基准。

### 必测故障场景

- self-play 中断与恢复；
- optimizer 中断与恢复；
- replay 或 manifest 被删除/损坏时拒绝恢复；
- 两个 uploader 竞争时旧 pointer 更新失败；
- 从前一个 hourly bundle 手动回滚并继续；
- game ID、metrics、样本数量无重复、无缺口。

## Implementation

批准后按以下工作包执行，先完成代码和验证，再启动 pilot：

1. **Cloud scheduling**
   - 修改 `deploy/submit_vertex_training.sh`，增加 Spot/Flex Start、等待上限和计费前 fail-fast。
   - 生成 CUDA training image，锁定 PyTorch/CUDA 版本并以 digest 提交。
2. **Checkpoint core**
   - 在训练层新增 wall-clock checkpoint coordinator 和 phase cursor。
   - self-play 增量提交 completed game IDs；optimizer 每 50–100 steps 形成安全点。
   - 使用不可变 bundle、manifest hash chain、GCS generation precondition。
3. **Resume and retention**
   - Vertex entrypoint 捕获终止信号，在最近安全点提交。
   - 增加 manifest-aware restore 和每日 retention job。
4. **Experiment configs**
   - 建立 T4/A100 throughput、8h recovery、48h formal 与 96h extension 配置。
   - 所有配置记录 seed、checkpoint SHA、镜像 digest 和预算上限。
5. **Evaluation scale-out**
   - promotion/evaluation 跨独立对局 batching。
   - 增加 shard 全局聚合、opening-suite hash 和 pair-cluster bootstrap CI。
6. **Verification**
   - 单元/集成测试、CUDA smoke、故障注入恢复、`--plan-only` 云提交审计。
   - 只在全部启动门槛通过后提交 Phase 1 pilot。

## Monitoring and Stop Conditions

每小时记录并画图：

- games/hour、positions/second；
- configured batch 与 mean/max effective batch；
- GPU utilization、显存、CPU utilization；
- self-play、optimizer、promotion、checkpoint 各阶段墙钟；
- replay size、loss、gradient norm、学习率；
- 黑/和/白 self-play 结果，仅作训练诊断；
- 每千局与每百万叶节点成本。

立即停止：

- 非法动作或非法 policy mass；
- NaN/Inf loss、gradient 或参数；
- checkpoint/replay digest 不匹配；
- committed bundle 无法恢复；
- 重复 OOM；
- 达到时间或预算硬上限。

停止扩展并回到性能改造：

- 调整一次 lanes/batch 后仍 <80 局/小时；
- batch 64 的平均有效 batch <32；
- 连续两小时吞吐比 recovery run 下降 >30%；
- GPU 空闲但 Python PUCT 持续成为主瓶颈。

模型策略：

- 候选未通过 promotion，则 champion 不变；
- 连续三次 promotion 无晋级，提前停止并分析 replay、搜索和学习率；
- 单次 loss 上升不直接回滚，必须结合持续数值异常或棋力退化。

## User Decisions

### 推荐默认选择

1. 平台：Vertex AI 新加坡区。
2. Pilot：T4 Spot 与 A100 Flex 各做固定 workload 对照，pilot 总预算上限 US$15。
3. 正式段：先 48 小时 / 约 2,400 局，预算上限 US$160。
4. 扩展：通过中期门槛后，再单独批准延长到累计 96 小时 / 6,400 局。
5. 初始化：从当前线上 immutable champion 权重 warm-start，新的 optimizer/run ID。
6. Checkpoint：72 个 hourly + 30 个 daily + 所有 milestone 永久保留。
7. 发布：只有 6,000 局 paired evaluation、回归和收敛门槛全部通过后才更新线上。

需要用户决定或明确接受：

- 是否按上述推荐默认值执行；
- 48 小时结束后是否允许自动延长，还是必须再次确认；推荐再次确认。
- 是否同时申请 L4 quota；推荐申请，但不阻塞 T4/A100 pilot。

## Risks and Assumptions

- A100 可能因 Python PUCT 而低利用率；pilot 的目的正是识别这个边界。
- Spot/Flex 的排队和库存不可保证；等待时间与实际价格要在提交前重新确认。
- 训练量扩大不保证棋力提升；最终 arena 是唯一发布证据。
- checkpoint v2 需要改变训练阶段的可恢复粒度，必须先做故障注入，不能只靠 happy-path 测试。
- promotion 和 6,000 局评估可能成为新的 `batch=1` 瓶颈，因此正式评估前必须合批/分片。
- 48 小时与 96 小时成本估算不包含不可预测的大量重跑；预算硬上限优先于完成局数。
- 当前模型保持 128×10，是为了先隔离吞吐与数据量变量；本轮不同时加深网络。若本轮成功，下一实验再独立比较 128×20 或 192×12，避免无法判断改进来源。

## Execution Gate

本计划依据“先给方案、后启动”的要求，到此停止。当前没有启动任何 Vertex/VPS 训练任务，也没有产生新的训练费用。

用户回复“按推荐方案执行”后，先实施 Phase 0 并提交一份最终 `--plan-only` 预览；只有代码验证、小时 checkpoint 恢复演练和预算审计都通过，才启动 Phase 1 pilot。Pilot 结束后将报告真实吞吐、成本和推荐机型；在得到第二次确认前，不启动 48 小时正式训练。
