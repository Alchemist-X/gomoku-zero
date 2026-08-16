const BOARD_SIZE = 15;
const BOARD_CELLS = BOARD_SIZE * BOARD_SIZE;
const EMPTY = 0;
const BLACK = 1;
const WHITE = -1;
const COLUMNS = "ABCDEFGHIJKLMNO";
const DEFAULT_BUDGETS = [40, 200, 1000, 3000, 6000];
const HASH_PREFIX = "gomoku-zero:position:v1|";

const elements = {
  board: document.querySelector("#board"),
  turnLabel: document.querySelector("#turn-label"),
  moveCount: document.querySelector("#move-count"),
  undoButton: document.querySelector("#undo-button"),
  resetButton: document.querySelector("#reset-button"),
  boardStatus: document.querySelector("#board-status"),
  serviceState: document.querySelector("#service-state"),
  serviceStateLabel: document.querySelector("#service-state-label"),
  modelPill: document.querySelector("#model-pill"),
  modeOptions: [...document.querySelectorAll(".mode-option")],
  analyzeButton: document.querySelector("#analyze-button"),
  analyzeButtonLabel: document.querySelector("#analyze-button-label"),
  analysisError: document.querySelector("#analysis-error"),
  progressSection: document.querySelector("#progress-section"),
  simulationCount: document.querySelector("#simulation-count"),
  budgetRail: document.querySelector("#budget-rail"),
  budgetProgress: document.querySelector("#budget-progress"),
  budgetStages: document.querySelector("#budget-stages"),
  outcomeEmpty: document.querySelector("#outcome-empty"),
  outcomeContent: document.querySelector("#outcome-content"),
  estimateKind: document.querySelector("#estimate-kind"),
  blackOutcome: document.querySelector("#black-outcome"),
  drawOutcome: document.querySelector("#draw-outcome"),
  whiteOutcome: document.querySelector("#white-outcome"),
  blackBar: document.querySelector("#black-bar"),
  drawBar: document.querySelector("#draw-bar"),
  whiteBar: document.querySelector("#white-bar"),
  candidateList: document.querySelector("#candidate-list"),
  provenanceToggle: document.querySelector("#provenance-toggle"),
  provenanceDetails: document.querySelector("#provenance-details"),
  modelId: document.querySelector("#model-id"),
  evaluatorName: document.querySelector("#evaluator-name"),
  checkpointName: document.querySelector("#checkpoint-name"),
  checkpointSha: document.querySelector("#checkpoint-sha"),
  modelWarning: document.querySelector("#model-warning"),
  positionHash: document.querySelector("#position-hash"),
};

const state = {
  board: Array(BOARD_CELLS).fill(EMPTY),
  history: [],
  mode: "instant",
  defaultBudgets: [...DEFAULT_BUDGETS],
  maxSimulations: DEFAULT_BUDGETS.at(-1),
  topMoves: [],
  running: false,
  controller: null,
  requestSerial: 0,
  model: null,
};

function boardKey(board = state.board) {
  return board.join(",");
}

function sideToPlay(board = state.board) {
  let black = 0;
  let white = 0;
  for (const cell of board) {
    if (cell === BLACK) black += 1;
    if (cell === WHITE) white += 1;
  }
  return black === white ? BLACK : WHITE;
}

function coordinateFor(index) {
  return `${COLUMNS[index % BOARD_SIZE]}${Math.floor(index / BOARD_SIZE) + 1}`;
}

function winnerFor(board = state.board) {
  const directions = [
    [0, 1],
    [1, 0],
    [1, 1],
    [1, -1],
  ];
  for (let row = 0; row < BOARD_SIZE; row += 1) {
    for (let col = 0; col < BOARD_SIZE; col += 1) {
      const stone = board[row * BOARD_SIZE + col];
      if (stone === EMPTY) continue;
      for (const [dr, dc] of directions) {
        const endRow = row + dr * 4;
        const endCol = col + dc * 4;
        if (endRow < 0 || endRow >= BOARD_SIZE || endCol < 0 || endCol >= BOARD_SIZE) {
          continue;
        }
        let connected = true;
        for (let distance = 1; distance < 5; distance += 1) {
          const index = (row + dr * distance) * BOARD_SIZE + col + dc * distance;
          if (board[index] !== stone) {
            connected = false;
            break;
          }
        }
        if (connected) return stone;
      }
    }
  }
  return EMPTY;
}

function isTerminal(board = state.board) {
  return winnerFor(board) !== EMPTY || !board.includes(EMPTY);
}

function lastMove() {
  return state.history.at(-1) ?? null;
}

function abortAnalysis() {
  state.requestSerial += 1;
  state.controller?.abort();
  state.controller = null;
  state.running = false;
  elements.analyzeButton.classList.remove("is-running");
  elements.analyzeButtonLabel.textContent = "开始分析";
}

function clearResults() {
  state.topMoves = [];
  elements.progressSection.hidden = true;
  elements.outcomeEmpty.hidden = false;
  elements.outcomeContent.hidden = true;
  elements.estimateKind.textContent = "等待分析";
  elements.candidateList.replaceChildren(candidatePlaceholder());
  elements.positionHash.textContent = "局面指纹：—";
}

function candidatePlaceholder() {
  const item = document.createElement("li");
  item.className = "candidate-placeholder";
  const rank = document.createElement("span");
  rank.textContent = "01";
  const copy = document.createElement("p");
  copy.textContent = "还没有推荐。搜索越深，候选顺序越稳定。";
  item.append(rank, copy);
  return item;
}

function renderBoard() {
  const fragment = document.createDocumentFragment();
  const currentWinner = winnerFor();
  const terminal = currentWinner !== EMPTY || !state.board.includes(EMPTY);
  const latest = lastMove();
  const recommendations = new Map(state.topMoves.map((move, index) => [move.move, index + 1]));

  state.board.forEach((cell, index) => {
    const button = document.createElement("button");
    const coordinate = coordinateFor(index);
    const recommendation = recommendations.get(index);
    button.type = "button";
    button.className = "intersection";
    button.id = `cell-${index}`;
    button.dataset.index = String(index);
    button.setAttribute("role", "gridcell");
    button.disabled = cell !== EMPTY || terminal;

    if (cell === EMPTY) {
      const suffix = recommendation ? `，候选第 ${recommendation} 位` : "，空位";
      button.setAttribute("aria-label", `${coordinate}${suffix}`);
    } else {
      const color = cell === BLACK ? "黑子" : "白子";
      button.setAttribute("aria-label", `${coordinate}，${color}`);
      const stone = document.createElement("span");
      stone.className = `stone stone--${cell === BLACK ? "black" : "white"}`;
      if (index === latest) stone.classList.add("stone--last");
      stone.setAttribute("aria-hidden", "true");
      button.append(stone);
    }

    if (recommendation && cell === EMPTY && !terminal) {
      const marker = document.createElement("span");
      marker.className = "recommendation-ring";
      marker.dataset.rank = String(recommendation);
      marker.textContent = String(recommendation);
      marker.setAttribute("aria-hidden", "true");
      button.append(marker);
    }

    button.addEventListener("click", () => placeStone(index));
    fragment.append(button);
  });

  elements.board.replaceChildren(fragment);
  const moves = state.history.length;
  const next = sideToPlay();
  elements.moveCount.textContent = String(moves);
  elements.undoButton.disabled = moves === 0;
  elements.resetButton.disabled = moves === 0;
  elements.analyzeButton.disabled = terminal;

  if (currentWinner === BLACK) {
    elements.turnLabel.textContent = "黑方已获胜";
    elements.boardStatus.textContent = "对局已结束，可悔棋或重置";
  } else if (currentWinner === WHITE) {
    elements.turnLabel.textContent = "白方已获胜";
    elements.boardStatus.textContent = "对局已结束，可悔棋或重置";
  } else if (terminal) {
    elements.turnLabel.textContent = "和棋";
    elements.boardStatus.textContent = "棋盘已满";
  } else {
    elements.turnLabel.textContent = next === BLACK ? "黑方落子" : "白方落子";
    if (!state.running) elements.boardStatus.textContent = moves === 0 ? "棋盘已就绪" : "局面已更新";
  }
}

function placeStone(index) {
  if (!Number.isInteger(index) || index < 0 || index >= BOARD_CELLS) return;
  if (state.board[index] !== EMPTY || isTerminal()) return;
  abortAnalysis();
  state.board[index] = sideToPlay();
  state.history.push(index);
  clearError();
  clearResults();
  renderBoard();
}

function undoMove() {
  const index = state.history.pop();
  if (index === undefined) return;
  abortAnalysis();
  state.board[index] = EMPTY;
  clearError();
  clearResults();
  renderBoard();
  document.querySelector(`#cell-${index}`)?.focus();
}

function resetBoard() {
  if (state.history.length === 0) return;
  abortAnalysis();
  state.board.fill(EMPTY);
  state.history = [];
  clearError();
  clearResults();
  renderBoard();
  elements.boardStatus.textContent = "棋盘已重置";
}

function setMode(mode) {
  if (mode !== "instant" && mode !== "deep") return;
  state.mode = mode;
  for (const option of elements.modeOptions) {
    const active = option.dataset.mode === mode;
    option.classList.toggle("is-active", active);
    option.setAttribute("aria-checked", String(active));
  }
}

function formatInteger(value) {
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }).format(value);
}

function formatPercent(value) {
  const safe = Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : 0;
  return `${(safe * 100).toFixed(1)}%`;
}

async function derivePositionHash(board, toPlay) {
  if (!globalThis.crypto?.subtle) return null;
  const canonical = `${HASH_PREFIX}${board.join(",")}|${toPlay}`;
  const data = new TextEncoder().encode(canonical);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function budgetsForMode() {
  if (state.mode === "instant") return [state.defaultBudgets[0] ?? 40];
  return [...state.defaultBudgets];
}

function setBudgetStages(budgets) {
  elements.budgetStages.replaceChildren(
    ...budgets.map((budget) => {
      const stage = document.createElement("span");
      stage.textContent = formatInteger(budget);
      return stage;
    }),
  );
  const maximum = budgets.at(-1) ?? 1;
  elements.budgetRail.setAttribute("aria-valuemax", String(maximum));
  elements.budgetRail.setAttribute("aria-valuenow", "0");
  elements.budgetProgress.style.width = "0%";
  elements.simulationCount.textContent = "0 simulations";
}

function validateFrame(frame, snapshot, expectedHash, budgets, streamState) {
  if (!frame || typeof frame !== "object") return "服务器返回了无效的分析帧。";
  if (frame.type === "error") {
    return frame.error?.message || "服务器未能完成分析。";
  }
  if (frame.type !== "analysis") return "服务器返回了未知的分析帧类型。";
  if (typeof frame.position_hash !== "string" || frame.position_hash.length !== 64) {
    return "分析帧缺少可验证的局面指纹。";
  }
  if (expectedHash && frame.position_hash !== expectedHash) {
    return "已丢弃与当前棋盘不匹配的搜索结果。";
  }
  if (streamState.positionHash && frame.position_hash !== streamState.positionHash) {
    return "同一搜索流的局面指纹发生了变化，结果已丢弃。";
  }
  if (!Array.isArray(frame.top_moves) || !frame.top_moves.every((move) => Number.isInteger(move.move))) {
    return "候选落点数据不完整。";
  }
  const expectedSequence = streamState.accepted + 1;
  const expectedBudget = budgets[streamState.accepted];
  if (
    !Number.isInteger(frame.simulations) ||
    frame.simulations !== expectedBudget ||
    frame.simulations <= streamState.lastSimulations
  ) {
    return "已拒绝重复、乱序或超出请求预算的搜索帧。";
  }
  if (frame.stage !== expectedSequence || frame.sequence !== expectedSequence) {
    return "搜索帧序号不连续，本次结果已丢弃。";
  }
  if (
    frame.stages !== budgets.length ||
    frame.target_simulations !== expectedBudget ||
    frame.complete !== (expectedSequence === budgets.length)
  ) {
    return "搜索帧的预算或完成标记与请求不一致。";
  }
  if (typeof frame.analysis_id !== "string" || frame.analysis_id.length === 0) {
    return "搜索帧缺少分析任务标识。";
  }
  if (streamState.analysisId && frame.analysis_id !== streamState.analysisId) {
    return "搜索流中混入了其他分析任务的结果。";
  }
  if (frame.to_play !== sideToPlay(snapshot)) return "搜索手方与请求局面不一致。";
  if (frame.legal_moves !== snapshot.filter((cell) => cell === EMPTY).length) {
    return "服务器与客户端的合法落点计数不一致。";
  }
  const seen = new Set();
  for (const move of frame.top_moves) {
    if (move.move < 0 || move.move >= BOARD_CELLS || snapshot[move.move] !== EMPTY || seen.has(move.move)) {
      return "服务器返回了非法或重复的候选点，本帧已拒绝。";
    }
    seen.add(move.move);
  }
  streamState.analysisId = frame.analysis_id;
  streamState.positionHash = frame.position_hash;
  streamState.lastSimulations = frame.simulations;
  streamState.accepted += 1;
  return null;
}

function renderFrame(frame, budgets) {
  const maximum = budgets.at(-1) ?? frame.simulations;
  const progress = maximum > 0 ? Math.min(100, (frame.simulations / maximum) * 100) : 0;
  elements.progressSection.hidden = false;
  elements.budgetProgress.style.width = `${progress}%`;
  elements.budgetRail.setAttribute("aria-valuenow", String(frame.simulations));
  elements.simulationCount.textContent = `${formatInteger(frame.simulations)} simulations`;
  elements.boardStatus.textContent = frame.complete
    ? `已完成 ${formatInteger(frame.simulations)} 次搜索`
    : `已累计 ${formatInteger(frame.simulations)} 次搜索`;

  renderOutcome(frame.outcome, frame.estimate_kind);
  renderCandidates(frame.top_moves);
  state.topMoves = frame.top_moves.map((move) => ({ ...move }));
  renderBoard();
  elements.positionHash.textContent = `局面指纹：${frame.position_hash}`;
  if (frame.model) updateModel(frame.model);
}

function renderOutcome(outcome, estimateKind) {
  if (!outcome || ![outcome.black, outcome.draw, outcome.white].every(Number.isFinite)) return;
  elements.outcomeEmpty.hidden = true;
  elements.outcomeContent.hidden = false;
  elements.estimateKind.textContent =
    estimateKind === "deterministic-heuristic"
      ? "未训练·启发式估计"
      : "检查点模型·MCTS";
  elements.blackOutcome.textContent = formatPercent(outcome.black);
  elements.drawOutcome.textContent = formatPercent(outcome.draw);
  elements.whiteOutcome.textContent = formatPercent(outcome.white);
  elements.blackBar.style.width = formatPercent(outcome.black);
  elements.drawBar.style.width = formatPercent(outcome.draw);
  elements.whiteBar.style.width = formatPercent(outcome.white);
}

function renderCandidates(moves) {
  if (!moves.length) {
    elements.candidateList.replaceChildren(candidatePlaceholder());
    return;
  }
  const items = moves.map((move, index) => {
    const item = document.createElement("li");
    const rank = document.createElement("span");
    rank.className = "candidate-rank";
    rank.textContent = String(index + 1).padStart(2, "0");

    const name = document.createElement("div");
    name.className = "candidate-name";
    const coordinate = document.createElement("strong");
    coordinate.textContent = move.coordinate ?? coordinateFor(move.move);
    const details = document.createElement("span");
    const q = Number.isFinite(move.q_value) ? `${move.q_value >= 0 ? "+" : ""}${move.q_value.toFixed(3)}` : "—";
    details.textContent = `Q ${q} · prior ${formatPercent(move.prior)}`;
    name.append(coordinate, details);

    const stat = document.createElement("div");
    stat.className = "candidate-stat";
    const share = document.createElement("strong");
    share.textContent = formatPercent(move.visit_share);
    const visits = document.createElement("span");
    visits.textContent = `${formatInteger(move.visits)} visits`;
    stat.append(share, visits);
    item.append(rank, name, stat);
    return item;
  });
  elements.candidateList.replaceChildren(...items);
}

function clearError() {
  elements.analysisError.hidden = true;
  elements.analysisError.textContent = "";
}

function showError(message) {
  elements.analysisError.textContent = message;
  elements.analysisError.hidden = false;
  elements.boardStatus.textContent = "分析未完成";
}

async function responseError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") return payload.detail;
    if (Array.isArray(payload.detail)) {
      return payload.detail.map((item) => item.msg).filter(Boolean).join("；");
    }
  } catch {
    // Fall back to the status line when an upstream proxy does not return JSON.
  }
  return `分析服务返回 ${response.status} ${response.statusText}`;
}

async function consumeNdjson(response, onFrame) {
  if (!response.body?.getReader) {
    const text = await response.text();
    for (const line of text.split("\n")) {
      if (line.trim()) onFrame(JSON.parse(line));
    }
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onFrame(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) onFrame(JSON.parse(buffer));
}

async function startAnalysis() {
  if (isTerminal()) {
    showError("当前对局已结束，没有合法候选点。");
    return;
  }

  abortAnalysis();
  clearError();
  state.topMoves = [];
  renderBoard();
  const serial = state.requestSerial;
  const snapshot = [...state.board];
  const snapshotKey = boardKey(snapshot);
  const toPlay = sideToPlay(snapshot);
  const budgets = budgetsForMode();
  const expectedHash = await derivePositionHash(snapshot, toPlay);

  if (serial !== state.requestSerial || snapshotKey !== boardKey()) return;
  const controller = new AbortController();
  state.controller = controller;
  state.running = true;
  elements.analyzeButton.classList.add("is-running");
  elements.analyzeButtonLabel.textContent = "取消分析";
  elements.boardStatus.textContent = "正在建立搜索树";
  elements.progressSection.hidden = false;
  setBudgetStages(budgets);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { Accept: "application/x-ndjson", "Content-Type": "application/json" },
      body: JSON.stringify({ board: snapshot, budgets, mode: state.mode }),
      signal: controller.signal,
    });
    if (!response.ok) throw new Error(await responseError(response));
    if (!response.headers.get("content-type")?.includes("application/x-ndjson")) {
      throw new Error("分析服务未返回 NDJSON 流。");
    }

    let received = 0;
    const streamState = {
      accepted: 0,
      lastSimulations: 0,
      analysisId: null,
      positionHash: null,
    };
    await consumeNdjson(response, (frame) => {
      if (serial !== state.requestSerial || snapshotKey !== boardKey()) return;
      const invalid = validateFrame(frame, snapshot, expectedHash, budgets, streamState);
      if (invalid) {
        controller.abort();
        throw new Error(invalid);
      }
      received += 1;
      renderFrame(frame, budgets);
    });
    if (serial === state.requestSerial && received !== budgets.length) {
      throw new Error("分析流在返回全部预算阶段前结束。");
    }
  } catch (error) {
    if (error?.name !== "AbortError" && serial === state.requestSerial) {
      showError(error instanceof Error ? error.message : "分析请求失败。");
    }
  } finally {
    if (serial === state.requestSerial) {
      state.running = false;
      state.controller = null;
      elements.analyzeButton.classList.remove("is-running");
      elements.analyzeButtonLabel.textContent = "重新分析";
    }
  }
}

function updateModel(model) {
  state.model = model;
  const bootstrap = model.status === "bootstrap-untrained" || model.trained === false;
  elements.modelPill.textContent = bootstrap ? "未训练启发式" : "已加载训练模型";
  elements.modelPill.classList.toggle("is-bootstrap", bootstrap);
  elements.modelId.textContent = model.model_id || model.status || "未知模型";
  elements.evaluatorName.textContent = model.evaluator || "—";
  elements.checkpointName.textContent = model.checkpoint || "未加载";
  elements.checkpointSha.textContent = model.checkpoint_sha256 || "—";
  elements.modelWarning.textContent =
    model.warning_zh || model.warning || "未提供额外模型说明。";
}

async function loadServiceMetadata() {
  try {
    const [healthResponse, modelResponse] = await Promise.all([fetch("/api/health"), fetch("/api/model")]);
    if (!healthResponse.ok || !modelResponse.ok) throw new Error("元数据请求失败");
    const health = await healthResponse.json();
    const model = await modelResponse.json();
    if (Array.isArray(health.default_budgets) && health.default_budgets.length) {
      state.defaultBudgets = health.default_budgets.filter(Number.isInteger);
    }
    if (Number.isInteger(health.max_simulations)) state.maxSimulations = health.max_simulations;
    const first = state.defaultBudgets[0] ?? 40;
    const last = state.defaultBudgets.at(-1) ?? state.maxSimulations;
    document.querySelector("#mode-instant small").textContent = `${formatInteger(first)} 次`;
    document.querySelector("#mode-deep small").textContent = `渐进至 ${formatInteger(last)} 次`;
    elements.serviceState.classList.add("is-ready");
    elements.serviceStateLabel.textContent = "分析服务已就绪";
    updateModel(model);
  } catch {
    elements.serviceState.classList.add("is-error");
    elements.serviceStateLabel.textContent = "分析服务不可用";
    elements.modelPill.textContent = "无法读取模型";
    showError("无法连接分析服务，请稍后重试。");
  }
}

elements.undoButton.addEventListener("click", undoMove);
elements.resetButton.addEventListener("click", resetBoard);
elements.analyzeButton.addEventListener("click", () => {
  if (state.running) abortAnalysis();
  else startAnalysis();
});
for (const option of elements.modeOptions) {
  option.addEventListener("click", () => setMode(option.dataset.mode));
  option.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const nextMode = state.mode === "instant" ? "deep" : "instant";
    setMode(nextMode);
    elements.modeOptions.find((item) => item.dataset.mode === nextMode)?.focus();
  });
}
elements.provenanceToggle.addEventListener("click", () => {
  const expanded = elements.provenanceToggle.getAttribute("aria-expanded") === "true";
  elements.provenanceToggle.setAttribute("aria-expanded", String(!expanded));
  elements.provenanceDetails.hidden = expanded;
});

renderBoard();
loadServiceMetadata();
