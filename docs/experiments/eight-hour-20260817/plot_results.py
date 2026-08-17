"""Regenerate the checked-in experiment figures from the exact raw metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "metrics.jsonl"
FIGURES = ROOT / "figures"
EXPECTED_METRICS_SHA256 = "3673a5bd79d34f3b7b48af59e44748326b51cb9ad94e3931c1de265afeae7ac0"

NAVY = "#14213D"
BLUE = "#2F6BFF"
TEAL = "#12A594"
CORAL = "#F56B5D"
GOLD = "#E3A624"
SLATE = "#637083"
GRID = "#DCE2EA"
PAPER = "#F8FAFC"


def load_metrics() -> list[dict]:
    payload = METRICS.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_METRICS_SHA256:
        raise RuntimeError(f"metrics.jsonl SHA-256 mismatch: {digest}")
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.titleweight": 700,
            "axes.labelcolor": NAVY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": NAVY,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.85,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def polish(axis: plt.Axes) -> None:
    axis.grid(axis="y")
    axis.spines[["top", "right"]].set_visible(False)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True))


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight")
    svg_path = FIGURES / f"{name}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    # Matplotlib emits spaces at the end of multiline SVG path commands.
    # Normalize them so generated assets pass Git whitespace checks.
    lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    plt.close(fig)


def training_curves(rows: list[dict]) -> None:
    iterations = [row["iteration"] for row in rows]
    total = [row["loss"] for row in rows]
    policy = [row["policy_loss"] for row in rows]
    wdl = [row["wdl_loss"] for row in rows]
    gradients = [row["gradient_norm"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    fig.suptitle(
        "Training signal across four iterations", x=0.055, ha="left", fontsize=18, fontweight=700
    )
    fig.text(
        0.055,
        0.925,
        "1,000 optimizer steps · learning rate 1e−3 → 2e−4 at iteration 3",
        color=SLATE,
    )

    ax = axes[0]
    ax.plot(iterations, total, color=BLUE, marker="o", linewidth=2.6, label="Total loss")
    ax.plot(iterations, policy, color=TEAL, marker="o", linewidth=2.6, label="Policy loss")
    ax.fill_between(iterations, policy, total, color=BLUE, alpha=0.08)
    ax.set_title("Loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_xticks(iterations)
    ax.legend(loc="upper right")
    polish(ax)
    for x, value in zip(iterations, total, strict=True):
        ax.annotate(
            f"{value:.3f}",
            (x, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=BLUE,
            fontsize=8.5,
        )

    ax = axes[1]
    ax.plot(iterations, wdl, color=CORAL, marker="o", linewidth=2.6, label="WDL loss")
    ax.set_title("Value learning and gradient scale")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("WDL loss", color=CORAL)
    ax.tick_params(axis="y", labelcolor=CORAL)
    ax.set_xticks(iterations)
    polish(ax)
    twin = ax.twinx()
    twin.plot(iterations, gradients, color=GOLD, marker="s", linewidth=2.2, label="Gradient norm")
    twin.set_ylabel("Gradient norm", color=GOLD)
    twin.tick_params(axis="y", labelcolor=GOLD)
    twin.spines["top"].set_visible(False)
    lines = ax.lines + twin.lines
    ax.legend(lines, [line.get_label() for line in lines], loc="upper right")
    for x, value in zip(iterations, wdl, strict=True):
        ax.annotate(
            f"{value:.3f}",
            (x, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            color=CORAL,
            fontsize=8.5,
        )

    save(fig, "training-curves")


def outcomes_and_samples(rows: list[dict]) -> None:
    iterations = [row["iteration"] for row in rows]
    black = [row["outcomes"]["black"] for row in rows]
    draw = [row["outcomes"]["draw"] for row in rows]
    white = [row["outcomes"]["white"] for row in rows]
    generated = [row["samples_generated"] for row in rows]
    replay = [row["replay_samples"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    fig.suptitle("Self-play production", x=0.055, ha="left", fontsize=18, fontweight=700)
    fig.text(
        0.055,
        0.925,
        "192 games · 9,371 positions · changing model state each iteration",
        color=SLATE,
    )

    ax = axes[0]
    ax.bar(iterations, black, color=NAVY, label="Black wins")
    ax.bar(iterations, white, bottom=black, color=CORAL, label="White wins")
    ax.bar(
        iterations,
        draw,
        bottom=[b + w for b, w in zip(black, white, strict=True)],
        color=GOLD,
        label="Draws",
    )
    ax.axhline(48, color=GRID, linewidth=1)
    ax.set_title("Outcomes per iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Games")
    ax.set_xticks(iterations)
    ax.set_ylim(0, 53)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    polish(ax)
    for x, b, w in zip(iterations, black, white, strict=True):
        ax.text(x, b / 2, str(b), ha="center", va="center", color="white", fontweight=700)
        ax.text(x, b + w / 2, str(w), ha="center", va="center", color="white", fontweight=700)

    ax = axes[1]
    bars = ax.bar(iterations, generated, color=TEAL, alpha=0.88, label="New positions")
    ax.set_title("New samples and replay growth")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("New positions", color=TEAL)
    ax.tick_params(axis="y", labelcolor=TEAL)
    ax.set_xticks(iterations)
    polish(ax)
    twin = ax.twinx()
    twin.plot(iterations, replay, color=BLUE, marker="o", linewidth=2.6, label="Replay samples")
    twin.set_ylabel("Cumulative replay", color=BLUE)
    twin.tick_params(axis="y", labelcolor=BLUE)
    twin.spines["top"].set_visible(False)
    twin.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))
    for bar, value in zip(bars, generated, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 55,
            f"{value:,}",
            ha="center",
            color=TEAL,
            fontsize=8.5,
        )
    lines = [bars, twin.lines[0]]
    ax.legend(lines, ["New positions", "Replay samples"], loc="upper right")

    save(fig, "outcomes-and-samples")


def runtime_and_promotion(rows: list[dict]) -> None:
    iterations = [row["iteration"] for row in rows]
    hours = [row["seconds"] / 3600 for row in rows]
    cumulative = []
    running = 0.0
    for value in hours:
        running += value
        cumulative.append(running)

    report = rows[-1]["promotion"]
    rate = report["candidate_score"]
    low, high = report["win_rate_wilson_interval"]
    threshold = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0), constrained_layout=True)
    fig.suptitle("Runtime and promotion gate", x=0.055, ha="left", fontsize=18, fontweight=700)
    fig.text(
        0.055,
        0.925,
        "Measured iteration time includes self-play, learning, checkpoints, and the scheduled gate",
        color=SLATE,
    )

    ax = axes[0]
    colors = [TEAL, TEAL, TEAL, GOLD]
    bars = ax.bar(iterations, hours, color=colors, alpha=0.9, label="Iteration hours")
    ax.set_title("Where the 5.84 measured hours went")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Hours per iteration")
    ax.set_xticks(iterations)
    polish(ax)
    twin = ax.twinx()
    twin.plot(
        iterations, cumulative, color=BLUE, marker="o", linewidth=2.5, label="Cumulative hours"
    )
    twin.set_ylabel("Cumulative hours", color=BLUE)
    twin.tick_params(axis="y", labelcolor=BLUE)
    twin.spines["top"].set_visible(False)
    for bar, value in zip(bars, hours, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.045,
            f"{value:.2f}h",
            ha="center",
            fontsize=8.5,
        )
    ax.annotate(
        "20-game gate",
        xy=(4, hours[-1]),
        xytext=(3.0, 1.82),
        arrowprops={"arrowstyle": "->", "color": SLATE},
        color=SLATE,
    )

    ax = axes[1]
    ax.set_title("Candidate promotion: 11 wins, 9 losses")
    ax.axvline(
        threshold * 100, color=CORAL, linestyle="--", linewidth=2, label="Promotion threshold"
    )
    ax.errorbar(
        rate * 100,
        0,
        xerr=[[rate * 100 - low * 100], [high * 100 - rate * 100]],
        fmt="o",
        color=BLUE,
        ecolor=BLUE,
        elinewidth=8,
        capsize=8,
        markersize=10,
        label="Candidate score (95% Wilson CI)",
    )
    ax.set_xlim(20, 80)
    ax.set_ylim(-0.65, 0.65)
    ax.set_yticks([])
    ax.set_xlabel("Candidate score (%)")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(loc="upper center")
    ax.text(rate * 100, -0.23, "55.0%", ha="center", fontweight=700, color=BLUE)
    ax.text((low + high) * 50, -0.43, "34.2%–74.2% · only 20 games", ha="center", color=SLATE)
    ax.text(55, 0.29, "Promoted exactly at threshold", ha="center", color=CORAL, fontsize=9)

    save(fig, "runtime-and-promotion")


def main() -> None:
    configure()
    rows = load_metrics()
    training_curves(rows)
    outcomes_and_samples(rows)
    runtime_and_promotion(rows)


if __name__ == "__main__":
    main()
