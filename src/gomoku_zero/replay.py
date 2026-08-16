"""Training samples, legal-policy validation, D4 augmentation, and replay."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

POLICY_SIZE = 225
WDL_SIZE = 3


def normalize_policy_target(
    weights: np.ndarray | Sequence[float],
    legal_mask: np.ndarray | Sequence[bool],
    *,
    illegal_tolerance: float = 1e-8,
) -> np.ndarray:
    """Mask and normalize visit weights, rejecting meaningful illegal mass.

    Tiny floating-point residue on illegal entries is scrubbed before legal
    entries are renormalized.  Material illegal mass is a data-integrity error,
    not something training should silently hide.
    """

    policy = np.asarray(weights, dtype=np.float64).reshape(-1)
    legal = np.asarray(legal_mask, dtype=np.bool_).reshape(-1)
    if policy.shape != (POLICY_SIZE,) or legal.shape != (POLICY_SIZE,):
        raise ValueError("policy target and legal mask must each have shape (225,)")
    if not np.all(np.isfinite(policy)):
        raise ValueError("policy target contains NaN or infinity")
    if np.any(policy < 0):
        raise ValueError("policy target contains negative weights")
    if not np.any(legal):
        raise ValueError("a policy target requires at least one legal action")
    illegal_mass = float(policy[~legal].sum(dtype=np.float64))
    if illegal_mass > illegal_tolerance:
        raise ValueError(f"policy target assigns {illegal_mass:.6g} mass to illegal actions")
    normalized = np.zeros(POLICY_SIZE, dtype=np.float64)
    legal_mass = float(policy[legal].sum(dtype=np.float64))
    if not np.isfinite(legal_mass) or legal_mass <= 0:
        raise ValueError("policy target has no positive legal mass")
    normalized[legal] = policy[legal] / legal_mass
    # A second normalization removes the last few ulps introduced by casting.
    result = normalized.astype(np.float32)
    result[~legal] = 0.0
    result[legal] /= result[legal].sum(dtype=np.float32)
    return result


def normalize_wdl_target(wdl: np.ndarray | Sequence[float]) -> np.ndarray:
    target = np.asarray(wdl, dtype=np.float64).reshape(-1)
    if target.shape != (WDL_SIZE,):
        raise ValueError("WDL target must have shape (3,) in [black, draw, white] order")
    if not np.all(np.isfinite(target)) or np.any(target < 0):
        raise ValueError("WDL target must contain finite, non-negative values")
    total = float(target.sum())
    if total <= 0:
        raise ValueError("WDL target must have positive mass")
    result = (target / total).astype(np.float32)
    result /= result.sum(dtype=np.float32)
    return result


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One pre-move AlphaZero training position.

    ``wdl_target`` is always absolute ``[black win, draw, white win]`` rather
    than being relative to the player to move.
    """

    encoded_state: np.ndarray
    policy_target: np.ndarray
    wdl_target: np.ndarray
    legal_mask: np.ndarray

    def __post_init__(self) -> None:
        state = np.asarray(self.encoded_state, dtype=np.float32)
        if state.ndim != 3 or state.shape[-2:] != (15, 15):
            raise ValueError("encoded state must have shape (channels, 15, 15)")
        if not np.all(np.isfinite(state)):
            raise ValueError("encoded state contains NaN or infinity")
        legal = np.asarray(self.legal_mask, dtype=np.bool_).reshape(-1)
        policy = normalize_policy_target(self.policy_target, legal)
        wdl = normalize_wdl_target(self.wdl_target)
        object.__setattr__(self, "encoded_state", np.ascontiguousarray(state))
        object.__setattr__(self, "legal_mask", np.ascontiguousarray(legal))
        object.__setattr__(self, "policy_target", np.ascontiguousarray(policy))
        object.__setattr__(self, "wdl_target", np.ascontiguousarray(wdl))


def _transform_plane(plane: np.ndarray, symmetry: int) -> np.ndarray:
    if symmetry not in range(8):
        raise ValueError("symmetry must be an integer from 0 through 7")
    value = plane
    if symmetry >= 4:
        value = np.flip(value, axis=-1)
    value = np.rot90(value, k=symmetry % 4, axes=(-2, -1))
    return np.ascontiguousarray(value)


def transform_sample(sample: TrainingSample, symmetry: int) -> TrainingSample:
    policy = _transform_plane(sample.policy_target.reshape(15, 15), symmetry).reshape(-1)
    legal = _transform_plane(sample.legal_mask.reshape(15, 15), symmetry).reshape(-1)
    return TrainingSample(
        encoded_state=_transform_plane(sample.encoded_state, symmetry),
        policy_target=policy,
        wdl_target=sample.wdl_target.copy(),
        legal_mask=legal,
    )


def all_symmetries(sample: TrainingSample) -> tuple[TrainingSample, ...]:
    return tuple(transform_sample(sample, symmetry) for symmetry in range(8))


class ReplayBuffer:
    """A deterministic bounded ring buffer with optional on-sample symmetry."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity must be > 0")
        self.capacity = int(capacity)
        self._samples: list[TrainingSample] = []
        self._next = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[TrainingSample]:
        if len(self._samples) < self.capacity or self._next == 0:
            yield from self._samples
            return
        yield from self._samples[self._next :]
        yield from self._samples[: self._next]

    def add(self, sample: TrainingSample) -> None:
        if not isinstance(sample, TrainingSample):
            raise TypeError("ReplayBuffer accepts TrainingSample instances")
        if len(self._samples) < self.capacity:
            self._samples.append(sample)
        else:
            self._samples[self._next] = sample
        self._next = (self._next + 1) % self.capacity

    def extend(self, samples: Iterable[TrainingSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self, batch_size: int, *, symmetry: str = "random") -> list[TrainingSample]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not self._samples:
            raise ValueError("cannot sample an empty replay buffer")
        if symmetry not in {"none", "random"}:
            raise ValueError("sample symmetry must be 'none' or 'random'")
        indices = self._rng.integers(0, len(self._samples), size=batch_size)
        result: list[TrainingSample] = []
        for index in indices:
            item = self._samples[int(index)]
            if symmetry == "random":
                item = transform_sample(item, int(self._rng.integers(0, 8)))
            result.append(item)
        return result

    def state_dict(self) -> dict[str, Any]:
        ordered = list(self)
        if ordered:
            # Board encodings are binary planes; uint8 keeps production resume
            # artifacts roughly four times smaller without information loss.
            states = np.stack([sample.encoded_state for sample in ordered]).astype(np.uint8)
            policies = np.stack([sample.policy_target for sample in ordered])
            wdls = np.stack([sample.wdl_target for sample in ordered])
            masks = np.packbits(
                np.stack([sample.legal_mask for sample in ordered]), axis=-1, bitorder="little"
            )
        else:
            states = np.empty((0, 3, 15, 15), dtype=np.float32)
            policies = np.empty((0, POLICY_SIZE), dtype=np.float32)
            wdls = np.empty((0, WDL_SIZE), dtype=np.float32)
            masks = np.empty((0, (POLICY_SIZE + 7) // 8), dtype=np.uint8)
        return {
            "version": 1,
            "capacity": self.capacity,
            "states": states,
            "policies": policies,
            "wdls": wdls,
            "packed_masks": masks,
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported replay-buffer state version")
        states = np.asarray(state["states"], dtype=np.float32)
        policies = np.asarray(state["policies"], dtype=np.float32)
        wdls = np.asarray(state["wdls"], dtype=np.float32)
        masks = np.unpackbits(
            np.asarray(state["packed_masks"], dtype=np.uint8), axis=-1, bitorder="little"
        )[..., :POLICY_SIZE].astype(np.bool_)
        count = states.shape[0]
        if not (policies.shape[0] == wdls.shape[0] == masks.shape[0] == count):
            raise ValueError("replay-buffer arrays have inconsistent lengths")
        if count > self.capacity:
            start = count - self.capacity
            states, policies, wdls, masks = (
                states[start:],
                policies[start:],
                wdls[start:],
                masks[start:],
            )
        self._samples = [
            TrainingSample(states[i], policies[i], wdls[i], masks[i])
            for i in range(states.shape[0])
        ]
        self._next = len(self._samples) % self.capacity
        if "rng_state" in state:
            self._rng.bit_generator.state = state["rng_state"]

    def save(self, path: str | Path) -> None:
        """Atomically save a compressed replay artifact."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        state = self.state_dict()
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                version=np.asarray(state["version"], dtype=np.int64),
                capacity=np.asarray(state["capacity"], dtype=np.int64),
                states=state["states"],
                policies=state["policies"],
                wdls=state["wdls"],
                packed_masks=state["packed_masks"],
                rng_state_json=np.asarray(json.dumps(state["rng_state"], sort_keys=True)),
            )
        temporary.replace(target)

    def load(self, path: str | Path) -> None:
        with np.load(Path(path), allow_pickle=False) as data:
            self.load_state_dict(
                {
                    "version": int(data["version"]),
                    "capacity": int(data["capacity"]),
                    "states": data["states"],
                    "policies": data["policies"],
                    "wdls": data["wdls"],
                    "packed_masks": data["packed_masks"],
                    "rng_state": json.loads(str(data["rng_state_json"].item())),
                }
            )


__all__ = [
    "POLICY_SIZE",
    "ReplayBuffer",
    "TrainingSample",
    "all_symmetries",
    "normalize_policy_target",
    "normalize_wdl_target",
    "transform_sample",
]
