from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gomoku_zero.game import BLACK, BOARD_CELLS, Board  # noqa: E402
from gomoku_zero.model import (  # noqa: E402
    DeterministicEvaluator,
    GomokuNet,
    load_checkpoint,
    masked_softmax,
    save_checkpoint,
)


def test_occupied_high_logit_is_zero_and_legal_mass_is_one() -> None:
    occupied = 112
    board = Board().play(occupied)
    logits = torch.zeros(BOARD_CELLS, dtype=torch.float64)
    logits[occupied] = 1.0e9
    logits[113] = 3.0
    mask = torch.from_numpy(board.legal_mask)

    probabilities = masked_softmax(logits, mask)

    assert probabilities[occupied].item() == 0.0
    assert probabilities[113] > probabilities[114]
    assert probabilities[mask].sum().item() == pytest.approx(1.0)
    assert torch.count_nonzero(probabilities[~mask]).item() == 0


def test_illegal_logits_have_exactly_zero_gradient() -> None:
    board = Board().play(0).play(1).play(2)
    logits = torch.linspace(-2.0, 2.0, BOARD_CELLS, dtype=torch.float64)
    logits.requires_grad_()
    mask = torch.from_numpy(board.legal_mask)

    probabilities = masked_softmax(logits, mask)
    loss = -torch.log(probabilities[10])
    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[~mask]).item() == 0
    assert torch.isfinite(logits.grad[mask]).all()
    assert logits.grad[mask].sum().item() == pytest.approx(0.0, abs=1e-12)


def test_masking_does_not_use_a_finite_negative_sentinel() -> None:
    logits = torch.tensor([-2.0e9, -3.0e9, 9.0e9], dtype=torch.float64)
    mask = torch.tensor([True, True, False])

    probabilities = masked_softmax(logits, mask)

    assert probabilities.tolist() == [1.0, 0.0, 0.0]


def test_illegal_nan_and_infinity_are_removed_before_softmax() -> None:
    logits = torch.tensor([0.0, 1.0, float("nan"), float("inf")], dtype=torch.float64)
    mask = torch.tensor([True, True, False, False])

    probabilities = masked_softmax(logits, mask)

    expected = torch.softmax(torch.tensor([0.0, 1.0], dtype=torch.float64), dim=0)
    torch.testing.assert_close(probabilities[:2], expected)
    assert probabilities[2:].tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("logits", "mask", "error"),
    [
        (torch.zeros(3), torch.ones(2, dtype=torch.bool), ValueError),
        (torch.zeros(3), torch.zeros(3, dtype=torch.bool), ValueError),
        (torch.zeros(3), torch.ones(3), TypeError),
        (
            torch.tensor([0.0, float("nan"), 1.0]),
            torch.ones(3, dtype=torch.bool),
            ValueError,
        ),
    ],
)
def test_masked_softmax_fails_closed(logits: object, mask: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        masked_softmax(logits, mask)  # type: ignore[arg-type]


def test_batched_masks_are_not_broadcast_and_normalize_per_sample() -> None:
    logits = torch.tensor([[1.0, 2.0, 9.0], [8.0, 3.0, 4.0]], requires_grad=True)
    mask = torch.tensor([[True, True, False], [False, True, True]])
    probabilities = masked_softmax(logits, mask, dim=1)

    assert probabilities[0, 2].item() == 0.0
    assert probabilities[1, 0].item() == 0.0
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2))

    with pytest.raises(ValueError, match="exactly match"):
        masked_softmax(logits, mask[0], dim=1)


def test_network_outputs_raw_policy_and_absolute_wdl_logits() -> None:
    model = GomokuNet(channels=8, residual_blocks=1)
    states = torch.zeros((2, 3, 15, 15), dtype=torch.float32)

    policy_logits, wdl_logits = model(states)

    assert policy_logits.shape == (2, BOARD_CELLS)
    assert wdl_logits.shape == (2, 3)
    (policy_logits.mean() + wdl_logits.mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((2, 3, 14, 15)))


def test_checkpoint_round_trip_and_mcts_evaluator_contract(tmp_path: object) -> None:
    torch.manual_seed(9)
    model = GomokuNet(channels=4, residual_blocks=0).eval()
    board = Board().play(112)
    state = board.to_tensor().unsqueeze(0)
    with torch.inference_mode():
        expected = model(state)

    path = save_checkpoint(tmp_path / "model.pt", model, step=7, metadata={"kind": "test"})
    loaded, payload = load_checkpoint(path)
    loaded.eval()
    with torch.inference_mode():
        actual = loaded(state)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
    assert payload["step"] == 7

    evaluator = DeterministicEvaluator(loaded)
    policy, absolute_wdl = evaluator(board)
    assert policy.shape == (BOARD_CELLS,)
    assert absolute_wdl.shape == (3,)
    assert policy[112] == 0.0
    assert policy.sum() == pytest.approx(1.0)
    assert absolute_wdl.sum() == pytest.approx(1.0)


def test_terminal_evaluator_returns_exact_absolute_wdl() -> None:
    cells = np.zeros((15, 15), dtype=np.int8)
    cells[4, 2:7] = BLACK
    terminal = Board.from_array(cells)
    evaluator = DeterministicEvaluator(GomokuNet(channels=4, residual_blocks=0))

    policy, wdl = evaluator(terminal)

    np.testing.assert_array_equal(policy, np.zeros(BOARD_CELLS, dtype=np.float32))
    np.testing.assert_array_equal(wdl, np.array([1.0, 0.0, 0.0], dtype=np.float32))

