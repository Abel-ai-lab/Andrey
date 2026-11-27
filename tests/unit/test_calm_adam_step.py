"""CALM's inline Adam step returns the bits of ``torch.optim.Adam``.

``_adam_step`` applies the update that ``torch.optim.Adam`` gives a CPU tensor, with fresh moment
buffers each round. It is pinned to the optimizer on gradient streams of widely different scales
over several thousand steps and rounds, in the parameters and in the moments, and
``_optimize_gated_dag`` and the ``calm`` facade are pinned to the same functions run with the
optimizer in its place.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey
from andrey.search import calm as calm_module

torch = pytest.importorskip("torch")


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.detach().numpy().tobytes() == b.detach().numpy().tobytes()


@pytest.mark.parametrize(
    ("lr", "steps", "shape", "rounds"),
    [
        (1e-2, 3000, (10, 10), 2),
        (1e-3, 3000, (7, 7), 2),
        (0.5, 500, (3, 5), 3),
        (1.0, 300, (4, 4), 1),
        (3e-4, 1000, (30, 30), 1),
    ],
)
def test_adam_step_gives_the_bits_of_torch_adam(lr, steps, shape, rounds) -> None:
    """After every step of every round, the parameters and both moments equal the optimizer's."""
    gen = torch.Generator().manual_seed(steps + rounds)
    start = [torch.randn(*shape, generator=gen, dtype=torch.double) for _ in range(2)]
    scales = 10.0 ** torch.randint(-12, 4, (steps, 2), generator=gen).double()
    grads = [
        [torch.randn(*shape, generator=gen, dtype=torch.double) * scales[k, i] for i in range(2)]
        for k in range(steps)
    ]
    reference = [p.clone().requires_grad_(True) for p in start]
    inline = [p.clone().requires_grad_(True) for p in start]
    for _ in range(rounds):
        optimizer = torch.optim.Adam(reference, lr=lr)
        moments = [(torch.zeros_like(p), torch.zeros_like(p)) for p in inline]
        for k, grad in enumerate(grads):
            for p, q, g in zip(reference, inline, grad, strict=True):
                p.grad, q.grad = g.clone(), g.clone()
            optimizer.step()
            with torch.no_grad():
                calm_module._adam_step(tuple(inline), moments, k + 1, lr)
            for p, q in zip(reference, inline, strict=True):
                assert _same_bits(p, q), f"parameters differ at step {k + 1}"
        for p, (exp_avg, exp_avg_sq) in zip(reference, moments, strict=True):
            assert _same_bits(optimizer.state[p]["exp_avg"], exp_avg)
            assert _same_bits(optimizer.state[p]["exp_avg_sq"], exp_avg_sq)


class _TorchAdamRound:
    """Stands in for ``_adam_step``: ``torch.optim.Adam``, built anew at each round's first step."""

    def __init__(self) -> None:
        self.optimizer: torch.optim.Adam | None = None

    def __call__(self, params, moments, step, lr) -> None:
        if step == 1:
            self.optimizer = torch.optim.Adam(list(params), lr=lr)
        assert self.optimizer is not None
        self.optimizer.step()


def _random_sem_covariance(d: int, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    order = rng.permutation(d)
    weights = np.zeros((d, d))
    for a in range(d):
        for b in range(a + 1, d):
            if rng.random() < 2.0 / d:
                weights[order[a], order[b]] = rng.uniform(0.5, 2.0) * rng.choice([-1.0, 1.0])
    x = rng.normal(size=(n, d)) @ np.linalg.inv(np.eye(d) - weights)
    x -= x.mean(axis=0)
    return x.T @ x / n


_FIT = {
    "lambda1": 0.005,
    "tau": 0.5,
    "rho_init": 1e-3,
    "rho_mult": 3.0,
    "rho_max": 1e16,
    "h_tol": 1e-8,
    "max_outer": 3,
    "subproblem_iter": 400,
    "lr": 1e-2,
}


def _fit_both_ways(monkeypatch, cov, mask, **options):
    """``_optimize_gated_dag`` with the inline step, then with ``torch.optim.Adam`` in its place."""
    args = {**_FIT, "seed": 0, **options}
    inline = calm_module._optimize_gated_dag(cov, mask, **args)
    with monkeypatch.context() as patch:
        patch.setattr(calm_module, "_adam_step", _TorchAdamRound())
        reference = calm_module._optimize_gated_dag(cov, mask, **args)
    return inline, reference


def _assert_same_fit(inline, reference) -> None:
    for got, want in zip(inline[:2], reference[:2], strict=True):
        assert got.tobytes() == want.tobytes()
    assert inline[2] == reference[2]


@pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_optimizer_matches_torch_adam_on_a_full_mask(monkeypatch, seed) -> None:
    cov = _random_sem_covariance(8, 300, seed)
    _assert_same_fit(*_fit_both_ways(monkeypatch, cov, 1.0 - np.eye(8), seed=seed))


@pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning")
def test_optimizer_matches_torch_adam_with_required_edges_and_a_sparse_mask(monkeypatch) -> None:
    cov = _random_sem_covariance(7, 300, 3)
    mask = np.zeros((7, 7))
    mask[np.triu_indices(7, 1)] = 1.0
    mask += mask.T
    required = np.zeros((7, 7), dtype=bool)
    required[0, 1] = required[2, 3] = True
    mask[0, 1] = mask[2, 3] = 1.0
    _assert_same_fit(*_fit_both_ways(monkeypatch, cov, mask, required=required))


@pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning")
def test_optimizer_matches_torch_adam_with_a_small_step_and_penalty(monkeypatch) -> None:
    """The settings of a long, slow schedule: a small learning rate and a small first penalty."""
    cov = _random_sem_covariance(6, 300, 4)
    _assert_same_fit(
        *_fit_both_ways(
            monkeypatch, cov, 1.0 - np.eye(6), lr=1e-3, rho_init=1e-5, subproblem_iter=800
        )
    )


@pytest.mark.filterwarnings("ignore::andrey.SearchLimitWarning")
@pytest.mark.parametrize("options", [{}, {"standardize": True}, {"use_moral_graph": False}])
def test_calm_matches_torch_adam(monkeypatch, options) -> None:
    """The facade returns the same weights and metadata with the optimizer in its place."""
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 5))
    x[:, 1] += x[:, 0]
    x[:, 3] += x[:, 1] - x[:, 2]
    fit = {"seed": 0, "subproblem_iter": 300, "max_iter": 3, **options}
    inline = andrey.calm(x, **fit)
    with monkeypatch.context() as patch:
        patch.setattr(calm_module, "_adam_step", _TorchAdamRound())
        reference = andrey.calm(x, **fit)
    assert inline.weighted_adjacency.tobytes() == reference.weighted_adjacency.tobytes()
    assert inline.metadata["h_final"] == reference.metadata["h_final"]
    assert inline.metadata["objective"] == reference.metadata["objective"]
