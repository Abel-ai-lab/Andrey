"""The simulated data of the local ISA-LiNG paper's experiments, and the scorer of its local graphs.

Dai, Ng, Zheng, Gao, and Zhang (2024), AISTATS, Sect. 5, simulate linear non-Gaussian models
``X = B X + E`` on 50 variables: the nonzero weights of ``B`` uniform in magnitude on [0.5, 0.9]
with a random sign, and each noise ``E_i`` uniform on ``[-c_i, c_i]`` raised to the fifth power,
with ``c_i`` uniform on [0.75, 1.25].

- ``cyclic``: a directed graph with cycles in which every variable has at most 4 neighbors. Each
  variable draws its own cap on its degree, uniform on 1 to 4; every variable in turn is joined to
  the others in a random order, each edge in a random direction (the reverse of an edge already
  drawn makes a 2-cycle), while both ends are below their caps, and an edge that would close a strongly connected component of more than 10 variables is
  dropped. A graph is kept when it has a cycle and its cycles are disjoint (every strongly
  connected component is one simple cycle), and its weights are drawn again until the spectral
  radius of ``B`` is below 1, as the paper states for its stable cyclic experiments. The target is
  drawn among the variables on a cycle.
- ``acyclic``: an Erdos-Renyi DAG with ``degree * d / 2`` edges, so expected degree ``degree``,
  oriented along a random order. The paper takes targets with "roughly 14 and 20 variables in the
  MB" at degrees 3 and 5; the target here is the variable whose true blanket size is closest to
  :data:`BLANKET_GOAL` of the degree, ties broken by a draw.

The columns are then listed in an order drawn from the same seed, and the truth permuted with them,
so the column order follows neither the graph nor the target. A draw keeps its seed's stream:
the same ``(setting, degree, seed)`` gives the same model for every ``n``, and the rows of a larger
``n`` are a fresh sample, not an extension.

The scorer reads the paper's local structure (its App. E.3, the authors' ``get_local_dcg`` with
the children's parents): the edges into and out of the target and the edges into each child of
the target, each graph's own children. :func:`local_scores` counts its structural Hamming distance
as the authors' ``count_dcg_accuracy`` does (extra and missing adjacencies plus edges whose reverse
is true) and the Euclidean distance between the two local weight matrices. Weights are
``w[i, j]`` for the edge ``i -> j``.

The module imports NumPy only, so a reference's own environment can import it.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

#: Weight magnitudes and the noise scale range of the paper.
WEIGHT_RANGE = (0.5, 0.9)
NOISE_RANGE = (0.75, 1.25)

#: The cyclic graphs: variables, the largest degree, and the largest strongly connected component.
CYCLIC = {"d": 50, "max_degree": 4, "max_cycle": 10}

#: The blanket size the acyclic target is chosen near, per expected degree (the paper's 14 and 20).
BLANKET_GOAL = {3: 14, 5: 20}

#: Draws of a cyclic graph or of stable weights before giving up.
MAX_DRAWS = 10_000


@dataclass(frozen=True)
class Draw:
    """One dataset: ``data`` (n, d) with shuffled columns, ``weights[i, j]`` for ``i -> j`` in the
    same columns, the target's column, and its true blanket (sorted columns)."""

    data: np.ndarray
    weights: np.ndarray
    target: int
    blanket: tuple[int, ...]


def _rng(setting: str, degree: int, seed: int) -> np.random.Generator:
    return np.random.default_rng([zlib.crc32(setting.encode()), degree, seed])


def _components(adj: np.ndarray) -> list[list[int]]:
    """Strongly connected components of ``adj[i, j] != 0`` for ``i -> j``."""
    d = len(adj)
    reach = (adj != 0) | np.eye(d, dtype=bool)
    for k in range(d):
        reach |= reach[:, [k]] & reach[[k], :]
    both = reach & reach.T
    seen, out = set(), []
    for i in range(d):
        if i not in seen:
            members = [int(j) for j in np.flatnonzero(both[i])]
            seen.update(members)
            out.append(members)
    return out


def _cyclic_support(
    rng: np.random.Generator, d: int, max_degree: int, max_cycle: int
) -> np.ndarray:
    adj = np.zeros((d, d), dtype=bool)
    caps = rng.integers(1, max_degree + 1, size=d)

    def degree(v: int) -> int:
        return int(adj[v].sum() + adj[:, v].sum())

    def component(v: int) -> int:
        reach = adj.copy() | np.eye(d, dtype=bool)
        for k in range(d):
            reach |= reach[:, [k]] & reach[[k], :]
        return int((reach[v] & reach[:, v]).sum())

    for i in range(d):
        for j in rng.permutation(d):
            if j == i:
                continue
            if degree(i) >= caps[i]:
                break
            forward = bool(rng.integers(2))
            a, b = (i, j) if forward else (j, i)
            if degree(int(j)) < caps[j] and not adj[a, b]:
                adj[a, b] = True
                if component(int(j)) > max_cycle:
                    adj[a, b] = False
    return adj


def _cycles(adj: np.ndarray) -> list[list[int]] | None:
    """The cycles of ``adj`` when every strongly connected component is one simple cycle, else
    ``None``."""
    cycles = [c for c in _components(adj) if len(c) > 1]
    for c in cycles:
        if int(adj[np.ix_(c, c)].sum()) != len(c):
            return None
    return cycles


def _weights(rng: np.random.Generator, support: np.ndarray, stable: bool) -> np.ndarray:
    lo, hi = WEIGHT_RANGE
    for _ in range(MAX_DRAWS):
        w = support * rng.uniform(lo, hi, size=support.shape)
        if not stable or np.max(np.abs(np.linalg.eigvals(w))) < 1:
            return w * rng.choice([-1.0, 1.0], size=support.shape)
    raise RuntimeError("no stable weights in MAX_DRAWS draws")


def _sample(rng: np.random.Generator, w: np.ndarray, n: int) -> np.ndarray:
    d = len(w)
    c = rng.uniform(*NOISE_RANGE, size=d)
    noise = rng.uniform(-c, c, size=(n, d)) ** 5
    # Rows are samples: x = x w + e, so x = e (I - w)^-1 with w[i, j] for i -> j.
    return noise @ np.linalg.inv(np.eye(d) - w)


def true_blanket(weights: np.ndarray, target: int) -> tuple[int, ...]:
    """Parents, children, and the children's other parents of ``target`` (cycles allowed)."""
    adj = weights != 0
    parents = set(np.flatnonzero(adj[:, target]))
    children = set(np.flatnonzero(adj[target]))
    spouses = {int(p) for c in children for p in np.flatnonzero(adj[:, c])}
    return tuple(sorted(int(v) for v in (parents | children | spouses) - {target}))


def sample(setting: str, n: int, seed: int, degree: int = 3) -> Draw:
    """The dataset of ``(setting, degree, seed)`` at ``n`` rows (``degree`` is ignored when cyclic)."""
    key = 4 if setting == "cyclic" else degree
    rng = _rng(setting, key, seed)
    if setting == "cyclic":
        for _ in range(MAX_DRAWS):
            support = _cyclic_support(rng, **{k: CYCLIC[k] for k in CYCLIC})
            cycles = _cycles(support)
            if cycles:
                break
        else:
            raise RuntimeError("no graph with disjoint cycles in MAX_DRAWS draws")
        w = _weights(rng, support, stable=True)
        pool = sorted(v for c in cycles for v in c)
    elif setting == "acyclic":
        d = CYCLIC["d"]
        pairs = np.array([(i, j) for i in range(d) for j in range(i + 1, d)])
        chosen = pairs[rng.choice(len(pairs), size=degree * d // 2, replace=False)]
        order = rng.permutation(d)
        support = np.zeros((d, d), dtype=bool)
        for i, j in chosen:
            a, b = (i, j) if order[i] < order[j] else (j, i)
            support[a, b] = True
        w = _weights(rng, support, stable=False)
        sizes = np.array([len(true_blanket(w, t)) for t in range(d)])
        gap = np.abs(sizes - BLANKET_GOAL[degree])
        pool = [int(t) for t in np.flatnonzero(gap == gap.min())]
    else:
        raise ValueError(f"unknown setting {setting!r}")
    target = int(rng.choice(pool))
    columns = rng.permutation(len(w))  # columns[k] = the variable listed in column k
    data_rng = np.random.default_rng([zlib.crc32(setting.encode()), key, seed, n])
    x = _sample(data_rng, w, n)[:, columns]
    w = w[np.ix_(columns, columns)]
    t = int(np.flatnonzero(columns == target)[0])
    return Draw(data=x, weights=w, target=t, blanket=true_blanket(w, t))


def local_structure(weights: np.ndarray, target: int) -> np.ndarray:
    """The edges into and out of ``target`` and into each of its children in ``weights``."""
    w = np.asarray(weights, dtype=np.float64)
    out = np.zeros_like(w)
    out[:, target], out[target] = w[:, target], w[target]
    for child in np.flatnonzero(w[target]):
        out[:, child] = w[:, child]
    return out


def local_scores(estimated: np.ndarray, true: np.ndarray, target: int) -> dict[str, float]:
    """SHD and weight distance of ``estimated``'s local structure at ``target`` against ``true``'s.

    ``shd`` counts the adjacencies one local structure has and the other lacks, plus the
    estimated edges absent from the truth whose reverse it holds (the authors'
    ``count_dcg_accuracy``); ``weight_dist`` is the Euclidean norm of the difference of the two
    local weight matrices; ``true_edges`` and ``est_edges`` count the directed edges.
    """
    est, tru = local_structure(estimated, target), local_structure(true, target)
    e, t = est != 0, tru != 0
    e_skel, t_skel = np.triu(e | e.T, 1), np.triu(t | t.T, 1)
    reversed_ = e & ~t & t.T
    shd = int((e_skel & ~t_skel).sum() + (t_skel & ~e_skel).sum() + reversed_.sum())
    return {
        "shd": shd,
        "weight_dist": float(np.linalg.norm(est - tru)),
        "true_edges": int(t.sum()),
        "est_edges": int(e.sum()),
        "correct_edges": int((e & t).sum()),
    }


#: Answer codes of :func:`pack_answer`: a fit that answered, and one that declined to run (the
#: authors' Algorithm 1 past its local-size limit), which is never scored as an empty answer.
ANSWERED, NOT_RUN = 0, 1


def pack_answer(
    weights: np.ndarray,
    blanket: list[int] | tuple[int, ...] | None,
    *,
    code: int = ANSWERED,
    m: int = 0,
    models: int = 0,
    fallback: bool = False,
) -> np.ndarray:
    """One array for the runner: ``weights`` (d, d), then a row marking the blanket's columns, then
    ``[code, m, models, fallback]`` (``m`` the local variables, ``models`` the equivalent models,
    ``fallback`` an acyclic fit that ran Algorithm 1); needs ``d >= 4``."""
    d = len(weights)
    out = np.zeros((d + 2, d))
    out[:d] = weights
    if blanket is not None:
        out[d, list(blanket)] = 1.0
    out[d + 1, :4] = [code, m, models, float(fallback)]
    return out


def unpack_answer(packed: np.ndarray) -> dict:
    """The parts of :func:`pack_answer`'s array."""
    a = np.asarray(packed, dtype=np.float64)
    d = a.shape[1]
    code, m, models, fallback = a[d + 1, :4]
    return {
        "weights": a[:d],
        "blanket": [int(j) for j in np.flatnonzero(a[d])],
        "code": int(code),
        "m": int(m),
        "models": int(models),
        "fallback": bool(fallback),
    }
