"""Datasets of the benchmark: merged cohorts, hand cases, and lagged cohorts.

Every dataset is a pure function of its cell name and seed. Each seed draws its own random streams
for the graph, the data, the units, the bridging rows, the scattered cells, and the row order
(``default_rng([seed, tag])``), so no two streams share a state.

A dataset is a dict:

- ``complete``: the ``(n, d)`` values before masking (the complete-data arm),
- ``data``: the same values with ``NaN`` where a cell is missing,
- ``design``: the structural cohort patterns before bridging and scatter, one row per cohort (the
  lagged design: one row per time point),
- ``dag``: ``(d, d)`` 0/1 with ``dag[i, j] = 1`` for ``i -> j``,
- ``names``: the column names.

The lagged design returns ``links``, a ``(TAU_MAX + 1, d, d)`` 0/1 array with
``links[k, i, j] = 1`` for ``X_i(t - k) -> X_j(t)``, in place of ``dag``.
"""

from __future__ import annotations

import numpy as np

#: Stream tags of ``default_rng([seed, tag])``.
SEM, DATA, UNITS, BRIDGE, SCATTER, PERM = 1, 2, 3, 4, 5, 6

#: Each configuration: the rows of each cohort, the columns every cohort measures (``core``), the
#: modules (size, cohorts that measure it), the columns one cohort alone measures, and the
#: expected degree of the graph.
CONFIGS = {
    "S3": {
        "sizes": (2000, 1700, 1300),
        "core": 8,
        "modules": ((4, (0, 1)), (4, (2,))),
        "only": (1, 1, 2),
        "degree": 3.0,
    },
    "M8": {
        "sizes": (4000, 3500, 3000, 2500, 2500, 2000, 1500, 1000),
        "core": 12,
        "modules": ((6, (0, 1, 2)), (6, (2, 3, 4)), (6, (4, 5, 6)), (6, (6, 7, 0))),
        "only": (1, 1, 0, 1, 0, 1, 0, 0),
        "degree": 3.0,
    },
    "L6": {
        "sizes": (8000, 6000, 5000, 4500, 3500, 3000),
        "core": 15,
        "modules": ((7, (0, 1, 2)), (7, (2, 3, 4)), (7, (0, 4, 5)), (6, (1, 3, 5))),
        "only": (2, 2, 1, 1, 1, 1),
        "degree": 3.0,
    },
}

#: Cohort variants as ``(bridge share, scatter share)``: the share of rows that observe every
#: column, and the share of the other cells missing completely at random. ``complete`` is the data
#: before masking.
VARIANTS = {
    "b0s0": (0.0, 0.0),
    "b0s10": (0.0, 0.10),
    "b2s0": (0.02, 0.0),
    "b2s10": (0.02, 0.10),
    "b5s0": (0.05, 0.0),
    "b5s10": (0.05, 0.10),
}


def rng(seed: int, tag: int) -> np.random.Generator:
    """The stream of ``seed`` with the stream tag ``tag``."""
    return np.random.default_rng([seed, tag])


def dims(config: str) -> int:
    """The number of columns of a configuration."""
    c = CONFIGS[config]
    return c["core"] + sum(m for m, _ in c["modules"]) + sum(c["only"])


def random_sem(d: int, degree: float, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A random DAG of expected degree ``degree``: its causal order, weights, and noise scales."""
    g = rng(seed, SEM)
    order = g.permutation(d)
    p = degree / (d - 1)
    w = np.zeros((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            if g.random() < p:
                w[order[i], order[j]] = g.uniform(0.3, 1.0) * g.choice([-1, 1])
    scale = g.uniform(0.5, 1.5, d)
    return order, w, scale


def simulate(order: np.ndarray, w: np.ndarray, scale: np.ndarray, n: int, g) -> np.ndarray:
    """``n`` rows of the linear-Gaussian model with weights ``w``, in the causal order."""
    d = w.shape[0]
    x = np.zeros((n, d))
    for j in order:
        x[:, j] = x @ w[:, j] + scale[j] * g.standard_normal(n)
    return x


def cohort_patterns(config: str) -> np.ndarray:
    """``(K, d)`` boolean: the columns each cohort measures."""
    c = CONFIGS[config]
    d = dims(config)
    out = []
    for k in range(len(c["sizes"])):
        m = np.zeros(d, dtype=bool)
        m[: c["core"]] = True
        start = c["core"]
        for size, cohorts in c["modules"]:
            if k in cohorts:
                m[start : start + size] = True
            start += size
        start += sum(c["only"][:k])
        m[start : start + c["only"][k]] = True
        out.append(m)
    return np.array(out)


def cohort(config: str, variant: str, seed: int) -> dict:
    """One merged-cohort dataset; the variant ``"complete"`` has no missing values."""
    c = CONFIGS[config]
    d = dims(config)
    sizes = c["sizes"]
    n = sum(sizes)
    order, w, scale = random_sem(d, c["degree"], seed)
    x = simulate(order, w, scale, n, rng(seed, DATA))
    u = rng(seed, UNITS)
    spread = u.uniform(0.1, 100.0, d)
    x = x * spread + spread * u.uniform(5.0, 50.0, d)  # lab-like units: mean 5-50 spreads
    pats = cohort_patterns(config)
    label = np.repeat(np.arange(len(sizes)), sizes)
    observed = pats[label].copy()
    bridge, scatter = VARIANTS.get(variant, (0.0, 0.0))
    if bridge:
        b = rng(seed, BRIDGE)
        start = 0
        for n_k in sizes:  # in proportion to each cohort's size, so n is unchanged
            take = b.choice(n_k, int(round(bridge * n_k)), replace=False)
            observed[start + take] = True
            start += n_k
    if scatter:
        observed &= rng(seed, SCATTER).random(observed.shape) >= scatter
    perm = rng(seed, PERM).permutation(n)  # merged files mix the cohorts
    x, observed = x[perm], observed[perm]
    return {
        "complete": x,
        "data": x.copy() if variant == "complete" else np.where(observed, x, np.nan),
        "design": pats,
        "dag": (w != 0).astype(np.int8),
        "names": [f"V{j}" for j in range(d)],
    }


#: Hand cases of three columns: the edges as ``(parent, child)`` over 0 = X, 1 = Y, 2 = the third
#: column, the cohort patterns, and the cohorts' shares of the rows.
HAND = {
    "H1": ([(2, 0), (2, 1)], [(1, 0, 1), (0, 1, 1)], (0.5, 0.5)),  # X <- Z -> Y
    "H1c": ([(0, 2), (1, 2)], [(1, 0, 1), (0, 1, 1)], (0.5, 0.5)),  # X -> Z <- Y
    "H4": ([(2, 0), (0, 1)], [(1, 0, 1), (0, 1, 1)], (0.5, 0.5)),  # Z -> X -> Y
}
#: ``H5k<k>`` is H1 plus ``k`` rows that observe all three columns.
H5_K = (3,)


def hand(case: str, n: int, seed: int) -> dict:
    """A hand case of ``n`` rows: weights 0.8, root noise 1, and child noise 0.6."""
    extra = 0
    if case.startswith("H5"):
        extra = int(case[3:])
        edges, pats, shares = HAND["H1"]
    else:
        edges, pats, shares = HAND[case]
    dag = np.zeros((3, 3), dtype=np.int8)
    for a, b in edges:
        dag[a, b] = 1
    total = n + extra
    g = rng(seed, DATA)
    x = np.zeros((total, 3))
    for j in _topological(dag):
        parents = np.flatnonzero(dag[:, j])
        noise = 0.6 if parents.size else 1.0
        x[:, j] = x[:, parents].sum(axis=1) * 0.8 + noise * g.standard_normal(total)
    pats = np.array(pats, dtype=bool)
    counts = _split(n, shares)
    rows = [np.tile(p, (k, 1)) for p, k in zip(pats, counts, strict=True)]
    rows.append(np.ones((extra, 3), dtype=bool))
    observed = np.vstack(rows)
    perm = rng(seed, PERM).permutation(total)
    x, observed = x[perm], observed[perm]
    return {
        "complete": x,
        "data": np.where(observed, x, np.nan),
        "design": pats,
        "dag": dag,
        "names": ["X", "Y", "Z"],
    }


def fixed_fork() -> dict:
    """The fork X <- Z -> Y of 1,000 rows drawn in one stream (Z, then X, then Y) from
    ``default_rng(0)``, with Y missing in the first 500 rows and X in the last 500."""
    g = np.random.default_rng(0)
    n = 1000
    z = g.normal(size=n)
    x = 0.8 * z + g.normal(scale=0.6, size=n)
    y = 0.8 * z + g.normal(scale=0.6, size=n)
    data = np.column_stack([x, y, z])
    data[: n // 2, 1] = np.nan
    data[n // 2 :, 0] = np.nan
    dag = np.zeros((3, 3), dtype=np.int8)
    dag[2, 0] = dag[2, 1] = 1
    return {
        "complete": None,
        "data": data,
        "design": np.array([[1, 0, 1], [0, 1, 1]], dtype=bool),
        "dag": dag,
        "names": ["X", "Y", "Z"],
    }


def _split(n: int, shares) -> list[int]:
    counts = [int(round(s * n)) for s in shares]
    counts[-1] = n - sum(counts[:-1])
    return counts


def _topological(dag: np.ndarray) -> list[int]:
    d = dag.shape[0]
    done: list[int] = []
    while len(done) < d:
        for j in range(d):
            if j not in done and all(i in done for i in np.flatnonzero(dag[:, j])):
                done.append(j)
    return done


#: The lagged design: 10 variables, 6,000 time points in three periods of 2,000. The core (4
#: variables) is observed in every period, the first module (3) in periods 0 and 1, and the second
#: (3) in period 2, so the two modules are never observed at the same time.
LAG_D, LAG_N, LAG_PERIOD = 10, 6000, 2000
LAG_CORE, LAG_M1, LAG_M2 = range(0, 4), range(4, 7), range(7, 10)
LAG_VARIANTS = ("nolap0", "nolap5", "complete")
TAU_MAX = 2


def lagged(variant: str, seed: int) -> dict:
    """A stable VAR(1) over ``LAG_D`` variables; ``nolap5`` also loses 5% of the observed cells."""
    d, n = LAG_D, LAG_N
    g = rng(seed, SEM)
    a = np.where(g.random((d, d)) < 2.0 / d, g.uniform(0.2, 0.4, (d, d)), 0.0)
    a *= g.choice([-1.0, 1.0], (d, d))
    np.fill_diagonal(a, g.uniform(0.3, 0.6, d))
    a *= 0.8 / max(abs(np.linalg.eigvals(a)))  # a stable process
    e = rng(seed, DATA).standard_normal((n + 200, d))
    x = np.zeros((n + 200, d))
    for t in range(1, n + 200):
        x[t] = x[t - 1] @ a + e[t]
    x = x[200:]  # burn-in
    u = rng(seed, UNITS)
    x = x * u.uniform(1, 50, d) + u.uniform(100, 1000, d)
    design = np.zeros((n, d), dtype=bool)
    design[:, list(LAG_CORE)] = True
    design[: 2 * LAG_PERIOD, list(LAG_M1)] = True
    design[2 * LAG_PERIOD :, list(LAG_M2)] = True
    observed = design.copy()
    if variant == "nolap5":
        observed &= rng(seed, SCATTER).random(observed.shape) >= 0.05
    links = np.zeros((TAU_MAX + 1, d, d), dtype=np.int8)
    links[1] = a != 0
    return {
        "complete": x,
        "data": x.copy() if variant == "complete" else np.where(observed, x, np.nan),
        "design": design,
        "links": links,
        "names": [str(j) for j in range(d)],
    }


def make(design: str, cell: str, seed: int) -> dict:
    """``design`` is ``hand``, ``cohort``, or ``lagged``; ``cell`` names its case or configuration
    and variant: ``H1_n1000``, ``H1_fixed``, ``S3_b0s10``, ``nolap5``."""
    if design == "hand":
        if cell == "H1_fixed":
            return fixed_fork()
        case, n = cell.split("_n")
        return hand(case, int(n), seed)
    if design == "cohort":
        config, variant = cell.split("_")
        return cohort(config, variant, seed)
    if design == "lagged":
        return lagged(cell, seed)
    raise ValueError(design)
