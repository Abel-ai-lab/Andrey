"""Pair classes on the structural cohort patterns, and the check of the rule that defines them.

The classes are fixed on the patterns before bridging rows and scattered cells, so the variants of
a configuration compare the same pairs:

- class i, together: some cohort pattern observes both columns;
- class ii, apart: no cohort pattern observes both;
- class iii: a class-i pair that is not adjacent in the DAG and that no cohort holding both can
  separate: for every pattern O_k holding X and Y, ``(An({X, Y}) & O_k) - {X, Y}`` does not
  d-separate them (the observed ancestors of a pair suffice when any separating subset exists).

``e_adj`` are the class-i true edges with an endpoint in some class-ii pair: the edges next to a
pair that is never observed together.
"""

from __future__ import annotations

import itertools

import numpy as np


def ancestors(dag: np.ndarray, nodes) -> set[int]:
    """The nodes with a directed path into ``nodes``, ``nodes`` included."""
    seen = set(nodes)
    stack = list(nodes)
    while stack:
        j = stack.pop()
        for i in np.flatnonzero(dag[:, j]).tolist():
            if i not in seen:
                seen.add(i)
                stack.append(i)
    return seen


def d_separated(dag: np.ndarray, x: int, y: int, z) -> bool:
    """Whether ``x`` and ``y`` are d-separated given ``z``: moralize the ancestral set, cut ``z``."""
    z = set(z)
    keep = ancestors(dag, {x, y} | z)
    adj: dict[int, set[int]] = {v: set() for v in keep}
    for j in keep:
        parents = [i for i in np.flatnonzero(dag[:, j]).tolist() if i in keep]
        for i in parents:
            adj[i].add(j)
            adj[j].add(i)
        for a, b in itertools.combinations(parents, 2):
            adj[a].add(b)
            adj[b].add(a)
    seen = {x}
    stack = [x]
    while stack:
        v = stack.pop()
        for u in adj[v]:
            if u == y:
                return False
            if u not in seen and u not in z:
                seen.add(u)
                stack.append(u)
    return True


def separable_in(dag: np.ndarray, x: int, y: int, pattern: np.ndarray) -> bool:
    """The rule: the observed ancestors of the pair, within ``pattern``, separate it."""
    an = ancestors(dag, {x, y})
    s = [c for c in an if pattern[c] and c not in (x, y)]
    return d_separated(dag, x, y, s)


def separable_brute(dag: np.ndarray, x: int, y: int, pattern: np.ndarray) -> bool:
    """Brute force: some subset of ``pattern`` less the pair separates it."""
    rest = [c for c in np.flatnonzero(pattern).tolist() if c not in (x, y)]
    for size in range(len(rest) + 1):
        for s in itertools.combinations(rest, size):
            if d_separated(dag, x, y, s):
                return True
    return False


def classes(dag: np.ndarray, patterns: np.ndarray) -> dict:
    """``cls[i, j]`` in {1, 2, 3} for i < j (0 on and below the diagonal), and the E_adj mask."""
    d = dag.shape[0]
    patterns = np.asarray(patterns, dtype=bool)
    skel = (dag + dag.T) > 0
    co = patterns.T.astype(int) @ patterns.astype(int) > 0
    cls = np.zeros((d, d), dtype=np.int8)
    for i in range(d):
        for j in range(i + 1, d):
            if not co[i, j]:
                cls[i, j] = 2
            elif skel[i, j]:
                cls[i, j] = 1
            else:
                holding = [p for p in patterns if p[i] and p[j]]
                cls[i, j] = 1 if any(separable_in(dag, i, j, p) for p in holding) else 3
    apart = np.zeros(d, dtype=bool)
    ii, jj = np.nonzero(cls == 2)
    apart[ii] = apart[jj] = True
    upper_skel = np.triu(skel, 1)
    e_adj = upper_skel & (cls == 1) & (apart[:, None] | apart[None, :])
    return {"cls": cls, "e_adj": e_adj}


def lagged_classes(design: np.ndarray, tau_max: int, least: int = 4) -> np.ndarray:
    """``cls[k, i, j]``: 2 when fewer than ``least`` time points observe X_i(t-k) and X_j(t), else
    1 (class iii is not computed for the lagged design). Lag 0 uses ``i < j`` only."""
    n, d = design.shape
    cls = np.ones((tau_max + 1, d, d), dtype=np.int8)
    for k in range(tau_max + 1):
        a = design[: n - k].astype(int)
        b = design[k:].astype(int)
        co = a.T @ b  # co[i, j]: time points with X_i(t-k) and X_j(t) observed
        cls[k] = np.where(co < least, 2, 1)
    cls[0] = np.triu(cls[0], 1)
    return cls


def check_rule(n_graphs: int = 200, seed: int = 0) -> dict:
    """Compare :func:`separable_in` with :func:`separable_brute` on random graphs of up to 12
    nodes: how many cases were checked and how many disagree."""
    g = np.random.default_rng([seed, 77])
    checked = mismatches = 0
    for _ in range(n_graphs):
        d = int(g.integers(4, 13))
        order = g.permutation(d)
        dag = np.zeros((d, d), dtype=np.int8)
        p = g.uniform(0.15, 0.5)
        for a in range(d):
            for b in range(a + 1, d):
                if g.random() < p:
                    dag[order[a], order[b]] = 1
        k = int(g.integers(1, 4))
        pats = g.random((k, d)) < g.uniform(0.4, 0.9)
        skel = (dag + dag.T) > 0
        for i in range(d):
            for j in range(i + 1, d):
                if skel[i, j]:
                    continue
                for pat in pats:
                    if not (pat[i] and pat[j]):
                        continue
                    if sum(pat) > 11:  # keep the brute force small
                        continue
                    checked += 1
                    if separable_in(dag, i, j, pat) != separable_brute(dag, i, j, pat):
                        mismatches += 1
    return {"checked": checked, "mismatches": mismatches}
