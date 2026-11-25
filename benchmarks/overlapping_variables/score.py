"""Score one fit record against the truth: ``score(record) -> dict`` of metrics.

The truth comes from the generators without the data: the DAG of a cohort seed, the fixed DAG of a
hand case, and the lag-1 link mask of a lagged seed. Pair classes are fixed on the structural
cohort patterns (:mod:`classes`). A metric is ``None`` where its denominator is empty.

Metrics of a graph (PC and FCI), over the pairs ``i < j``:

- ``shd``: for ``pc`` the CPDAG against the true CPDAG, one for each pair with a missing, extra, or
  differently marked edge; for ``fci`` the skeleton's errors. ``shd_empty`` is the empty graph's,
  the number of true edges;
- ``all_*``, ``i_*``, ``ii_*``: precision, recall, and F1 of the skeleton over all pairs, over
  the class-i pairs (observed together), and over the class-ii pairs (never observed together);
  ``i3_*`` adds the class-iii pairs to the class-i pairs;
- ``eadj_rec``: recall over the true edges next to a pair that is never observed together;
- ``false_colliders``: unshielded colliders of the graph that the true graph does not have;
- ``head_prec``, ``head_rec``: arrowhead precision, and the share of the CPDAG's compelled edges
  whose arrowhead the graph has, leaving the undecided pairs out;
- ``ul_n``, ``ul_true``: the pairs ``undecided_pairs`` lists, and how many are true edges;
- ``und_drawn_n``, ``ii_drawn``, ``ii_unlisted``: listed pairs that are drawn, class-ii pairs that
  are drawn, and class-ii pairs neither drawn nor listed (all should be 0);
- ``ps_n``, ``ps_share``: the drawn edges ``possibly_separable`` lists, and their share of the
  drawn edges; ``iii_adj`` is the drawn false edges of class iii.

A lagged fit has ``all_*``, ``i_*``, and ``ii_*`` over the links (lags 1 to 2 for ``pcmci``, 0 to 2
for ``pcmciplus``), ``und_n``, ``und_drawn``, and ``ii_adj_unlisted``.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np

from . import classes, generate


def _div(a: float, b: float) -> float | None:
    return None if b == 0 else a / b


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


@functools.lru_cache(maxsize=4096)
def truth(design: str, cell: str, seed: int) -> dict:
    """The true DAG and CPDAG of a fit's data, its pair classes, and the ancestor matrix."""
    if design == "hand":
        ds = generate.make("hand", cell, 0)  # the DAG and patterns do not depend on the seed
        dag, patterns = ds["dag"], ds["design"]
    elif design == "cohort":
        config = cell.split("_")[0]
        d = generate.dims(config)
        _, w, _ = generate.random_sem(d, generate.CONFIGS[config]["degree"], seed)
        dag = (w != 0).astype(np.int8)
        patterns = generate.cohort_patterns(config)
    else:
        raise ValueError(design)
    c = classes.classes(dag, patterns)
    from andrey.metrics.graph import to_cpdag

    cpdag = to_cpdag(dag).to_numpy()
    anc = np.zeros_like(dag, dtype=bool)  # anc[i, j]: i is an ancestor of j
    for j in range(dag.shape[0]):
        for i in classes.ancestors(dag, {j}):
            if i != j:
                anc[i, j] = True
    return {"dag": dag, "cls": c["cls"], "e_adj": c["e_adj"], "cpdag": cpdag, "anc": anc}


def colliders(m: np.ndarray) -> list[tuple[int, int, int]]:
    """Unshielded colliders ``x *-> z <-* y`` of a mark matrix (``m[i, j]``: the mark at i)."""
    d = m.shape[0]
    out = []
    for z in range(d):
        heads = [x for x in range(d) if m[x, z] != 0 and m[z, x] == 2]
        for x, y in itertools.combinations(heads, 2):
            if m[x, y] == 0:
                out.append((x, y, z))
    return out


def score(rec: dict) -> dict:
    """The metrics of one fit record; empty unless the fit finished."""
    if rec.get("status") != "ok":
        return {}
    if rec["design"] == "lagged":
        return score_lagged(rec)
    t = truth(rec["design"], rec["cell"], int(rec["seed"]))
    dag, cls, e_adj, cpdag, anc = t["dag"], t["cls"], t["e_adj"], t["cpdag"], t["anc"]
    d = dag.shape[0]
    m = np.zeros((d, d), dtype=np.int8)
    for i, j, mi, mj in rec["edges"]:
        m[i, j], m[j, i] = mi, mj
    iu = np.triu_indices(d, 1)
    skel = (dag + dag.T) > 0
    est, true, c, eadj = (m != 0)[iu], skel[iu], cls[iu], e_adj[iu]

    def sym(pairs) -> np.ndarray:
        z = np.zeros((d, d), dtype=bool)
        for a, b in pairs or []:
            z[a, b] = z[b, a] = True
        return z

    und_full, flag_full = sym(rec.get("undecided")), sym(rec.get("possibly"))
    und, flag = und_full[iu], flag_full[iu]
    out: dict = {}

    def block(prefix: str, mask: np.ndarray) -> None:
        tp = int((est & true & mask).sum())
        fp = int((est & ~true & mask).sum())
        fn = int((~est & true & mask).sum())
        p, r = _div(tp, tp + fp), _div(tp, tp + fn)
        out[f"{prefix}prec"], out[f"{prefix}rec"], out[f"{prefix}f1"] = p, r, _f1(p, r)
        out[f"{prefix}shd"] = fp + fn
        out[f"{prefix}rho"] = _div(int((true & mask).sum()), int(mask.sum()))

    block("all_", np.ones_like(true))
    block("i_", c == 1)
    block("i3_", (c == 1) | (c == 3))  # class iii pairs hold no true edge: they add false edges
    block("ii_", c == 2)
    out["iii_adj"] = int((est & (c == 3)).sum())
    out["eadj_rec"] = _div(int((est & true & eadj).sum()), int((true & eadj).sum()))
    out["true_edges"] = int(true.sum())
    out["shd_empty"] = int(true.sum())
    if rec["method"] == "pc":
        out["shd"] = sum(
            (m[i, j], m[j, i]) != (cpdag[i, j], cpdag[j, i]) for i, j in zip(*iu, strict=True)
        )
    else:
        out["shd"] = out["all_shd"]
    out["ul_n"] = int(und.sum())
    out["ul_true"] = int((und & true).sum())
    out["und_drawn_n"] = int((und & est).sum())
    out["ii_drawn"] = int((est & (c == 2)).sum())
    out["ii_unlisted"] = int((~est & ~und & (c == 2)).sum())
    out["ps_n"] = int((flag & est).sum())
    out["ps_share"] = _div(int((flag & est).sum()), int(est.sum()))
    # arrowheads on decided edges: the pairs the arm lists as undecided are left out
    leave = und_full if rec["arm"] in ("a", "amb") else np.zeros_like(und_full)
    heads = good = 0
    for x in range(d):
        for y in range(d):
            if x == y or m[x, y] == 0 or leave[x, y] or m[y, x] != 2:
                continue
            heads += 1  # an arrowhead at y on x *-> y
            if rec["method"] == "pc":
                good += int(dag[x, y] == 1 and m[x, y] == 1)
            else:
                good += int(not anc[y, x])
    compelled = int(((cpdag == 2) & (cpdag.T == 1)).sum())
    found = sum(
        int(cpdag[y, x] == 2 and cpdag[x, y] == 1 and m[y, x] == 2 and not leave[x, y])
        for x in range(d)
        for y in range(d)
    )
    out["head_prec"], out["head_rec"], out["heads"] = (
        _div(good, heads),
        _div(found, compelled),
        heads,
    )
    fc = 0
    for x, y, z in colliders(m):
        real = dag[x, z] == 1 and dag[y, z] == 1 and not skel[x, y]
        fc += int(not real)
    out["false_colliders"] = fc
    return out


@functools.lru_cache(maxsize=1024)
def lagged_truth(seed: int) -> dict:
    """The lag-1 links of a lagged seed and the classes of its structural mask."""
    ds = generate.lagged("nolap0", seed)
    return {"links": ds["links"], "cls": classes.lagged_classes(ds["design"], generate.TAU_MAX)}


def score_lagged(rec: dict) -> dict:
    """The metrics of a lagged fit; lag 0 is scored for ``pcmciplus`` only."""
    t = lagged_truth(int(rec["seed"]))
    links, cls = t["links"], t["cls"]
    d = links.shape[1]
    tau = generate.TAU_MAX
    lo = 1 if rec["method"] == "pcmci" else 0
    est = np.zeros((tau + 1, d, d), dtype=bool)
    for k, i, j in rec["links"]:
        est[k, i, j] = True
    for i, j, _, _ in rec.get("lag0", []):
        est[0, min(i, j), max(i, j)] = True
    und = np.zeros_like(est)
    for k, i, j in rec.get("undecided", []):
        if k == 0:
            und[0, min(i, j), max(i, j)] = True
        else:
            und[k, i, j] = True
    scope = np.zeros_like(est)
    scope[lo:] = True
    scope[0] = np.triu(scope[0], 1)
    true = links.astype(bool) & scope
    out: dict = {}

    def block(prefix, mask):
        tp = int((est & scope & true & mask).sum())
        fp = int((est & scope & ~true & mask).sum())
        fn = int((~est & true & mask).sum())
        p, r = _div(tp, tp + fp), _div(tp, tp + fn)
        out[f"{prefix}prec"], out[f"{prefix}rec"], out[f"{prefix}f1"] = p, r, _f1(p, r)
        out[f"{prefix}shd"] = fp + fn

    block("all_", scope)
    block("i_", scope & (cls == 1))
    block("ii_", scope & (cls == 2))
    out["und_n"] = int((und & scope).sum())
    out["und_drawn"] = int((und & est & scope).sum())
    out["ii_adj_unlisted"] = int((est & scope & (cls == 2) & ~und).sum())
    out["true_edges"] = int(true.sum())
    out["shd"] = out["all_shd"]
    out["shd_empty"] = int(true.sum())
    return out
