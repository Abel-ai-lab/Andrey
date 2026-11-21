"""The steps every public method shares, run around its facade from the method's spec.

:func:`method_spec` wraps a facade so that each call maps a former argument name or value to the
current one, checks every argument against its :class:`~andrey.spec.models.ParamSpec`, resolves
the seed, warns that an experimental method is experimental, checks the parts of its background
knowledge against the ones the method honors, picks a test or a score the call leaves to the
method, restricts a CPDAG result by the knowledge's causal paths, and names the method, the seed,
the test and score the fit used, the score's penalties, and the configuration's tie-break rule in
the result's metadata. The data check that follows, in :func:`andrey.api._adapt.data_matrix`,
warns when the data are larger than the method's measured boundary. A facade keeps only its
method's own logic.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from andrey.api._adapt import default_score, default_test, label_of
from andrey.core import environments
from andrey.core.backend import config
from andrey.core.ci import (
    DISCRETE_INDEP_TESTS,
    SHIPPED_INDEP_TESTS,
    STOCHASTIC_INDEP_TESTS,
    available_indep_tests,
)
from andrey.core.ci_missing import (
    NOT_TESTED_CLAIM,
    POSSIBLY_SEPARABLE_CLAIM,
    TESTWISE,
    TESTWISE_CLAIM,
    named_pairs,
)
from andrey.core.knowledge import PATH_PARTS, Knowledge, resolve
from andrey.core.output import RegimeOutput
from andrey.core.removed import removal_hint
from andrey.core.score import LINEAR_SCORES, PENALIZED_SCORES, RANK_SCORES
from andrey.core.seeding import resolve_seed
from andrey.core.structure import ARROW, TAIL, column_labels
from andrey.core.warning_policy import (
    DeprecatedArgumentWarning,
    ExperimentalWarning,
    KnowledgeWarning,
    log_once,
    warn_each,
    warn_once,
)

if TYPE_CHECKING:
    from andrey.core.knowledge import Constraints
    from andrey.core.output import StructureOutput
    from andrey.spec.models import MethodSpec, ParamSpec

__all__ = ["method_spec"]

_log = logging.getLogger(__name__)

Facade = TypeVar("Facade", bound=Callable[..., Any])


@functools.cache
def spec_of(name: str) -> MethodSpec:
    """Return the method's spec, read from the registry on the first call."""
    from andrey.spec.registry import REGISTRY

    return REGISTRY[name]


def method_spec(name: str) -> Callable[[Facade], Facade]:
    """Run the facade of method ``name`` inside the steps its spec defines.

    Before the facade: a former argument name or value becomes the current one, and a dropped
    argument is removed, with one :class:`~andrey.DeprecatedArgumentWarning` for each; an argument
    or a choice that an earlier release removed raises as an unknown one does, and the message
    gives its current name (:mod:`andrey.core.removed`); every argument the call passes is checked
    and converted by :meth:`~andrey.spec.models.ParamSpec.check`, which raises ``TypeError`` or
    ``ValueError``; for a method that takes ``env``, data given as each environment's rows apart
    become one table and ``env`` (:func:`pool_environments`); a row-aligned argument that names a
    column of the data is taken out of it (:func:`take_columns`); a
    seeded method's seed is resolved (:func:`~andrey.core.seeding.resolve_seed`) and passed on as
    an int; an experimental method warns :class:`~andrey.ExperimentalWarning`
    (:func:`warn_experimental`); a ``knowledge`` part the method does not honor raises
    ``NotImplementedError``, one it honors experimentally warns, and ``required_paths`` and
    ``forbidden_paths`` are resolved and set apart from the knowledge the facade gets
    (:func:`path_constraints`); a ``None`` ``indep_test`` or ``score_func`` is picked from the
    declared column types, and the facade gets the pick and the columns it reads
    (:func:`chosen_settings`); a call that passes ``ebic_gamma`` or ``equivalent_sample_size`` with
    a score that does not use it raises ``ValueError`` (:func:`check_penalties`), and so does one
    that passes ``degrees_of_freedom`` with a test that does not read it
    (:func:`check_degrees_of_freedom`).
    After it: a CPDAG result keeps the DAGs it stands for that meet ``required_paths`` and
    ``forbidden_paths`` (:func:`restrict_paths`), a PAG or a DAG is checked against them
    (:func:`meet_paths`), and a claim the result does not meet is dropped with a warning
    (:func:`drop_unmet`); each result's ``metadata["algorithm"]`` is
    ``name``, ``metadata["indep_test"]`` and ``metadata["score_func"]`` are the test and the score
    the fit used, for a method that takes them, with the penalties that score used
    (:func:`used_penalties`);
    ``metadata["tie_break"]`` is :data:`andrey.config`'s ``tie_break`` for a method whose result it
    can change (``MethodSpec.tie_break``); when the fit draws random numbers
    (:func:`draws_random_numbers`), ``metadata["seed"]`` is the seed used; and, when its tests met
    missing values, ``metadata["missing"]`` is how it handled them (:func:`missing_used`).
    """

    def decorate(facade: Facade) -> Facade:
        signature = inspect.signature(facade)

        @functools.wraps(facade)
        def run(*args: Any, **kwargs: Any) -> Any:
            spec = spec_of(name)
            current = current_names(spec, kwargs)
            try:
                bound = signature.bind(*args, **current)
            except TypeError:
                if (hinted := removed_name_error(name, signature, current)) is None:
                    raise
                raise hinted from None
            params = {p.name: p for p in spec.params}
            for key, value in bound.arguments.items():
                if key in params:
                    bound.arguments[key] = check_argument(params[key], value, name)
            pool_environments(spec, bound.arguments)
            take_columns(spec, bound.arguments)
            paths = path_constraints(spec, bound.arguments)
            stamp: dict[str, object] = {"algorithm": name}
            if spec.seeded:
                seed = bound.arguments["seed"] = resolve_seed(bound.arguments.get("seed"))
                if draws_random_numbers(spec, bound.arguments):
                    stamp["seed"] = seed
            if spec.status == "experimental":
                warn_experimental(name)
            chosen = chosen_settings(spec, bound.arguments)
            check_penalties(spec, bound.arguments, chosen.get("score_func"))
            check_degrees_of_freedom(spec, bound.arguments, chosen.get("indep_test"))
            out = facade(*bound.args, **bound.kwargs)
            if paths is not None:
                out = (restrict_paths if spec.restricts_paths else meet_paths)(out, paths, name)
            stamp.update(chosen)
            stamp.update(used_penalties(spec, bound.arguments, chosen.get("score_func")))
            if spec.tie_break:
                stamp["tie_break"] = config.tie_break
            if isinstance(out, list):
                return [stamped(o, stamp | missing_used(spec, bound.arguments, o)) for o in out]
            if isinstance(out, RegimeOutput):  # the run and each regime's graph carry the stamp
                outputs = tuple(stamped(o, stamp) for o in out.outputs)
                metadata = {"algorithm": name, **out.metadata, **stamp}
                return dataclasses.replace(out, _outputs=outputs, _metadata=metadata)
            return stamped(out, stamp | missing_used(spec, bound.arguments, out))

        return run  # ty: ignore[invalid-return-type]  # functools.wraps keeps the signature

    return decorate


def warn_experimental(name: str) -> None:
    """Warn on the first call of experimental method ``name`` in this process."""
    warn_once(
        f"andrey.{name} is experimental: it is not supported (see its Limitations), and its API "
        "may change without deprecation.",
        ExperimentalWarning,
    )


def unhonored_part(method: str, part: str, setting: str = "") -> str:
    """Return the message for knowledge ``part`` given to ``method``, which does not honor it.

    ``setting`` names the arguments under which ``method`` does not honor it, such as an undirected
    ``structure``, and is empty when it never does. The message names the methods that honor it,
    from the registry, and for ``required_paths`` and ``forbidden_paths`` the methods that honor
    them only with a CPDAG result (:func:`undirected_setting`).
    """
    from andrey.spec.registry import REGISTRY

    honoring = [name for name, spec in REGISTRY.items() if part in spec.knowledge]
    others = (
        f"the methods that honor it are {', '.join(honoring)}"
        if honoring
        else "no method honors it"
    )
    local = [name for name in honoring if structure_param(REGISTRY[name]) is not None]
    if part in PATH_PARTS and local:
        others += f" ({', '.join(local)} with structure='cpdag' and no target)"
    who = f"{method} with {setting}" if setting else method
    return f"{who} does not honor knowledge {part}; {others}"


def structure_param(spec: MethodSpec) -> ParamSpec | None:
    """Return the ``structure`` choice of a local-structure method, else ``None``."""
    return next((p for p in spec.params if p.concept == "structure"), None)


def undirected_setting(spec: MethodSpec, arguments: dict[str, Any]) -> str | None:
    """Return the setting under which the call's result is an undirected graph, else ``None``.

    A method with a ``structure`` choice returns one for a ``target`` and for a ``structure`` that
    :data:`~andrey.spec.concepts.BLANKET_STRUCTURES` makes undirected, the default when the call
    passes none. An undirected graph stands for no set of DAGs that the causal paths could restrict.
    """
    from andrey.spec.concepts import BLANKET_STRUCTURES

    structure = structure_param(spec)
    if structure is None:
        return None
    if arguments.get("target") is not None:
        return "a target"
    choice = arguments.get("structure", structure.default)
    return f"structure={choice!r}" if BLANKET_STRUCTURES[choice] == "undirected" else None


def path_constraints(spec: MethodSpec, arguments: dict[str, Any]) -> Constraints | None:
    """Check the call's ``knowledge`` against the parts ``spec`` honors, and set its paths apart.

    A part the method does not honor raises ``NotImplementedError`` (:func:`unhonored_part`), as
    do ``required_paths`` and ``forbidden_paths`` for a call whose result is undirected
    (:func:`undirected_setting`), and a part it honors experimentally warns
    :class:`~andrey.ExperimentalWarning` once. Knowledge with ``required_paths`` or
    ``forbidden_paths`` is resolved against the data's columns before the fit
    (:func:`~andrey.core.knowledge.resolve`, which raises
    :class:`~andrey.core.knowledge.KnowledgeError` for knowledge that contradicts itself). For a
    method that ``restricts_paths``, the facade gets the knowledge without them and the return
    value holds them for :func:`restrict_paths`; a method whose search reads them gets the whole
    knowledge, checked with the ancestral reading of tiers, and the return value holds them for
    :func:`meet_paths`. Without them the return value is ``None``. A method that learns time lags
    resolves its own lagged claims. Anything but a :class:`~andrey.Knowledge` is left to the
    facade.
    """
    knowledge = arguments.get("knowledge")
    if not isinstance(knowledge, Knowledge):
        return None
    given = knowledge.parts()
    for part in given:
        if part not in spec.knowledge:
            raise NotImplementedError(unhonored_part(spec.name, part))
    for part in given:
        if part in spec.knowledge_experimental:
            warn_once(
                f"andrey.{spec.name}: knowledge {part} is experimental: no published benchmark "
                "of it has passed, and its meaning may change without deprecation.",
                ExperimentalWarning,
            )
    if not set(given) & set(PATH_PARTS) or spec.family == "temporal":
        return None
    if (setting := undirected_setting(spec, arguments)) is not None:
        part = next(p for p in given if p in PATH_PARTS)
        raise NotImplementedError(unhonored_part(spec.name, part, setting))
    if spec.restricts_paths:
        arguments["knowledge"] = knowledge.edge_parts()
    data = arguments[spec.data.inputs[0].name]
    if isinstance(data, (list, tuple)) and data:  # groups of rows share their columns
        data = data[0]
    shape = np.shape(data)
    if len(shape) != 2:  # the facade's data check names the problem
        return None
    return resolve(
        knowledge,
        shape[1],
        column_labels(data),
        method=spec.name,
        ancestral_tiers=not spec.restricts_paths,
    )


def restrict_paths(out: StructureOutput, constraints: Constraints, method: str) -> StructureOutput:
    """Return ``out`` with its CPDAG restricted by the causal paths of ``constraints``.

    Of the DAGs the CPDAG stands for, the structure becomes the MPDAG of those that meet
    ``required_paths`` and ``forbidden_paths`` (:func:`~andrey.core.ancestral.restrict`), and
    ``metadata["ancestor_clauses"]`` lists the clauses the MPDAG cannot show, each
    ``[x, [c1, c2, ...]]``: ``x`` is a direct cause of at least one of ``c1, c2, ...``. A claim no
    such DAG meets is dropped (:func:`drop_unmet`). A CPDAG with no consistent extension stands for
    no DAG: there a claim on a pair joined by an undirected edge directs that edge, Meek's rules
    follow, and ``metadata["edge_claims"]`` lists the claims applied this way, each
    ``[part, a, b]``; the list is empty for a CPDAG with a consistent extension. An undirected
    structure, which the paths do not orient, is returned as it is.
    """
    from andrey.core.ancestral import restrict
    from andrey.core.orient import from_structure, to_structure

    structure = out.structure
    if getattr(structure, "kind", None) != "cpdag":
        return out
    # The claims name the data's columns by position, and the graph is over the same columns.
    assert structure.n_nodes == constraints.d, (structure.n_nodes, constraints.d)
    labels = structure.labels
    name = label_of(labels)
    edge_claims: list[tuple[str, int, int]] = []
    mpdag, clauses, unmet = restrict(
        from_structure(structure),  # ty: ignore[invalid-argument-type]  # a cpdag is a graph
        constraints.required_paths,
        constraints.forbidden_paths,
        edge_claims=edge_claims,
    )
    metadata = {
        **out.metadata,
        "ancestor_clauses": [[name(t), [name(h) for h in sorted(heads)]] for t, heads in clauses],
        "edge_claims": [[part, name(a), name(b)] for part, a, b in edge_claims],
    }
    out = dataclasses.replace(
        out, _structure=to_structure(mpdag, kind="cpdag", labels=labels), _metadata=metadata
    )
    return drop_unmet(out, unmet, method)


def meet_paths(out: Any, constraints: Constraints, method: str) -> Any:
    """Check ``out``, a result or a list of them, against the causal paths of ``constraints``.

    A PAG or a DAG drops the claims it does not meet (:func:`drop_unmet`): for a PAG, those
    :func:`~andrey.constraint.fci.unmet_path_claims` finds, and for a DAG, a required path it lacks
    and a forbidden path it has. A PAG with an index column after the data's columns reads the
    claims over its first columns.
    """
    from andrey.constraint.fci import reach, unmet_path_claims

    if isinstance(out, list):
        return [meet_paths(each, constraints, method) for each in out]
    structure = out.structure
    kind = getattr(structure, "kind", None)
    if kind not in ("pag", "dag"):
        return out
    marks = structure.to_numpy()
    d = constraints.d
    assert len(marks) >= d, (len(marks), d)
    claims = {}
    for part, pairs in (
        ("required_paths", constraints.required_paths),
        ("forbidden_paths", constraints.forbidden_paths),
    ):
        full = np.zeros(marks.shape, dtype=bool)
        if pairs is not None:
            full[:d, :d] = pairs
        claims[part] = full
    if kind == "pag":
        unmet = unmet_path_claims(marks, claims["required_paths"], claims["forbidden_paths"])
    else:
        directed = (marks == TAIL) & (marks.T == ARROW)
        unmet = [
            (part, a, b)
            for part, pairs in claims.items()
            for a, b in np.argwhere(pairs).tolist()
            if bool(reach(directed, a)[b]) != (part == "required_paths")
        ]
    return drop_unmet(out, unmet, method)


def drop_unmet(out: Any, unmet: Sequence[Sequence[Any]], method: str) -> Any:
    """Return ``out`` with ``metadata["unmet_knowledge"]``, the claims its graph does not meet.

    Each claim in ``unmet`` is ``(part, a, b)``, or ``(part, a, b, lag)`` for a lagged claim, with
    ``a`` and ``b`` column positions, recorded as ``[part, a, b]`` or ``[part, a, b, lag]`` with
    the columns' labels. When there is one, the fit warns once with
    :class:`~andrey.KnowledgeWarning`, naming them all.
    """
    name = label_of(out.structure.labels)
    listed = [[part, name(a), name(b), *lag] for part, a, b, *lag in unmet]
    if listed:
        claims = ", ".join(
            f"{part} ({', '.join(repr(v) for v in claim)})" for part, *claim in listed
        )
        warn_each(
            f"andrey.{method}: the graph found does not meet knowledge {claims}, so the fit drops "
            "them and lists them in metadata['unmet_knowledge']",
            KnowledgeWarning,
        )
    return dataclasses.replace(out, _metadata={**out.metadata, "unmet_knowledge": listed})


def removed_name_error(
    method: str, signature: inspect.Signature, kwargs: dict[str, Any]
) -> TypeError | None:
    """Return the error for the first argument in ``kwargs`` that a release removed, else ``None``.

    The message is Python's for an unknown argument, with what the name is called now
    (:func:`~andrey.core.removed.removal_hint`).
    """
    for key in kwargs:
        hint = None if key in signature.parameters else removal_hint(method, key)
        if hint is not None:
            return TypeError(f"{method}() got an unexpected keyword argument {key!r}; {hint}")
    return None


def pool_environments(spec: MethodSpec, arguments: dict[str, Any]) -> None:
    """Pool data given as each environment's rows apart into one table and ``env``, in place.

    Only for a method that takes ``env``: a list of tables labels each environment by its
    position, and a mapping by its key, so the method, its checks, and its metadata read one table
    and the label of each row (:func:`~andrey.core.environments.pool_environments`, which raises
    ``ValueError`` when the call also passes ``env``).
    """
    if not any(i.name == "env" for i in spec.data.inputs[1:]):
        return
    data_key = spec.data.inputs[0].name
    arguments[data_key], env = environments.pool_environments(
        arguments[data_key],
        arguments.get("env"),
        arguments.get("targets"),
        spec.name,
        missing=spec.data.missing,
    )
    if env is not None:
        arguments["env"] = env


def take_columns(spec: MethodSpec, arguments: dict[str, Any]) -> None:
    """Take each column that a row-aligned argument names out of the data, in ``arguments``.

    The row-aligned arguments are the inputs after the data and a ``targets`` parameter. One given
    as a string names a column of the DataFrame ``data``: the data lose that column, and the
    argument becomes the column's values, so the method, its default test or score, and its
    metadata read only the variables. Raises ``ValueError`` when the data have no such column or
    two arguments name the same one (:func:`~andrey.core.environments.take_columns`).
    """
    keys = [i.name for i in spec.data.inputs[1:]] + [
        p.name for p in spec.params if p.type == "targets"
    ]
    named = {key: arguments[key] for key in keys if isinstance(arguments.get(key), str)}
    data_key = spec.data.inputs[0].name
    arguments[data_key], values = environments.take_columns(arguments[data_key], named, spec.name)
    arguments.update(values)


def check_argument(param: ParamSpec, value: Any, method: str) -> object:
    """Return ``value`` checked and converted by ``param``.

    Raises what :meth:`~andrey.spec.models.ParamSpec.check` raises; when ``value`` is a choice that
    a release removed, the ``ValueError`` adds what the choice is called now
    (:func:`~andrey.core.removed.removal_hint`), and when it is a choice of the shared parameter
    that this method does not offer, why not (:class:`~andrey.spec.concepts.Deviation`).
    """
    try:
        return param.check(value, method, choices=runtime_choices(param))
    except ValueError as error:
        hint = removal_hint(method, param.name, value)
        if hint is not None:
            raise ValueError(f"{error}; {hint}") from None
        reason = withheld_reason(param, value, method)
        if reason is None:
            raise
        raise ValueError(f"{error}. {reason}") from None


def withheld_reason(param: ParamSpec, value: Any, method: str) -> str | None:
    """Why ``method`` does not offer ``value``, a choice of the concept ``param`` implements.

    ``None`` when the concept offers no such choice or no deviation of the method withholds it.
    """
    from andrey.spec.concepts import CONCEPTS

    concept = CONCEPTS.get(param.concept) if param.concept else None
    if concept is None or concept.param is None or value not in (concept.param.choices or ()):
        return None
    for deviation in concept.deviations:
        if deviation.method == method and value not in (deviation.choices or (value,)):
            return deviation.reason
    return None


def draws_random_numbers(spec: MethodSpec, arguments: dict[str, Any]) -> bool:
    """Whether a fit of the seeded method ``spec`` with ``arguments`` draws random numbers.

    A method that takes ``tuning_iter`` always does: its optimization draws in every fit, whatever
    test it runs; so does a method that returns regimes, whose every start draws its first regime
    assignment. A method with an ``indep_test`` draws them only under a randomized test
    (:data:`~andrey.core.ci.STOCHASTIC_INDEP_TESTS`) or, when it takes ``n_starts``, for the starts
    after the first, with ``n_starts`` above 1, unless it also takes a ``score_func``: then a
    search over variable orders, visited in a random order, runs before the tests (``boss_fci``,
    ``grasp_fci``), and it always does. A method with ``n_perturbations`` draws them only for the
    random changes before a later start, with ``n_starts`` above 1; a method with a ``family``
    draws them only for a count family (the randomized probability transform); every other seeded
    method always does. ``pc``'s ``missing="mvpc"`` draws them only on data with missing values,
    which this check does not read: ``pc`` records its seed itself when the correction ran.
    """
    if any(p.name == "tuning_iter" for p in spec.params) or spec.output.regimes:
        return True  # an optimization's draws, or a regime search's random first assignments
    test = next((p for p in spec.params if p.name == "indep_test"), None)
    starts = next((p for p in spec.params if p.name == "n_starts"), None)
    order_search = starts is not None and any(p.name == "score_func" for p in spec.params)
    later = starts is not None and arguments.get("n_starts", starts.default) > 1
    if test is not None and not order_search:
        return later or arguments.get("indep_test", test.default) in STOCHASTIC_INDEP_TESTS
    if any(p.concept == "n_perturbations" for p in spec.params):
        return later
    family = next((p for p in spec.params if p.name == "family"), None)
    if family is not None:
        from andrey.fcm.cpcm import FAMILIES

        named = arguments.get("family", family.default)
        names = named if isinstance(named, (list, tuple)) else [named]
        return any(FAMILIES[n].counts for n in names if isinstance(n, str) and n in FAMILIES)
    return True


#: The concepts of the arguments that name a test or a score, each with the function that picks it
#: from the declared column types when the call passes none.
CHOSEN = {"indep_test": default_test, "score_func": default_score}


def chosen_settings(spec: MethodSpec, arguments: dict[str, Any]) -> dict[str, str]:
    """Return the test and the score a fit of ``spec`` uses, by argument name, set in ``arguments``.

    A method with an ``indep_test`` or a ``score_func`` has it in the result: the name the call
    passed, or, for ``None``, the one picked from the declared column types
    (:func:`~andrey.api._adapt.default_test`, :func:`~andrey.api._adapt.default_score`), which
    replaces ``None`` in ``arguments``. A registered test is its registered name. ``categorical``
    in ``arguments`` becomes the columns the test reads, or the score for a method without a test:
    a mixed-data pick lists them by position, and an all-categorical pick needs none.
    """
    data = arguments[spec.data.inputs[0].name]
    declared = arguments.get("categorical")
    chosen: dict[str, str] = {}
    columns: dict[str, object] = {}
    for param in spec.params:
        concept = param.concept or ""
        pick = CHOSEN.get(concept)
        if pick is None:
            continue
        name, columns[concept] = pick(
            data, arguments.get(param.name, param.default), categorical=declared
        )
        arguments[param.name] = chosen[param.name] = name
    if columns:
        arguments["categorical"] = columns.get("indep_test", columns.get("score_func"))
    return chosen


#: The scores each penalty concept weighs; the other scores ignore it.
PENALTIES = {
    "penalty_discount": PENALIZED_SCORES,
    "ebic_gamma": (*LINEAR_SCORES, *RANK_SCORES),
    "equivalent_sample_size": ("bdeu",),
}
#: What each argument that only one kind of score reads weighs; a call that passes it with another
#: score raises. ``penalty_discount`` is not among them: every score but BDeu uses it, and BDeu
#: ignores it.
SCORE_ONLY = {
    "ebic_gamma": "the extended-BIC term",
    "equivalent_sample_size": "the Dirichlet prior",
}


def used_penalties(
    spec: MethodSpec, arguments: dict[str, Any], score: str | None
) -> dict[str, float]:
    """Return the penalties the score ``score`` used in a fit of ``spec``, by argument name.

    ``penalty_discount`` weighs every score but BDeu, ``ebic_gamma`` only the linear-Gaussian BIC,
    on the values or on the normal scores, and ``equivalent_sample_size`` only BDeu
    (:data:`PENALTIES`); a penalty the score ignores is left out. A method without ``score_func``
    has one score, which uses every penalty it takes. The value is the call's, or the default.
    """
    one_score = not any(p.concept == "score_func" for p in spec.params)
    return {
        p.name: arguments.get(p.name, p.default)
        for p in spec.params
        if score in PENALTIES.get(p.concept or "", ()) or (one_score and p.concept in PENALTIES)
    }


def check_penalties(spec: MethodSpec, arguments: dict[str, Any], score: str | None) -> None:
    """Raise ``ValueError`` when the call passes an argument that only one score reads with another.

    ``ebic_gamma`` weighs the extended-BIC term of ``"bic"`` and ``"normal_scores"`` alone, and
    ``equivalent_sample_size`` the Dirichlet prior of ``"bdeu"`` alone (:data:`SCORE_ONLY`).
    ``score`` is the one the fit uses (:func:`chosen_settings`). A method without ``score_func`` has
    one score, which reads every argument it takes.
    """
    passed = [
        (p.name, p.concept) for p in spec.params if p.concept in SCORE_ONLY and p.name in arguments
    ]
    if not passed or not any(p.concept == "score_func" for p in spec.params):
        return
    for name, concept in passed:
        used = PENALTIES[concept or ""]
        if score not in used:
            weighed = SCORE_ONLY[concept or ""]
            raise ValueError(
                f"{spec.name}: {name} weighs {weighed} of score_func={used[0]!r}, which "
                f"score_func={score!r} does not use; drop {name}, or pass score_func={used[0]!r}"
            )


def check_degrees_of_freedom(spec: MethodSpec, arguments: dict[str, Any], test: str | None) -> None:
    """Raise ``ValueError`` when the call passes ``degrees_of_freedom`` with a test that ignores it.

    Only the categorical tests (:data:`~andrey.core.ci.DISCRETE_INDEP_TESTS`) read it; ``test`` is
    the one the fit uses (:func:`chosen_settings`). A default is never checked, so a method whose
    default is ``"nonempty"`` runs with any test.
    """
    if "degrees_of_freedom" in arguments and test not in DISCRETE_INDEP_TESTS:
        raise ValueError(
            f"{spec.name}: degrees_of_freedom sets the degrees of freedom of indep_test='chisq' "
            f"and 'gsq', which indep_test={test!r} does not use; drop degrees_of_freedom, or pass "
            "indep_test='gsq'"
        )


def runtime_choices(param: ParamSpec) -> tuple[str, ...] | None:
    """Return the values ``param`` takes at run time where they differ from its spec, else ``None``.

    An ``indep_test`` takes its shipped choices and the tests registered with
    :func:`~andrey.core.ci.register_indep_test`.
    """
    if param.concept != "indep_test" or param.choices is None:
        return None
    registered = [name for name in available_indep_tests() if name not in SHIPPED_INDEP_TESTS]
    return (*param.choices, *registered)


def current_names(spec: MethodSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return ``kwargs`` with the former names and values of ``spec``'s arguments made current.

    The arguments are the data inputs and the parameters. Each former name or value warns once at
    the caller's line, and so does each dropped argument, which is removed. Raises ``ValueError``
    when a call passes an argument under both its former and its current name.
    """
    kwargs = dict(kwargs)
    for old in spec.dropped_args:
        if old in kwargs:
            del kwargs[old]
            warn_once(
                f"andrey.{spec.name}: {old} is ignored and stops being accepted in the next "
                "release; drop it",
                DeprecatedArgumentWarning,
            )
    for argument in (*spec.data.inputs, *spec.params):
        for old in argument.aliases:
            if old not in kwargs:
                continue
            if argument.name in kwargs:
                raise ValueError(
                    f"{spec.name}: {old} is the former name of {argument.name}; pass only "
                    f"{argument.name}"
                )
            kwargs[argument.name] = kwargs.pop(old)
            warn_once(
                f"andrey.{spec.name}: {old} is now {argument.name}; the former name stops working "
                "in the next release",
                DeprecatedArgumentWarning,
            )
    for param in spec.params:
        value = kwargs.get(param.name)
        for old, new in param.choice_aliases:
            if isinstance(value, str) and value == old:
                kwargs[param.name] = new
                warn_once(
                    f"andrey.{spec.name}: {param.name}={old!r} is now {param.name}={new!r}; the "
                    "former value stops working in the next release",
                    DeprecatedArgumentWarning,
                )
    return kwargs


def missing_used(
    spec: MethodSpec, arguments: dict[str, Any], out: StructureOutput
) -> dict[str, object]:
    """Return ``{"missing": value}`` when the fit ``out`` met missing values, else ``{}``.

    A fit met them when its tests counted the rows they read, ``metadata["rows_tested"]``, when it
    ran structural EM, ``metadata["em"]``, or when its data hold a missing value, as a fit that
    runs no test does; such a fit also gets ``{"rows_tested": {"min": 0, "median": 0}}``, as
    :func:`~andrey.core.ci_missing.rows_metadata` gives when no query was tested. The value is the
    call's ``missing`` or, for ``None``, the method's default handling, ``"testwise"``, which the
    call did not name: the first such fit of the process logs :func:`notice_default_missing`. A
    fit without structural EM on data with pairs of columns never observed together also gets
    ``{"never_observed_together": ...}``, the pairs by name (:func:`never_observed_together`),
    counted from the data before the tests read them.
    """
    if not any(p.concept == "missing" for p in spec.params):
        return {}
    data = arguments.get(spec.data.inputs[0].name)
    stamp: dict[str, object] = {}
    if "rows_tested" not in out.metadata and "em" not in out.metadata:
        missing = _missing_cells(data)
        if missing is None or not missing.any():
            return {}
        stamp["rows_tested"] = {"min": 0, "median": 0}
    stamp["missing"] = arguments.get("missing") or TESTWISE
    if "em" not in out.metadata and (pairs := never_observed_together(data)):
        stamp["never_observed_together"] = pairs
    if arguments.get("missing") is None:
        notice_default_missing(spec.name, {**out.metadata, **stamp})
    return stamp


def _missing_cells(data: object) -> np.ndarray | None:
    """Return where the table ``data`` holds a missing value.

    ``None`` for data that is neither a DataFrame nor one table of numbers.
    """
    import pandas as pd

    if isinstance(data, pd.DataFrame):  # of any column types
        return np.asarray(data.isna(), dtype=bool)
    try:
        array = np.asarray(data)
        if array.dtype.kind in "biu":  # no value of these types is missing
            missing = np.zeros(array.shape, dtype=bool)
        elif array.dtype.kind == "f":
            missing = np.isnan(array)
        else:
            missing = np.isnan(np.asarray(array, dtype=np.float64))
    except (TypeError, ValueError):
        return None
    return missing if missing.ndim == 2 else None


def never_observed_together(data: object) -> list[list]:
    """Return the pairs of columns of ``data`` with no row observing both, by name.

    Named by the DataFrame's labels, or by 0-based position, as
    :func:`~andrey.core.ci_missing.named_pairs` lists pairs; ``[]`` for data that is not one table.
    From one product of the observed masks, so it costs one pass over the data.
    """
    if (missing := _missing_cells(data)) is None:
        return []
    observed = (~missing).astype(np.float64)
    together = observed.T @ observed
    pairs = zip(*np.nonzero(np.triu(together == 0, 1)))
    return named_pairs(((int(a), int(b)) for a, b in pairs), label_of(column_labels(data)))


#: The key of :func:`notice_default_missing` in :func:`~andrey.core.warning_policy.log_once`.
MISSING_DEFAULT_NOTICE = "missing_default"
#: The most pairs the notice names; past it, the notice gives their number, and the metadata
#: lists them all.
_NOTICE_PAIRS = 5


def notice_default_missing(method: str, metadata: dict | None = None) -> None:
    """Log, once per process, that ``method`` ran test-wise deletion as its default handling.

    The notice says which handling ran, when it is valid
    (:data:`~andrey.core.ci_missing.TESTWISE_CLAIM`), and how to choose it by name. When the fit's
    ``metadata`` names pairs never observed together, has tests not tested, or names undecided
    pairs, the notice says so too (:func:`_not_tested_notice`); the metadata holds every pair of
    every fit. The worker processes of :func:`~andrey.bootstrap` log none: the calling process
    logs it for their fits.
    """
    log_once(
        _log,
        MISSING_DEFAULT_NOTICE,
        f"andrey.{method}: the data hold missing values and missing=None, so the fit ran the "
        "default handling, test-wise deletion: each test read only the rows where its variables "
        f'are observed. {TESTWISE_CLAIM} Pass missing="testwise" to choose it by name, and read '
        f'the rows the tests kept in metadata["rows_tested"].{_not_tested_notice(metadata or {})} '
        "This notice is logged once per process.",
    )


def _not_tested_notice(metadata: dict) -> str:
    """Return what :func:`notice_default_missing` says of a fit's ``metadata``.

    ``""`` when the fit has no pair never observed together, no test not tested, and no undecided
    pair. Otherwise it gives the number of pairs never observed together and, with tests not
    tested or undecided pairs, the :data:`~andrey.core.ci_missing.NOT_TESTED_CLAIM` and the number
    of undecided pairs, and with possibly separable edges the
    :data:`~andrey.core.ci_missing.POSSIBLY_SEPARABLE_CLAIM` and their number, naming the pairs when
    there are at most ``_NOTICE_PAIRS``. The number of
    conditioning sets not tested, ``metadata["not_tested"]``, is left out: it includes the sets the
    search skipped because no row can test them, so its size follows the search, not what a user
    can change.
    """

    def counted(pairs: list[list], of: str = "", noun: str = "pair") -> str:
        n = len(pairs)
        text = f"{n:,} {noun if n == 1 else noun + 's'}{of}"
        if n <= _NOTICE_PAIRS:
            text += " (" + ", ".join(f"{a} and {b}" for a, b in pairs) + ")"
        return text

    text = ""
    if pairs := metadata.get("never_observed_together"):
        verb = "is" if len(pairs) == 1 else "are"
        text += (
            f" {counted(pairs, ' of columns')} {verb} never observed together, listed in "
            'metadata["never_observed_together"].'
        )
    undecided = metadata.get("undecided_pairs") or []
    if metadata.get("not_tested") or undecided:
        text += f" {NOT_TESTED_CLAIM}"
        if undecided:
            verb = "is" if len(undecided) == 1 else "are"
            text += f" {counted(undecided)} {verb} undecided."
        else:
            text += " No pair is undecided."
    if possibly := metadata.get("possibly_separable"):
        verb = "is" if len(possibly) == 1 else "are"
        text += (
            f" {POSSIBLY_SEPARABLE_CLAIM} {counted(possibly, noun='drawn edge')} {verb} flagged."
        )
    return text


def stamped(out: StructureOutput, stamp: dict[str, object]) -> StructureOutput:
    """Return ``out`` with ``stamp``'s keys in its metadata, ``algorithm`` first."""
    metadata = {"algorithm": stamp["algorithm"], **out.metadata, **stamp}
    return dataclasses.replace(out, _metadata=metadata)
