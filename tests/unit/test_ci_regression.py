"""The regression-based tests GCM, wGCM, and PCM: statistics on fixed inputs, size and power on
fixed cases, and p-values that depend on neither the row, column, nor argument order."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtr
from sklearn.ensemble import _gradient_boosting as sklearn_kernels

from andrey.core import ci
from andrey.core import ci_regression as cr
from tests.release import release_cases

N = 300


def _chain(n: int = N, seed: int = 0) -> np.ndarray:
    """``x0 -> x1 -> x2`` with nonlinear links, and an independent ``x3``."""
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal(n)
    x1 = x0 + np.sin(2 * x0) + 0.3 * rng.standard_normal(n)
    x2 = np.tanh(x1) + 0.3 * rng.standard_normal(n)
    return np.column_stack([x0, x1, x2, rng.standard_normal(n)])


def _sign_changing(n: int = 400, seed: int = 0) -> np.ndarray:
    """``x`` and ``y`` share noise with the sign of ``z``: a conditional covariance that averages
    to zero over ``z``, which the GCM cannot see and a weight on ``z`` can."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    shared = rng.standard_normal(n)
    x = np.sign(z) * shared + 0.5 * rng.standard_normal(n)
    y = shared + 0.5 * rng.standard_normal(n)
    return np.column_stack([x, y, z])


# Linear regressions keep these fast; the default regressions are checked further down.
LINEAR = {
    "gcm": {"regression": "linear"},
    "wgcm": {"regression": "linear"},
    "wgcm_fix": {"regression": "linear", "variant": "fix"},
    "pcm": {"regression": "linear"},
}
CLASSES = {"gcm": cr.GCM, "wgcm": cr.WGCM, "wgcm_fix": cr.WGCM, "pcm": cr.PCM}


def _test(name: str, data: np.ndarray, **options) -> cr._RegressionTest:
    return CLASSES[name](data, **{**LINEAR[name], **options})


def test_the_tests_are_registered_and_seeded():
    for name in ("gcm", "wgcm", "pcm"):
        assert name in ci.SHIPPED_INDEP_TESTS
        assert name in ci.STOCHASTIC_INDEP_TESTS
        assert name in ci.CONTINUOUS_INDEP_TESTS
        assert name in ci.EXPERIMENTAL_INDEP_TESTS
    assert isinstance(ci.make_indep_test("wgcm", _chain(60), variant="fix"), cr.WGCM)


def test_gcm_statistic_on_fixed_residuals():
    rx = np.array([1.0, -1.0, 2.0, 0.5])
    ry = np.array([0.5, -2.0, 1.0, 1.0])
    products = rx * ry  # 0.5, 2, 2, 0.5: mean 1.25, variance over n 0.5625
    statistic, p = cr._gcm_answer(rx, ry)
    assert statistic == pytest.approx(2.0 * 1.25 / 0.75)
    assert p == pytest.approx(2.0 * ndtr(-statistic))
    assert products.mean() == 1.25


def test_a_covariance_without_noise_is_a_dependence():
    x = np.repeat([-1.0, 1.0], 30)
    data = np.column_stack([x, x, np.random.default_rng(0).standard_normal(60)])
    for test in (cr.GCM(data, regression="linear"), cr.WGCM(data, regression="linear")):
        assert test(0, 1, []) == 0.0
    statistic, p = cr._fixed_answer(np.ones(8), np.arange(8.0)[:, None], np.ones((8, 9)))
    assert statistic == np.inf
    assert p == 0.1


def test_pcm_needs_rows_for_a_split_in_each_half():
    with pytest.raises(ValueError, match="12"):
        cr.PCM(_chain(11))
    assert 0.0 <= cr.PCM(_chain(12), n_splits=1)(0, 1, []) <= 1.0


def test_wgcm_est_needs_rows_for_a_split_in_its_estimation_part():
    with pytest.raises(ValueError, match="17"):
        cr.WGCM(_chain(16))
    assert 0.0 <= cr.WGCM(_chain(17), regression="linear")(0, 1, [2]) <= 1.0
    assert 0.0 <= cr.WGCM(_chain(4), regression="linear", variant="fix")(0, 1, [2]) <= 1.0


def test_one_sided_statistics_report_zero_below_the_null():
    statistic, p = cr._estimated_answer(np.array([1.0, -2.0, -1.0, 0.5]), np.ones(4))
    assert statistic == 0.0
    assert p > 0.5


def test_fixed_weights_on_fixed_input():
    products = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    z = np.arange(8.0)[:, None]
    # The draws use the centered columns: a constant multiplier gives draws of 0, below any
    # positive statistic, and so the smallest p-value.
    statistic, p = cr._fixed_answer(products, z, np.full((8, 9), 2.0))
    assert statistic > 0.0
    assert p == 0.1


@pytest.mark.parametrize("name", LINEAR)
def test_size_and_power_on_the_chain(name):
    test = _test(name, _chain())
    assert test(0, 2, [1]) > 0.05  # x0 and x2 are independent given x1
    assert test(0, 3, []) > 0.05  # x3 is independent of everything
    assert test(0, 1, []) < 0.01
    assert test(0, 2, []) < 0.01


@pytest.mark.parametrize("name", LINEAR)
def test_p_values_ignore_row_column_and_argument_order(name):
    x = _chain(120)
    rows = np.random.default_rng(1).permutation(len(x))
    cols = [2, 0, 3, 1]  # data column j sits at position cols.index(j)
    test, shuffled = _test(name, x, seed=3), _test(name, x[rows][:, cols], seed=3)
    at = cols.index
    for a, b, s in [(0, 2, [1]), (1, 3, [0, 2]), (0, 1, [])]:
        p = test(a, b, s)
        assert test(b, a, s[::-1]) == p
        assert shuffled(at(a), at(b), [at(c) for c in s]) == p


# The tree regressions on few rows, so that the checks below cover the layout they read.
TREES = {
    "gcm": {"regression": "boosting"},
    "wgcm": {"regression": "boosting"},
    "pcm": {"regression": "forest", "n_splits": 1},
}


def _nonlinear(n: int = 80, seed: int = 0) -> np.ndarray:
    """``x`` and ``y`` both nonlinear in ``z``, with an interaction of ``x`` and ``z`` in ``y``."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    x = np.sin(2 * z) + 0.5 * rng.standard_normal(n)
    y = np.cos(z) + 0.3 * x * z + 0.5 * rng.standard_normal(n)
    return np.column_stack([x, y, z])


ALL_OPTIONS = [
    *(pytest.param(CLASSES[k], LINEAR[k], id=f"{k}-linear") for k in LINEAR),
    *(pytest.param(CLASSES[k], TREES[k], id=f"{k}-trees") for k in TREES),
    pytest.param(cr.PCM, {"n_splits": 1}, id="pcm-additive"),
]


@pytest.mark.parametrize(("cls", "options"), ALL_OPTIONS)
def test_p_values_ignore_columns_outside_the_query(cls, options):
    x = _nonlinear()
    rng = np.random.default_rng(5)
    wider = np.column_stack([rng.standard_normal(len(x)), x, rng.standard_normal(len(x))])
    test, other = cls(x, seed=3, **options), cls(wider, seed=3, **options)
    for a, b, s in [(0, 1, [2]), (0, 2, [])]:
        assert other(a + 1, b + 1, [c + 1 for c in s]) == test(a, b, s)


@pytest.mark.parametrize("name", ["gcm", "wgcm", "pcm"])
def test_testwise_deletion_outside_the_query_equals_the_complete_test(name):
    x = _nonlinear()
    wider = np.column_stack([x, np.random.default_rng(5).standard_normal(len(x))])
    options = {"seed": 3, **LINEAR[name], **({"n_splits": 1} if name == "pcm" else {})}
    complete = ci.make_indep_test(name, wider, **options)
    wider[7, 3] = np.nan
    assert ci.make_indep_test(name, wider, **options)(0, 1, [2]) == complete(0, 1, [2])


@pytest.mark.parametrize(("cls", "options"), release_cases(ALL_OPTIONS, {"wgcm-trees"}))
def test_p_values_ignore_the_units_and_signs(cls, options):
    x = _nonlinear()
    scaled = x * np.array([3.7, 0.013, 11.3]) + np.array([2.5, -1.0, 2.5])
    flipped = x * np.array([-1.0, 1.0, -1.0])
    test = cls(x, seed=3, **options)
    for other in (cls(scaled, seed=3, **options), cls(flipped, seed=3, **options)):
        for query in [(0, 1, [2]), (0, 2, [1]), (1, 2, [])]:
            assert other(*query) == pytest.approx(test(*query), abs=1e-9)


@pytest.mark.parametrize("name", LINEAR)
def test_a_constant_variable_is_independent(name):
    x = _chain(60)
    x[:, 3] = 2.0
    assert _test(name, x)(0, 3, [1]) == 1.0
    assert _test(name, x)(3, 2, []) == 1.0


def test_wgcm_detects_a_covariance_whose_sign_follows_the_conditioning_set():
    x = _sign_changing()
    assert cr.GCM(x, regression="linear")(0, 1, [2]) > 0.05
    assert cr.WGCM(x, regression="linear")(0, 1, [2]) < 0.01
    assert cr.WGCM(x, regression="linear", variant="fix")(0, 1, [2]) < 0.01


def test_pcm_detects_an_interaction_the_gcm_misses():
    rng = np.random.default_rng(0)
    n = 400
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = x * z + 0.5 * rng.standard_normal(n)  # E[y | x, z] depends on x, E[xy | z] = 0
    data = np.column_stack([x, y, z])
    assert cr.GCM(data, regression="linear")(0, 1, [2]) > 0.05
    assert cr.PCM(data, regression="forest", n_splits=1)(0, 1, [2]) < 0.01


def _sine_null(n: int, seed: int) -> np.ndarray:
    """``x`` and ``y`` independent given ``z``, ``x`` almost a copy of their shared conditional
    mean ``sin(2 pi z)``: the PCM paper's Section 6.1 null with one conditioning variable."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    mean = np.sin(2 * np.pi * z)
    return np.column_stack([mean + 0.1 * rng.standard_normal(n), mean + rng.standard_normal(n), z])


@pytest.mark.release
def test_pcm_keeps_its_level_where_x_copies_the_conditional_mean():
    rejections = sum(cr.PCM(_sine_null(200, s), seed=s)(0, 1, [2]) < 0.05 for s in range(60))
    assert rejections <= 9  # the 0.999 quantile of the binomial count at 5%


def test_pcm_detects_the_squared_effect_of_its_papers_first_setting():
    # Section 6.1, setting 1: y depends on x through 0.2 x**2 beyond sin(2 pi z1), with z ~ N7.
    rng = np.random.default_rng(0)
    z = rng.standard_normal((500, 7))
    mean = np.sin(2 * np.pi * z[:, 0])
    x = mean + rng.standard_normal(500)
    y = mean + 0.2 * x**2 + rng.standard_normal(500)
    test = cr.PCM(np.column_stack([x, y, z]), seed=0)
    rows, cols = test._layout(range(9))
    block, features = test._data[rows], [c for c in cols if c > 1]
    # The paper's direction, y the response, whichever variable the query's rule would pick.
    statistics = [test._split_statistic(block, s, 1, 0, features) for s in test._splits]
    assert ndtr(-np.mean(statistics)) < 0.05


def test_default_regressions_on_the_chain():
    x = _chain(150)
    for test in (cr.GCM(x), cr.WGCM(x, variant="fix")):
        assert test(0, 2, [1]) > 0.05
        assert test(0, 1, [2]) < 0.01


def test_early_stopping_keeps_ten_rounds_past_the_best():
    curve = np.array([5.0, 4.0, 3.0, *([3.5] * 20)])
    cut = cr._early_stopped(curve)
    assert np.isfinite(cut[:13]).all()
    assert np.isinf(cut[13:]).all()
    assert np.array_equal(cr._early_stopped(curve[:12]), curve[:12])


@pytest.fixture(params=["lean", "sklearn"])
def route(request, monkeypatch):
    """The boosting search by each of its routes: the lean loops over scikit-learn's compiled
    prediction, and the loops over scikit-learn's own fits that stand where it is missing."""
    if request.param == "lean":
        assert cr._lean_kernels() is not None, "the lean loops must run on this scikit-learn"
    else:
        monkeypatch.setattr(cr, "_lean_kernels", lambda: None)
    return request.param


@pytest.fixture
def fresh_kernels():
    """``_lean_kernels`` answers again, and answers again after the test."""
    cr._lean_kernels.cache_clear()
    yield
    cr._lean_kernels.cache_clear()


def test_mean_cv_error_equals_the_staged_curves(route, monkeypatch):
    """The folds grown together in steps give the error and rounds of each fold's full staged
    curve, averaged over the folds and stopped 10 rounds after its best."""
    monkeypatch.setattr(cr, "_MEAN_CV_ROUNDS", 60)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((60, 2))
    target = np.sin(2 * z[:, 0]) + 0.3 * rng.standard_normal(60)
    folds = (np.arange(60) % 10)[rng.permutation(60)]
    for rate, depth in [(0.5, 3), (0.1, 1)]:  # stops early; runs to the last round
        curves = []
        for fold in range(10):
            held = folds == fold
            model = cr._boosting(depth, 60, rate).fit(z[~held], target[~held])
            curves.append(
                [np.sqrt(np.mean((target[held] - p) ** 2)) for p in model.staged_predict(z[held])]
            )
        mean = cr._early_stopped(np.mean(curves, axis=0))
        best = int(np.argmin(mean))
        assert cr._mean_cv_error(z, target, folds, rate, depth) == (mean[best], best + 1)


def _boosting_case(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The features, target, and folds of one ``_fit_boosting`` call, as ``_fit`` builds them."""
    rng = np.random.default_rng(1)
    if kind == "smooth":  # depth 1 keeps improving to the last round, deeper trees stop early
        z = rng.standard_normal((80, 1))
        y = np.sin(2 * z[:, 0]) + 0.3 * rng.standard_normal(80)
    elif kind == "tied codes":  # three-level columns: every split threshold is shared by many rows
        z = rng.integers(0, 3, size=(80, 2)).astype(float)
        y = (z[:, 0] == 1) + rng.integers(0, 3, size=80)
    elif kind == "duplicated column":  # two features split alike, so every split ties
        z = np.repeat(rng.standard_normal((80, 1)), 2, axis=1)
        y = np.sin(2 * z[:, 0]) + 0.3 * rng.standard_normal(80)
    elif kind == "binary codes":  # two-level columns: the splits of the two features tie in gain
        z = rng.integers(0, 2, size=(80, 2)).astype(float)
        y = z[:, 0] * z[:, 1] + 0.3 * rng.standard_normal(80)
    elif kind == "constant fold":  # fold 3 holds both rows of the rare value, so its model sees 0
        z = rng.standard_normal((60, 2))
        y = np.zeros(60)
        y[[3, 13]] = 1.0
    else:  # the fewest rows a test is fitted on, and a dozen
        rows = {"four rows": 4, "twelve rows": 12}[kind]
        z = rng.standard_normal((rows, 2))
        y = np.sin(2 * z[:, 0]) + 0.3 * rng.standard_normal(rows)
    block = cr._canonical_columns(np.column_stack([y, z]))
    m = len(y)
    folds = np.arange(m) % min(cr._BOOST_FOLDS, m)
    return block[:, 1:], block[:, 0], folds if kind == "constant fold" else rng.permutation(folds)


def _staged_boosting(features, target, folds):
    """The earlier ``_fit_boosting``: every fold grown to the last round, its held-out curve from
    staged prediction cut by ``_early_stopped``. Returns each fold's curve and the predictor."""
    curves = {}
    error = np.zeros((cr._BOOST_ROUNDS, len(cr._BOOST_DEPTHS)))
    for j, depth in enumerate(cr._BOOST_DEPTHS):
        for fold in range(int(folds.max()) + 1):
            held = folds == fold
            model = cr._boosting(depth, cr._BOOST_ROUNDS).fit(features[~held], target[~held])
            curve = np.array(
                [
                    np.sqrt(np.mean((target[held] - p) ** 2))
                    for p in model.staged_predict(features[held])
                ]
            )
            curves[depth, fold] = cr._early_stopped(curve)
            error[:, j] += curves[depth, fold]
    j, rounds = divmod(int(np.argmin(error.T)), cr._BOOST_ROUNDS)
    return curves, cr._boosting(cr._BOOST_DEPTHS[j], rounds + 1).fit(features, target).predict


@pytest.mark.parametrize(
    "kind",
    [
        "smooth",
        "tied codes",
        "duplicated column",
        "binary codes",
        "constant fold",
        "four rows",
        "twelve rows",
    ],
)
def test_boosting_equals_the_staged_loop_bit_for_bit(kind, route, monkeypatch):
    """Each fold's curve, and the predictor fitted from them, equal the earlier loop's."""
    features, target, folds = _boosting_case(kind)
    curves, predict = _staged_boosting(features, target, folds)
    seen = {}
    fold_curve = cr._fold_curve

    def record(features, target, held, depth):
        seen[depth, int(folds[held][0])] = curve = fold_curve(features, target, held, depth)
        return curve

    monkeypatch.setattr(cr, "_fold_curve", record)
    fitted = cr._fit_boosting(features, target, folds)
    assert seen.keys() == curves.keys()
    assert all(seen[key].tobytes() == curves[key].tobytes() for key in curves)
    assert fitted(features).tobytes() == predict(features).tobytes()
    # The cases reach the regimes of the stop: cut early, run to the last round, and ties.
    early = [np.isinf(c).any() for c in curves.values()]
    full = [np.isfinite(c).all() for c in curves.values()]
    if kind in ("smooth", "tied codes"):
        assert any(early) and any(full)
    if kind == "tied codes":
        assert np.unique(features[:, 0]).size == 3
    if kind == "constant fold":  # a fold with no variation to learn repeats one error, then stops
        finite = curves[1, 3][np.isfinite(curves[1, 3])]
        assert finite.size == cr._BOOST_PATIENCE + 1 and np.ptp(finite) == 0


def test_a_fold_grows_only_through_its_stop(route, monkeypatch):
    features, target, folds = _boosting_case("smooth")
    grown = []
    if route == "lean":
        grow = cr._Boosted.grow

        def record_round(self):
            grown.append(grow(self))
            return grown[-1]

        monkeypatch.setattr(cr._Boosted, "grow", record_round)
    else:
        boosting = cr._boosting

        def record_fit(*args, **kwargs):
            grown.append(boosting(*args, **kwargs))
            return grown[-1]

        monkeypatch.setattr(cr, "_boosting", record_fit)
    curve = cr._fold_curve(features, target, folds == 0, cr._BOOST_DEPTHS[-1])
    stop = int(np.isfinite(curve).sum())
    assert 0 < stop < cr._BOOST_ROUNDS
    assert (len(grown) if route == "lean" else grown[0].estimators_.shape[0]) == stop


@pytest.mark.parametrize("depth", [1, 3, 6])
def test_lean_trees_and_predictions_equal_the_staged_fit(depth):
    """The lean trees are the fit's trees, and the compiled kernel adds each round's prediction as
    the fit's ``staged_predict`` does."""
    kernels = cr._lean_kernels()
    assert kernels is not None
    features, target, _ = _boosting_case("smooth")
    rounds = 20
    fit = cr._boosting(depth, rounds).fit(features, target)
    lean = cr._Boosted(features, target, depth, cr._BOOST_RATE, rounds)
    rows = np.ascontiguousarray(features, dtype=np.float32)
    prediction = np.full((len(target), 1), lean.init)
    for i, staged in enumerate(fit.staged_predict(features)):
        tree = lean.grow().tree_
        assert lean.trees[i, 0].criterion == fit.estimators_[i, 0].criterion
        expected = fit.estimators_[i, 0].tree_
        for name in ("feature", "threshold", "value"):
            assert getattr(tree, name).tobytes() == getattr(expected, name).tobytes()
        kernels.predict_stage(lean.trees, i, rows, lean.rate, prediction)
        assert prediction[:, 0].tobytes() == staged.tobytes()


@pytest.mark.parametrize("rows", [7, 130, 1000, 5001])
def test_the_lean_start_is_the_fits_mean(rows):
    rng = np.random.default_rng(rows)
    target = rng.standard_normal(rows) * 10.0 ** rng.integers(-3, 4, size=rows)
    features = rng.standard_normal((rows, 2))
    fit = cr._boosting(1, 1).fit(features, target)
    start = cr._Boosted(features, target, 1, cr._BOOST_RATE, 1).init
    assert np.float64(start).tobytes() == fit.init_.constant_.tobytes()


def test_resume_sums_the_rounds_as_the_finished_model_predicts():
    """A warm-start refit resumes from the compiled sum, which the finished model's ``predict``
    returns; on a platform that fuses the multiply and the add it differs from the NumPy sum."""
    kernels = cr._lean_kernels()
    assert kernels is not None
    features, target, _ = _boosting_case("smooth")
    fit = cr._boosting(3, 15).fit(features, target)
    lean = cr._Boosted(features, target, 3, cr._BOOST_RATE, 15)
    for _ in range(15):
        lean.grow()
    lean.resume(kernels)
    assert lean.raw[:, 0].tobytes() == fit.predict(features).tobytes()


def _mean_cv_case(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The features, target, and folds of one wGCM search of 60 rows."""
    rng = np.random.default_rng(3)
    if kind == "smooth":
        z = rng.standard_normal((60, 2))
        y = np.sin(2 * z[:, 0]) + 0.3 * rng.standard_normal(60)
    else:  # two binary features, where a split of either feature ties with the other's
        z = rng.integers(0, 2, size=(60, 2)).astype(float)
        y = z[:, 0] * z[:, 1] + 0.3 * rng.standard_normal(60)
    block = cr._canonical_columns(np.column_stack([y, z]))
    return block[:, 1:], block[:, 0], rng.permutation(np.arange(60) % 10)


@pytest.mark.parametrize("kind", ["smooth", "binary"])
def test_the_lean_mean_cv_error_equals_the_sklearn_loop(kind, monkeypatch):
    """At each learning rate, the lean search returns the error and rounds of the loop over
    scikit-learn's fits bit for bit, past the refits at rounds 10, 20, and 40."""
    monkeypatch.setattr(cr, "_MEAN_CV_ROUNDS", 80)
    kernels = cr._lean_kernels()
    assert kernels is not None
    features, target, folds = _mean_cv_case(kind)
    reached = []
    for rate in cr._MEAN_CV_RATES:
        for depth in (1, 3):
            lean = cr._lean_mean_cv_error(features, target, folds, rate, depth, kernels)
            error, rounds = cr._gbr_mean_cv_error(features, target, folds, rate, depth)
            assert np.float64(lean[0]).tobytes() == np.float64(error).tobytes()
            assert lean[1] == rounds
            reached.append(rounds + cr._BOOST_PATIENCE)
    assert max(reached) > 4 * cr._WARM_START_ROUNDS
    assert min(reached) < 2 * cr._WARM_START_ROUNDS


def test_the_lean_search_resumes_each_fold_every_ten_rounds(monkeypatch):
    """Where a warm-start refit resumes, the lean loop sums the grown rounds again: each fold, once
    at each multiple of 10 rounds the search reaches."""
    monkeypatch.setattr(cr, "_MEAN_CV_ROUNDS", 80)
    kernels = cr._lean_kernels()
    assert kernels is not None
    resumed: dict[int, list[int]] = {}
    resume = cr._Boosted.resume

    def record(self, kernels):
        resumed.setdefault(id(self), []).append(self.grown)
        resume(self, kernels)

    monkeypatch.setattr(cr._Boosted, "resume", record)
    features, target, folds = _mean_cv_case("smooth")
    _, rounds = cr._lean_mean_cv_error(features, target, folds, 0.1, 1, kernels)
    last = min(rounds - 1 + cr._BOOST_PATIENCE, cr._MEAN_CV_ROUNDS - 1)
    assert last >= 3 * cr._WARM_START_ROUNDS
    expected = list(range(cr._WARM_START_ROUNDS, last + 1, cr._WARM_START_ROUNDS))
    assert len(resumed) == 10
    assert all(grown == expected for grown in resumed.values())


def test_wgcm_boosting_selection_equals_the_sklearn_loop(monkeypatch):
    """The learning rate, depth, and rounds that the lean search picks, and the predictor fitted
    with them, equal the loop over scikit-learn's fits, on a case of tied splits."""
    monkeypatch.setattr(cr, "_MEAN_CV_ROUNDS", 60)
    features, target, folds = _mean_cv_case("binary")
    lean = cr._fit_boosting_mean_cv(features, target, folds)
    monkeypatch.setattr(cr, "_lean_kernels", lambda: None)
    expected = cr._fit_boosting_mean_cv(features, target, folds)
    assert lean(features).tobytes() == expected(features).tobytes()


def test_the_lean_loops_need_the_kernel_to_reproduce_the_fits(monkeypatch, fresh_kernels):
    assert cr._lean_kernels() is not None
    cr._lean_kernels.cache_clear()
    predict_stage = sklearn_kernels.predict_stage

    def shifted(trees, stage, features, scale, out):
        predict_stage(trees, stage, features, scale, out)
        out += 1e-12

    monkeypatch.setattr(sklearn_kernels, "predict_stage", shifted)
    assert cr._lean_kernels() is None


def test_the_lean_loops_are_not_used_without_the_kernel(monkeypatch, fresh_kernels):
    monkeypatch.delattr(sklearn_kernels, "predict_stages")
    assert cr._lean_kernels() is None


def test_gcm_boosting_p_value_equals_the_staged_loop(monkeypatch):
    data = _chain(80)
    p_values = [cr.GCM(data, seed=0)(0, 2, [1]), cr.GCM(data, seed=0)(0, 3, [1, 2])]

    def staged(features, target, folds):
        return _staged_boosting(features, target, folds)[1]

    monkeypatch.setattr(cr._RegressionTest, "_fit_boosting", staticmethod(staged))
    expected = [cr.GCM(data, seed=0)(0, 2, [1]), cr.GCM(data, seed=0)(0, 3, [1, 2])]
    assert [float(p).hex() for p in p_values] == [float(p).hex() for p in expected]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("where", ["features", "target"])
def test_boosting_rejects_a_non_finite_input(where, bad):
    features, target, folds = _boosting_case("twelve rows")
    (features if where == "features" else target)[3] = bad
    with pytest.raises(ValueError, match="finite"):
        cr._fit_boosting(features, target, folds)


def _linear_null(n: int, seed: int) -> np.ndarray:
    """The wGCM paper's one-dimensional null at its linear grid point: ``x`` and ``y`` are
    ``0.3 z`` plus independent noise of scale 0.3."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    return np.column_stack(
        [0.3 * z + 0.3 * rng.standard_normal(n), 0.3 * z + 0.3 * rng.standard_normal(n), z]
    )


@pytest.mark.parametrize("variant", ["est", "fix"])
def test_wgcm_holds_its_level_on_a_null_design(variant):
    """Of 1,000 draws of a true independence, the share rejected at 0.05 stays within three
    binomial standard deviations of 5% (29 to 71), with regressions that fit the design."""
    rejected = sum(
        cr.WGCM(_linear_null(200, seed), variant=variant, regression="linear", seed=seed)(0, 1, [2])
        <= 0.05
        for seed in range(1000)
    )
    assert 29 <= rejected <= 71


def test_wgcm_est_weights_by_the_fit(monkeypatch):
    weights = []
    answer = cr._estimated_answer

    def record(products, weight):
        weights.append(weight)
        return answer(products, weight)

    monkeypatch.setattr(cr, "_estimated_answer", record)
    cr.WGCM(_sign_changing(), regression="linear")(0, 1, [2])
    (weight,) = weights
    assert np.unique(weight).size == weight.size  # the fitted values, not their signs
    assert (weight < 0).any() and (weight > 0).any()


def test_pcm_variance_adds_the_shift_to_the_positive_part():
    fitted = np.array([-0.5, 0.2, -0.005, -1.0])
    variance = cr._conditional_variance(fitted, 0.3)
    assert np.array_equal(variance, [0.3, 0.5, 0.3, 0.3])
    assert np.array_equal(cr._conditional_variance(fitted, 0.0), [0.01, 0.2, 0.01, 0.01])


def test_pcm_split_statistic_divides_its_standard_deviation_by_the_test_rows(monkeypatch):
    products = []
    normalized_mean = cr._normalized_mean

    def record(values):
        products.append(values)
        return normalized_mean(values)

    monkeypatch.setattr(cr, "_normalized_mean", record)
    test = cr.PCM(_chain(40), regression="linear", n_splits=1)
    rows, _ = test._layout([0, 1, 2])
    split = test._splits[0]
    statistic = test._split_statistic(test._data[rows], split, 1, 0, [2])
    (values,) = products
    m = split[1].size
    assert values.size == m
    assert statistic == pytest.approx(np.sqrt(m) * values.mean() / values.std(ddof=0))


def test_variance_shift_brings_the_mean_ratio_to_one():
    squares = np.array([1.0, 4.0, 9.0])
    fitted = np.array([0.0, 1.0, 1.0])
    shift = cr._variance_shift(squares, fitted)
    assert np.mean(squares / (fitted + shift)) == pytest.approx(1.0)
    assert cr._variance_shift(squares, squares * 2) == 0.0


@pytest.mark.parametrize(
    ("cls", "options", "match"),
    [
        (cr.GCM, {"regression": "spline"}, "regression"),
        (cr.WGCM, {"variant": "both"}, "variant"),
        (cr.PCM, {"n_splits": 0}, "n_splits"),
        (cr.PCM, {"n_splits": 1.9}, "n_splits"),
        (cr.PCM, {"n_splits": True}, "n_splits"),
    ],
)
def test_invalid_settings_raise(cls, options, match):
    with pytest.raises(ValueError, match=match):
        cls(_chain(20), **options)
