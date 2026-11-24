"""The R libraries of the GCM, wGCM, and PCM agreement check: their own variables.

`start_regression_ci_r.sh` runs here against stand-ins for R, curl, and git, so the checks cover its
library handling, not the build. The R lane's library, `ANDREY_BENCH_R_LIBS`, stays an empty
directory throughout: the start script must not write to it, and the agreement script must not take
`R_LIBS` from it.
"""

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest
import regression_ci_agreement as agreement

BENCH = Path(__file__).resolve().parents[1]
EXPORT = re.compile(r"^export (ANDREY_BENCH_[A-Z0-9_]+)=(.*)$")
STANDINS = {
    # The libraries each Rscript call was given, one line per call.
    "Rscript": 'printf "%s\\n" "$R_LIBS" >> "$STANDIN_LOG"',
    "curl": 'while [ $# -gt 0 ]; do [ "$1" = -o ] && : > "$2"; shift; done',
    "sha256sum": "cat > /dev/null",
    "git": f'if [ "$1" = rev-parse ]; then exec {shutil.which("git")} "$@"; fi',
}


@pytest.fixture(scope="module")
def start(tmp_path_factory):
    """Run the start script once; its checkout, the R lane's library, its output, and the libraries
    each Rscript call was given."""
    root = tmp_path_factory.mktemp("start")
    repo, standins, lane_lib = root / "repo", root / "standins", root / "lane-lib"
    (repo / "benchmarks" / ".pcm-code" / ".git").mkdir(parents=True)
    standins.mkdir()
    lane_lib.mkdir()
    for name, body in STANDINS.items():
        (standins / name).write_text(f"#!/bin/bash\n{body}\n")
        (standins / name).chmod(0o755)
    subprocess.run([shutil.which("git"), "init", "--quiet", str(repo)], check=True)
    env = {
        "PATH": f"{standins}:/usr/bin:/bin",
        "HOME": str(root),
        "STANDIN_LOG": str(root / "rscript.log"),
        "ANDREY_BENCH_R_MODULE": " ",  # no module to load
        "ANDREY_BENCH_R_LIBS": str(lane_lib),
    }
    run = subprocess.run(
        [shutil.which("bash"), str(BENCH / "start_regression_ci_r.sh")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    printed = [line for line in run.stdout.splitlines() if line.startswith("export ")]
    return {
        "repo": repo.resolve(),
        "lane_lib": lane_lib,
        "printed": printed,
        "exports": {m[1]: shlex.split(m[2])[0] for m in map(EXPORT.match, printed) if m},
        "rscript_libs": (root / "rscript.log").read_text().splitlines(),
    }


def test_the_start_script_prints_the_two_libraries_and_the_pcm_code(start) -> None:
    assert all(EXPORT.match(line) for line in start["printed"])
    repo = start["repo"] / "benchmarks"
    assert {k: Path(v).resolve() for k, v in start["exports"].items()} == {
        "ANDREY_BENCH_R_LIBS_GCM": repo / ".r-lib-regression-ci",
        "ANDREY_BENCH_R_LIBS_WGCM": repo / ".r-lib-regression-ci-wgcm",
        "ANDREY_BENCH_PCM_CODE": repo / ".pcm-code",
    }


def test_the_start_script_leaves_the_lane_library_alone(start) -> None:
    exports = start["exports"]
    assert list(start["lane_lib"].iterdir()) == []
    assert start["rscript_libs"] == [exports["ANDREY_BENCH_R_LIBS_GCM"]] + [
        exports["ANDREY_BENCH_R_LIBS_WGCM"]
    ]


def test_run_tests_applies_the_printed_libraries_beside_the_lane_library(start) -> None:
    text = (BENCH / "run_tests.sh").read_text()
    function = re.search(r"^apply_exports\(\) \{.*?^\}$", text, re.M | re.S)[0]
    names = [*start["exports"], "ANDREY_BENCH_R_LIBS"]
    lane = shlex.quote(str(start["lane_lib"]))
    shell = (
        f"{function}\nexport ANDREY_BENCH_R_LIBS={lane}\n"
        f"apply_exports <<< {shlex.quote(chr(10).join(start['printed']))}\n"
        + "".join(f'echo "{name}=${name}"\n' for name in names)
    )
    run = subprocess.run(
        [shutil.which("bash"), "-c", shell], capture_output=True, text=True, check=True
    )
    applied = dict(line.split("=", 1) for line in run.stdout.splitlines())
    assert applied == {**start["exports"], "ANDREY_BENCH_R_LIBS": str(start["lane_lib"])}


@pytest.fixture
def reference_env(start, monkeypatch, tmp_path):
    """The agreement script's environment as the printed exports set it, beside the lane's library,
    and the `R_LIBS` each reference run was given (`None`: the environment's own)."""
    monkeypatch.setenv("ANDREY_BENCH_R_LIBS", str(start["lane_lib"]))
    for name, value in start["exports"].items():
        monkeypatch.setenv(name, value)
    given = []

    def fake_run(command, check, env):
        given.append(None if env is None else env["R_LIBS"])
        folder, mode = Path(command[-2]), command[-1]
        (folder / f"reference_{mode}.csv").write_text("p\n0.5\n")

    monkeypatch.setattr(agreement.subprocess, "run", fake_run)
    return tmp_path, given


@pytest.mark.parametrize(
    ("mode", "variable"),
    [
        ("gcm", "ANDREY_BENCH_R_LIBS_GCM"),
        *((mode, "ANDREY_BENCH_R_LIBS_GCM") for mode, *_ in agreement.PCM_MODES),
        ("wgcm_est", "ANDREY_BENCH_R_LIBS_WGCM"),
        ("wgcm_fix", "ANDREY_BENCH_R_LIBS_WGCM"),
    ],
)
def test_a_reference_run_takes_the_library_of_its_test(
    reference_env, start, mode, variable
) -> None:
    folder, given = reference_env
    agreement.run_reference(folder, mode, ["c0"], [1])
    assert given == [start["exports"][variable]]


def test_the_lane_library_alone_does_not_set_the_reference_library(
    reference_env, monkeypatch
) -> None:
    folder, given = reference_env
    for name in ("ANDREY_BENCH_R_LIBS_GCM", "ANDREY_BENCH_R_LIBS_WGCM"):
        monkeypatch.delenv(name)
    for mode in ("gcm", "wgcm_est", "pcm_lm"):
        agreement.run_reference(folder, mode, ["c0"], [1])
    assert given == [None, None, None]
