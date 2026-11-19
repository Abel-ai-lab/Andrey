"""The Tuebingen lane: the requested units, the summary of partial runs, and the reference setup.

Checked in bench-env without the authors' code: the driver's manifest, the summary's reading of run
directories, the digest the reference records carry, and the library export the R setup scripts
print.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import tuebingen_pairs
import tuebingen_summary

from andrey_bench.adapters import grci_strobl, heci_xu, qccd_bqcd

BENCH = Path(__file__).resolve().parents[1]


def _record(pair: str, method: str, decision: str = "x->y") -> dict:
    return {
        "pair": pair,
        "method": method,
        "weight": 1.0,
        "status": "ok",
        "wall_s": 0.1,
        "package_version": "0",
        "params": {},
        "cap_wall_s": 1800.0,
        "cap_mem_mb": None,
        "decision": decision,
        "correct": decision == "x->y",
        "cpu": "test",
    }


def _run(path: Path, pairs: list[str], solutions: list[str], records: list[dict]) -> Path:
    path.mkdir()
    tuebingen_pairs.update_manifest(path / "manifest.json", dict.fromkeys(pairs, 1.0), solutions)
    if records:
        (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_a_rerun_into_one_directory_requests_only_what_it_asked(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    tuebingen_pairs.update_manifest(path, {"pair0001": 1.0}, ["anm"])
    tuebingen_pairs.update_manifest(path, {"pair0002": 0.5}, ["qccd"])
    manifest = json.loads(path.read_text())
    assert manifest["pairs"] == {"pair0001": 1.0, "pair0002": 0.5}
    assert manifest["units"] == [["pair0001", "anm"], ["pair0002", "qccd"]]


def test_pooled_runs_count_no_unit_that_was_never_asked(tmp_path) -> None:
    runs = [
        _run(tmp_path / "a", ["pair0001"], ["anm"], [_record("pair0001", "anm")]),
        _run(tmp_path / "b", ["pair0002"], ["qccd"], [_record("pair0002", "qccd")]),
    ]
    summary = tuebingen_summary.summarize(
        tuebingen_summary.read_records(runs), *tuebingen_summary.read_manifest(runs)
    )
    for method in ("anm", "qccd"):
        assert summary["solutions"][method]["statuses"] == {"ok": 1}
        assert summary["solutions"][method]["requested"] == 1
    assert "| anm | 1/1 |" in tuebingen_summary.markdown(summary, [])


def test_a_run_with_no_fit_finished_yet_reads_as_every_unit_missing(tmp_path) -> None:
    run = _run(tmp_path / "a", ["pair0001", "pair0002"], ["anm", "qccd"], [])
    out = tmp_path / "out"
    assert tuebingen_summary.main([str(run), "--out", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text())
    for method in ("anm", "qccd"):
        assert summary["solutions"][method]["statuses"] == {"missing": 2}
        assert summary["solutions"][method]["accuracy"] is None


def test_a_directory_with_neither_file_is_refused(tmp_path) -> None:
    (tmp_path / "typo").mkdir()
    with pytest.raises(SystemExit, match="neither"):
        tuebingen_summary.read_records([tmp_path / "typo"])


@pytest.mark.parametrize("module", [qccd_bqcd, heci_xu, grci_strobl])
def test_reference_records_name_the_files_as_read(module, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(module.CODE_ENV, str(tmp_path))
    assert module.source_sha256() == "unknown"
    for name in module.REQUIRED_FILES:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text("code\n")
    before = module.source_sha256()
    (tmp_path / module.REQUIRED_FILES[-1]).write_text("changed code\n")
    assert module.source_sha256() not in (before, "unknown")


@pytest.mark.parametrize("script", ["start_qccd.sh", "start_grci.sh"])
@pytest.mark.parametrize(
    ("before", "after"), [("/lib/old", "/lib/new:/lib/old"), (None, "/lib/new")]
)
def test_the_printed_library_export_keeps_the_path_already_set(script, before, after) -> None:
    [fmt] = re.findall(r"^printf '(export R_LIBS_USER=[^']*)' ", (BENCH / script).read_text(), re.M)
    env = {"PATH": "/usr/bin:/bin"} | ({"R_LIBS_USER": before} if before else {})
    shell = f'eval "$(printf \'{fmt}\' /lib/new)"; printf %s "$R_LIBS_USER"'
    run = subprocess.run(["bash", "-c", shell], env=env, capture_output=True, text=True, check=True)
    assert run.stdout == after


def test_records_from_another_runtime_are_not_pooled(tmp_path) -> None:
    a, b = _record("pair0001", "heci"), _record("pair0002", "heci")
    a["runtime"], b["runtime"] = {"numpy": "2.3.0"}, {"numpy": "2.4.0"}
    runs = [_run(tmp_path / "a", ["pair0001"], ["heci"], [a])]
    runs.append(_run(tmp_path / "b", ["pair0002"], ["heci"], [b]))
    with pytest.raises(SystemExit, match="runtime"):
        tuebingen_summary.read_records(runs)


def test_a_resumption_with_another_identity_stops_before_any_fit(tmp_path) -> None:
    record = _record("pair0001", "anm") | {"runtime": {"numpy": "2.4.0"}}
    run = _run(tmp_path / "a", ["pair0001"], ["anm"], [record])
    same = {k: record[k] for k in ("package_version", "params", "cap_wall_s", "cap_mem_mb")}
    same["runtime"] = record["runtime"]
    tuebingen_pairs.check_resumption(run / "records.jsonl", {"anm": same})
    tuebingen_pairs.check_resumption(run / "records.jsonl", {"qccd": same | {"cap_wall_s": 60.0}})
    with pytest.raises(SystemExit, match="new directory"):
        tuebingen_pairs.check_resumption(
            run / "records.jsonl", {"anm": same | {"cap_wall_s": 60.0}}
        )


def test_a_selection_with_no_two_column_pair_is_refused(monkeypatch, tmp_path) -> None:
    import andrey.data

    pair = SimpleNamespace(name="pair0052", multivariate=True)
    monkeypatch.setattr(andrey.data, "load_dataset", lambda *args, **kwargs: [pair])
    with pytest.raises(SystemExit):
        tuebingen_pairs.main(["--out", str(tmp_path / "run"), "--pairs", "52"])
    assert not (tmp_path / "run").exists()
