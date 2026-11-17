"""Shared unit-test fixtures."""

from __future__ import annotations

import pytest


def tuebingen_archive(meta: str | None = None) -> bytes:
    """A synthetic archive in the Tuebingen collection's layout, with its layout's quirks.

    Six pairs: two from one source (weights 0.5, tab- and space-separated, one line with trailing
    blanks); a pair whose cause is the second column; a pair with a third column the metadata
    does not use; a multivariate pair with weight 0, whose cause block is the second; and one with
    weight 0.5, whose cause block is the first. The descriptions of the odd pairs are UTF-8 and
    those of the even pairs Latin-1, as in the real archive. ``meta`` replaces the metadata file.
    """
    import io
    import zipfile

    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, 40)
    cols = {
        1: np.column_stack([x, x**3 + rng.normal(0, 0.3, 40)]),
        2: np.column_stack([x, np.exp(x) + rng.normal(0, 0.3, 40)]),
        3: np.column_stack([np.sin(x) + rng.normal(0, 0.1, 40), x]),
        4: np.column_stack([x, 2 * x + rng.normal(0, 0.1, 40), np.full(40, np.nan)]),
        5: rng.normal(size=(30, 4)),
        6: rng.normal(size=(30, 4)),
    }
    layout = {
        1: "1 1 2 2 0.5",
        2: "1 1 2 2 0.5",
        3: "2 2 1 1 1",
        4: "1 1 2 2 0.3333",
        5: "3 4 1 2 0",
        6: "1 3 4 4 0.5",
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        archive.writestr("README", "synthetic fixture\n")
        archive.writestr(
            "pairmeta.txt", meta or "".join(f"{k:04d} {v}\n" for k, v in layout.items())
        )
        for k, table in cols.items():
            sep = "\t" if k % 2 else "  "
            lines = [sep.join(f"{v:.6g}" for v in row) + ("  " if k == 2 else "") for row in table]
            archive.writestr(f"pair{k:04d}.txt", "\n".join(lines) + "\n")
            text = f"Information for pair{k:04d}\n10\xb0C\n"
            archive.writestr(f"pair{k:04d}_des.txt", text.encode("utf-8" if k % 2 else "latin-1"))
    return out.getvalue()


def serve_datasets_offline(monkeypatch, cache):
    """Serve dataset downloads from fixtures/, into the empty directory ``cache``, with no network.

    Sachs is the 853-row file, under its own SHA-256; Sachs's nine conditions are the published
    experimental file, and ASIA, CHILD, and INSURANCE the published BIFs, under the recorded
    SHA-256s; the Tuebingen collection is a synthetic archive of six pairs (``tuebingen_archive``)
    under its own SHA-256. Returns the list of URLs fetched, so a test can tell a download from a
    cache hit.
    """
    import dataclasses
    import gzip
    import hashlib
    import urllib.error
    from pathlib import Path

    from andrey.data import real

    fixtures = Path(__file__).parent / "fixtures"
    sachs = gzip.decompress((fixtures / "sachs-observational.txt.gz").read_bytes())
    served = {"https://example.invalid/sachs.txt": sachs}
    for name in ("asia", "child", "insurance"):
        served |= dict.fromkeys(
            real.DATASETS[name].sources, (fixtures / f"{name}.bif.gz").read_bytes()
        )
    pairs = tuebingen_archive()
    served |= dict.fromkeys(real.DATASETS["tuebingen"].sources, pairs)
    monkeypatch.setitem(
        real.DATASETS,
        "tuebingen",
        dataclasses.replace(real.DATASETS["tuebingen"], sha256=hashlib.sha256(pairs).hexdigest()),
    )
    fetched: list[str] = []

    def fetch(url):
        fetched.append(url)
        if url not in served:
            raise urllib.error.URLError(f"no fixture for {url}")
        return served[url]

    remote = dataclasses.replace(
        real.DATASETS["sachs"],
        sources=("https://example.invalid/sachs.txt",),
        sha256=hashlib.sha256(sachs).hexdigest(),
    )
    monkeypatch.setitem(real.DATASETS, "sachs", remote)
    served |= dict.fromkeys(
        real.SACHS_CONDITIONS_FILE.sources,
        gzip.decompress((fixtures / "sachs-experimental.txt.gz").read_bytes()),
    )
    monkeypatch.setattr(real, "fetch", fetch)
    monkeypatch.setenv("ANDREY_DATA_DIR", str(cache))
    return fetched


@pytest.fixture
def make_tuebingen_archive():
    """The builder of the synthetic Tuebingen archive, to pass it a metadata file of its own."""
    return tuebingen_archive


@pytest.fixture
def offline_datasets(monkeypatch, tmp_path):
    """Dataset downloads served from fixtures/ for one test; see ``serve_datasets_offline``."""
    return serve_datasets_offline(monkeypatch, tmp_path / "cache")


@pytest.fixture(scope="module")
def offline_datasets_module(tmp_path_factory):
    """Dataset downloads served from fixtures/ for a whole module, its module fixtures included."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield serve_datasets_offline(monkeypatch, tmp_path_factory.mktemp("cache"))
