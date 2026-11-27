"""Enforce memory caps on the worker's process tree.

* :data:`POLL`: sample the tree's memory and kill the group on a breach, recording an OOM. The
  cap reads the tree's Pss where Linux reports it (:func:`tree_memory_mb`), else its RSS.
* :data:`SLURM`: use ``srun --mem=`` with ``ConstrainRAMSpace=yes`` for kernel enforcement;
  :func:`stderr_reports_oom` checks the step's reported reason.
"""

from __future__ import annotations

import ctypes
import functools
import math
import os
import platform
import re
import signal
import threading
from collections.abc import Callable
from typing import Mapping

#: Route names. ``NONE`` is what an uncapped run records.
POLL = "poll"
SLURM = "slurm"
NONE = "none"
ROUTES: tuple[str, ...] = (POLL, SLURM, NONE)

#: Sampling period for the polling route.
DEFAULT_POLL_INTERVAL_S = 0.05

_MB = 1024.0 * 1024.0


def job_mem_mb(env: Mapping[str, str] | None = None) -> float | None:
    """Return the Slurm job's memory allocation in MB, or ``None`` if unavailable."""
    e = os.environ if env is None else env
    if per_node := _float_or_none(e.get("SLURM_MEM_PER_NODE")):
        return per_node
    per_cpu = _float_or_none(e.get("SLURM_MEM_PER_CPU"))
    cpus = _float_or_none(e.get("SLURM_CPUS_ON_NODE"))
    return per_cpu * cpus if per_cpu and cpus else None


def _float_or_none(value: str | None) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def resolve_route(
    cap_mem_mb: float | None,
    requested: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Return ``NONE`` without a cap; otherwise default to ``POLL``.

    Reject a cap paired with ``NONE``.
    """
    if requested is not None and requested not in ROUTES:
        raise ValueError(f"unknown memory-cap route {requested!r}; expected one of {ROUTES}")
    if cap_mem_mb is None:
        return NONE
    if requested == NONE:
        raise ValueError(
            f"cap_mem_mb={cap_mem_mb} with route {NONE!r} records a cap nothing enforces; "
            f"drop the cap, or route it through {POLL!r} or {SLURM!r}"
        )
    route = requested or POLL
    if route == SLURM:
        # Each fit is an srun step asking for --mem=<cap>. A step wanting more than the job owns is
        # not schedulable, and Slurm queues it rather than refusing it: the run holds the node with
        # no step, no record, and no error until the wall runs out. Only raise when the job's memory
        # is actually readable — off-scheduler this is silent, not blocked.
        owned = job_mem_mb(env)
        if owned is not None and cap_mem_mb > owned:
            raise ValueError(
                f"cap_mem_mb={cap_mem_mb:.0f} exceeds the {owned:.0f} MB this job owns, and route "
                f"{SLURM!r} runs each fit as 'srun --mem={int(math.ceil(cap_mem_mb))}M'. Slurm "
                "would block that step instead of failing it. Raise the job's --mem above the cap, "
                f"or route the cap through {POLL!r}."
            )
    return route


def srun_prefix(cap_mem_mb: float) -> list[str]:
    """Return the ``srun`` prefix for a Slurm memory cap."""
    return ["srun", "--ntasks=1", f"--mem={int(math.ceil(cap_mem_mb))}M"]


#: Slurm has worded the OOM message several ways across versions, so match the stable fragments
#: rather than one release's sentence.
_OOM_STDERR = re.compile(
    r"oom[-_ ]?kill|out[- ]of[- ]memory|exceeded (?:step|job) memory limit",
    re.IGNORECASE,
)


def stderr_reports_oom(text: str | None) -> bool:
    """Return whether Slurm stderr indicates an OOM kill."""
    return bool(text) and bool(_OOM_STDERR.search(text))


#: ``kcmp`` syscall numbers by machine, and its ``KCMP_VM`` type.
_KCMP_SYSCALL = {"x86_64": 312, "aarch64": 272}
_KCMP_VM = 1


@functools.cache
def _syscall() -> Callable[..., int] | None:
    """The C library's ``syscall``, or ``None`` where it cannot be loaded."""
    try:
        return ctypes.CDLL(None, use_errno=True).syscall
    except (OSError, AttributeError):
        return None


def _parent(pid: int) -> int | None:
    """The parent PID of ``pid`` from ``/proc/<pid>/stat``, or ``None``."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            # The comm field may contain spaces and parentheses; after it, index 1 is field 4.
            return int(fh.read().rsplit(") ", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def shares_memory(pid: int, other: int) -> bool:
    """Whether ``pid`` and ``other`` run in one address space (Linux ``kcmp``); ``False`` when the
    kernel cannot tell (no ``kcmp``, or another machine)."""
    number = _KCMP_SYSCALL.get(platform.machine())
    call = _syscall()
    if number is None or call is None:
        return False
    return call(number, pid, other, _KCMP_VM, 0, 0) == 0


def shares_parent_memory(pid: int) -> bool:
    """Whether ``pid`` runs in its parent's address space (:func:`shares_memory`), as a child
    started by ``vfork`` does until it calls exec.

    Until then its ``/proc`` files report the parent's memory, so a tree total would count that
    memory twice.
    """
    parent = _parent(pid)
    return parent is not None and shares_memory(pid, parent)


def pss_kb(pid: int) -> int | None:
    """Proportional set size (KiB) of ``pid`` from ``/proc/<pid>/smaps_rollup``: its resident
    pages, each shared page divided among the processes that map it. ``None`` where it cannot be
    read (another OS, an older kernel, or a process that has exited).

    The kernel walks the process's page tables for it, holding the process's memory map, which
    stalls the process's own mapping and unmapping of large arrays: read it rarely during a timed
    fit.
    """
    try:
        with open(f"/proc/{pid}/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("Pss:"):
                    return int(line.split()[1])
    except (OSError, IndexError, ValueError):
        return None
    return None


def tree_memory_mb(pid: int, *, pss: bool = True) -> tuple[float, float | None]:
    """Return the summed RSS and Pss (:func:`pss_kb`) in MB of a process and its descendants.

    RSS counts a page that processes share once per process; Pss counts it once in all. Pss is
    ``None`` without ``pss`` or when the root's cannot be read, and a descendant that exits
    mid-sample adds none. A descendant that runs in its parent's address space
    (:func:`shares_parent_memory`) is left out of both.
    """
    import psutil  # the benchmarks' own dependency; importing the harness does not need it

    try:
        root = psutil.Process(pid)
        procs = [root, *root.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0.0, None
    rss = 0
    total: int | None = None
    for proc in procs:
        if proc is not root and shares_parent_memory(proc.pid):
            continue
        try:
            rss += proc.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        kb = pss_kb(proc.pid) if pss else None
        if proc is root:
            total = kb
        elif total is not None and kb is not None:
            total += kb
    return rss / _MB, None if total is None else total * 1024 / _MB


def tree_rss_mb(pid: int) -> float:
    """Return summed RSS in MB for a process and its descendants (:func:`tree_memory_mb`)."""
    return tree_memory_mb(pid, pss=False)[0]


def _kill_tree(pid: int) -> None:
    """Kill the process group if it still exists.

    The child must have its own session (``setsid``) to avoid killing the caller's group.
    """
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


class TreeMemoryPoller(threading.Thread):
    """Kill the process group when the sampled tree memory exceeds the cap.

    Each sample reads the tree's RSS sum. Only a sum over the cap reads the tree's Pss as well,
    which decides where Linux reports it: Pss is at most the RSS sum, so a tree under the cap by
    RSS is under it by Pss, and the slow Pss read (:func:`pss_kb`) stays out of a fit under the cap.
    ``breached`` records a cap violation; ``peak_mb`` (RSS) may underestimate the true peak, and
    ``peak_pss_mb`` is the largest Pss read, ``None`` while the RSS sum stays under the cap.
    """

    def __init__(
        self,
        pid: int,
        cap_mem_mb: float,
        *,
        interval: float = DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        super().__init__(daemon=True)
        self.pid = pid
        self.cap_mem_mb = float(cap_mem_mb)
        self.peak_mb = 0.0
        self.peak_pss_mb: float | None = None
        self.breached = False
        self._interval = interval
        # Not ``_stop`` — that name shadows ``threading.Thread._stop``, which ``join`` calls.
        self._stop_evt = threading.Event()

    def sample(self) -> float:
        """Sample the tree's memory and enforce the cap; return the sample the cap reads."""
        total, _ = tree_memory_mb(self.pid, pss=False)
        self.peak_mb = max(self.peak_mb, total)
        if total > self.cap_mem_mb:
            _, pss = tree_memory_mb(self.pid)
            if pss is not None:
                self.peak_pss_mb = max(self.peak_pss_mb or 0.0, pss)
                total = pss
        if not self.breached and total > self.cap_mem_mb:
            self.breached = True
            _kill_tree(self.pid)
        return total

    def run(self) -> None:
        while not self._stop_evt.is_set():
            self.sample()
            self._stop_evt.wait(self._interval)

    def stop(self) -> None:
        """Stop polling and join a running thread, including after an interrupted start."""
        self._stop_evt.set()
        if self.is_alive():
            self.join(timeout=2.0)


__all__ = [
    "POLL",
    "SLURM",
    "NONE",
    "ROUTES",
    "DEFAULT_POLL_INTERVAL_S",
    "TreeMemoryPoller",
    "resolve_route",
    "srun_prefix",
    "stderr_reports_oom",
    "pss_kb",
    "shares_memory",
    "shares_parent_memory",
    "tree_memory_mb",
    "tree_rss_mb",
]
