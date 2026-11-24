#!/usr/bin/env python3
"""Build and run the tests of the SUNDIALS ExtSTS time stepper of the gyrokinetic app.

Steps:
  1. Build the gyrokinetic library (and the libraries it depends on), the unit
     tests and the regression executables. `make <regression target>` alone does
     not rebuild the libraries after a source change, hence `make gyrokinetic` first.
  2. Run the unit tests ctest_nvector_array and ctest_array_reduce.
  3. Run rt_gk_sheath_extsts_1x2v_p1 (ExtSTS) and rt_gk_sheath_1x2v_p1 (the same
     problem with SSP-RK3), each in its own directory.
  4. Compare the .gkyl output:
     - ExtSTS against SSP-RK3. This is a sanity bound that catches gross errors
       (a missing or doubled term), not a convergence test: the two methods have
       different orders, and the runs end at slightly different times because the
       last step is not clipped to t_end. The report includes the difference that
       the end-time offset alone would cause, estimated from the average rate of
       change of the SSP-RK3 solution.
     - ExtSTS against an earlier ExtSTS run (--ref), to detect any change in results.
     - ExtSTS on N MPI ranks against the serial ExtSTS run (--mpi N).
  5. Optionally (--valgrind), run the unit tests and a short ExtSTS run under Valgrind.

Every invocation writes to a new directory <run-dir>/<date>-<time>-<commit>, which
records how the results were produced (provenance.txt, source.diff, config.mak).
Pass that directory to --ref in later invocations to use it as the reference.
The executables find the Gkeyll libraries through a relative rpath, so each run
directory gets a symlink to the build directory.

Examples:
  ci/test_sundials_extsts.py
  ci/test_sundials_extsts.py --mpi 2 --ref ~/gkyl-runs/extsts/20261123-210000-14a4a8d16
  ci/test_sundials_extsts.py --skip-build --valgrind

Exit status: 0 if every check passed, 1 otherwise.
"""

import argparse
import datetime
import hashlib
import itertools
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UNIT_TESTS = ["ctest_nvector_array", "ctest_array_reduce"]
EXTSTS_RT = "rt_gk_sheath_extsts_1x2v_p1"
BASELINE_RT = "rt_gk_sheath_1x2v_p1"
VALGRIND_STEPS = 20


class Report:
    """Collects the outcome of each check."""

    def __init__(self):
        self.failures = []
        self.num_checks = 0

    def check(self, ok, name, detail=""):
        self.num_checks += 1
        print(
            f"[{' OK ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""),
            flush=True,
        )
        if not ok:
            self.failures.append(name)
        return ok


def run_logged(cmd, log_path, cwd):
    """Run cmd in cwd, writing stdout and stderr to log_path. Return the exit status."""
    with open(log_path, "w") as log:
        log.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        log.flush()
        return subprocess.run(
            cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT
        ).returncode


def log_tail(log_path, num_lines=25):
    lines = Path(log_path).read_text(errors="replace").splitlines()
    return "\n".join("    " + l for l in lines[-num_lines:])


def read_config():
    """Return the key=value pairs of config.mak."""
    config = {}
    path = REPO / "config.mak"
    if not path.exists():
        sys.exit(f"{path} not found: run ./configure first.")
    for line in path.read_text().splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            config[key.strip()] = value.strip()
    return config


def git_output(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout


def write_provenance(out_dir, config, build_dir):
    """Record the source, configuration and tools used to produce the results."""
    head = git_output("rev-parse", "HEAD").strip()
    diff = git_output("diff", "HEAD", "--binary")
    untracked = git_output("ls-files", "--others", "--exclude-standard").split()
    (out_dir / "source.diff").write_text(diff)
    shutil.copy(REPO / "config.mak", out_dir / "config.mak")

    compiler = config.get("CC", "cc")
    try:
        cc_version = subprocess.run(
            [compiler, "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0]
    except (OSError, IndexError):
        cc_version = "unknown"
    sundials_version = "unknown"
    sundials_config = (
        Path(config.get("CONF_SUNDIALS_INC_DIR", "")) / "sundials/sundials_config.h"
    )
    if sundials_config.exists():
        match = re.search(
            r'#define SUNDIALS_VERSION "([^"]+)"', sundials_config.read_text()
        )
        sundials_version = match.group(1) if match else "unknown"

    lines = [
        f"command: {' '.join(sys.argv)}",
        f"date: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"host: {platform.node()} ({platform.platform()})",
        f"repository: {REPO}",
        f"commit: {head}",
        f"uncommitted changes (vs. commit, in source.diff): sha256 {hashlib.sha256(diff.encode()).hexdigest()}",
        f"untracked files not in source.diff: {len(untracked)}",
        f"build directory: {REPO / build_dir}",
        f"compiler: {compiler}: {cc_version}",
        f"SUNDIALS: {sundials_version}",
    ]
    (out_dir / "provenance.txt").write_text("\n".join(lines) + "\n")


def msgpack_unpack(buf, pos):
    """Decode the msgpack object at buf[pos:]. Return (object, position after it)."""
    t = buf[pos]
    pos += 1
    if t <= 0x7F:
        return t, pos
    if t >= 0xE0:
        return t - 0x100, pos
    if 0x80 <= t <= 0x8F or t in (0xDE, 0xDF):
        if t <= 0x8F:
            n = t & 0x0F
        else:
            fmt = ">H" if t == 0xDE else ">I"
            (n,) = struct.unpack_from(fmt, buf, pos)
            pos += struct.calcsize(fmt)
        obj = {}
        for _ in range(n):
            key, pos = msgpack_unpack(buf, pos)
            obj[key], pos = msgpack_unpack(buf, pos)
        return obj, pos
    if 0x90 <= t <= 0x9F or t in (0xDC, 0xDD):
        if t <= 0x9F:
            n = t & 0x0F
        else:
            fmt = ">H" if t == 0xDC else ">I"
            (n,) = struct.unpack_from(fmt, buf, pos)
            pos += struct.calcsize(fmt)
        obj = []
        for _ in range(n):
            item, pos = msgpack_unpack(buf, pos)
            obj.append(item)
        return obj, pos
    if 0xA0 <= t <= 0xBF or t in (0xD9, 0xDA, 0xDB, 0xC4, 0xC5, 0xC6):
        if t <= 0xBF:
            n = t & 0x1F
        else:
            fmt = {
                0xD9: ">B",
                0xDA: ">H",
                0xDB: ">I",
                0xC4: ">B",
                0xC5: ">H",
                0xC6: ">I",
            }[t]
            (n,) = struct.unpack_from(fmt, buf, pos)
            pos += struct.calcsize(fmt)
        raw = bytes(buf[pos : pos + n])
        return (raw if t in (0xC4, 0xC5, 0xC6) else raw.decode()), pos + n
    scalars = {
        0xCA: ">f",
        0xCB: ">d",
        0xCC: ">B",
        0xCD: ">H",
        0xCE: ">I",
        0xCF: ">Q",
        0xD0: ">b",
        0xD1: ">h",
        0xD2: ">i",
        0xD3: ">q",
    }
    if t in scalars:
        (value,) = struct.unpack_from(scalars[t], buf, pos)
        return value, pos + struct.calcsize(scalars[t])
    if t in (0xC0, 0xC2, 0xC3):
        return {0xC0: None, 0xC2: False, 0xC3: True}[t], pos
    raise ValueError(f"unsupported msgpack type 0x{t:02x}")


def read_gkyl(path):
    """
    Read a .gkyl grid file (one range, or the multi-range files written by MPI runs).

    Return a dict with the metadata, the cell counts, the number of components and
    the values in row-major cell order, or None for other file types (e.g. the
    dynamic vectors that hold integrated diagnostics).
    """
    buf = Path(path).read_bytes()
    if buf[:5] != b"gkyl0":
        raise ValueError(f"{path}: not a gkyl file")
    version, file_type, meta_size = struct.unpack_from("<QQQ", buf, 5)
    if version != 1:
        raise ValueError(f"{path}: unsupported version {version}")
    if file_type not in (1, 3):
        return None
    pos = 29
    meta = msgpack_unpack(buf, pos)[0] if meta_size > 0 else {}
    pos += meta_size
    real_type, ndim = struct.unpack_from("<QQ", buf, pos)
    pos += 16
    if real_type != 2:
        raise ValueError(f"{path}: only double data is supported")
    cells = struct.unpack_from(f"<{ndim}Q", buf, pos)
    pos += 8 * ndim + 16 * ndim  # Cells, then lower and upper coordinates.
    esznc, tot_cells = struct.unpack_from("<QQ", buf, pos)
    pos += 16
    ncomp = esznc // 8

    if file_type == 1:
        values = struct.unpack_from(f"<{tot_cells * ncomp}d", buf, pos)
    else:
        (num_ranges,) = struct.unpack_from("<Q", buf, pos)
        pos += 8
        blocks = []
        for _ in range(num_ranges):
            lo = struct.unpack_from(f"<{ndim}Q", buf, pos)
            up = struct.unpack_from(f"<{ndim}Q", buf, pos + 8 * ndim)
            (size,) = struct.unpack_from("<Q", buf, pos + 16 * ndim)
            pos += 16 * ndim + 8
            blocks.append((lo, up, struct.unpack_from(f"<{size * ncomp}d", buf, pos)))
            pos += size * esznc
        glo = [min(b[0][d] for b in blocks) for d in range(ndim)]
        strides = [1] * ndim
        for d in range(ndim - 2, -1, -1):
            strides[d] = strides[d + 1] * cells[d + 1]
        merged = [None] * (tot_cells * ncomp)
        for lo, up, data in blocks:
            idx_ranges = [range(lo[d], up[d] + 1) for d in range(ndim)]
            for k, idx in enumerate(itertools.product(*idx_ranges)):
                lin = sum((idx[d] - glo[d]) * strides[d] for d in range(ndim))
                merged[lin * ncomp : (lin + 1) * ncomp] = data[
                    k * ncomp : (k + 1) * ncomp
                ]
        if any(v is None for v in merged):
            raise ValueError(f"{path}: the ranges do not cover the grid")
        values = merged

    return {"meta": meta, "cells": tuple(cells), "ncomp": ncomp, "values": values}


def rel_diff(a, b):
    """Max abs difference of all DG coefficients, relative to the max magnitude of b."""
    scale = max(abs(v) for v in b)
    diff = max(abs(x - y) for x, y in zip(a, b))
    return diff / scale if scale > 0 else diff


def grid_files(run_dir, prefix):
    """Map the suffix of each .gkyl file of a run (name after '<prefix>-') to its path."""
    return {
        p.name[len(prefix) + 1 : -5]: p
        for p in sorted(run_dir.glob(f"{prefix}-*.gkyl"))
    }


def compare_runs(
    report, name, dir_a, prefix_a, dir_b, prefix_b, tol, verbose, offset_estimate=False
):
    """
    Compare every grid file the two runs have in common, checking that each differs
    by at most tol. With offset_estimate, also estimate the difference caused by the
    runs ending at different times, from the frame-0 and frame-1 files of run b.
    """
    files_a, files_b = grid_files(dir_a, prefix_a), grid_files(dir_b, prefix_b)
    rows = []
    for suffix in sorted(set(files_a) & set(files_b)):
        a, b = read_gkyl(files_a[suffix]), read_gkyl(files_b[suffix])
        if a is None or b is None:
            continue
        if (a["cells"], a["ncomp"]) != (b["cells"], b["ncomp"]):
            rows.append((suffix, float("inf"), None, "shapes differ"))
            continue
        diff = rel_diff(a["values"], b["values"])
        estimate = None
        t_a, t_b = a["meta"].get("time"), b["meta"].get("time")
        note = ""
        if t_a is not None and t_b is not None and t_a != t_b:
            note = f"t = {t_a:.6e} vs {t_b:.6e}"
            frame0 = files_b.get(re.sub(r"_(\d+)$", "_0", suffix))
            if offset_estimate and frame0 is not None and frame0 != files_b[suffix]:
                b0 = read_gkyl(frame0)
                t_0 = b0["meta"].get("time")
                if t_0 is not None and t_b != t_0:
                    estimate = (
                        rel_diff(b0["values"], b["values"])
                        * abs(t_a - t_b)
                        / abs(t_b - t_0)
                    )
        rows.append((suffix, diff, estimate, note))

    if not rows:
        return report.check(False, name, "no common grid files")
    if verbose:
        for suffix, diff, estimate, note in rows:
            if diff != 0.0:
                est = (
                    f"  end-time offset estimate {estimate:.2e}"
                    if estimate is not None
                    else ""
                )
                print(f"       {suffix:40s} {diff:.2e}{est}  {note}")
        print(f"       ({sum(r[1] == 0.0 for r in rows)} identical files not listed)")
    worst = max(rows, key=lambda r: r[1])
    detail = (
        f"{len(rows)} files, max rel. diff {worst[1]:.2e} ({worst[0]}), tol {tol:.1e}"
    )
    if worst[2] is not None:
        detail += f"; end-time offset alone would give ~{worst[2]:.2e}"
    if worst[3]:
        detail += f"; {worst[3]}"
    return report.check(worst[1] <= tol, name, detail)


def check_extsts_log(report, name, log_path):
    """Check the ExtSTS stepper ran (its summary is printed when the app is released)."""
    text = Path(log_path).read_text(errors="replace")
    steps = re.search(
        r"^ExtSTS: (\d+) steps.*?(\d+) steps exceeded the explicit CFL", text, re.M
    )
    mem = re.search(r"^ExtSTS: at most .*$", text, re.M)
    if not steps:
        return report.check(
            False, name, "no ExtSTS summary in the log: was the ExtSTS stepper used?"
        )
    detail = steps.group(0).removeprefix("ExtSTS: ")
    if mem:
        detail += "\n       " + mem.group(0).removeprefix("ExtSTS: ")
    return report.check(True, name, detail)


def run_regression(
    report,
    out_dir,
    build_dir,
    name,
    label,
    extra_args=(),
    launcher=(),
    expect_complete=True,
):
    """
    Run a regression executable in out_dir/label. Return the run directory, or None if it
    failed (or, with expect_complete, did not reach t_end).
    """
    run_dir = out_dir / label
    run_dir.mkdir()
    (run_dir / build_dir).symlink_to(REPO / build_dir)
    exe = Path(build_dir) / "gyrokinetic" / "creg" / name
    status = run_logged(
        [*launcher, f"./{exe}", *extra_args], run_dir / "run.log", run_dir
    )
    ok = status == 0
    if expect_complete:
        ok = ok and "Completed 100%" in (run_dir / "run.log").read_text(
            errors="replace"
        )
    if not report.check(
        ok, f"run {label}", f"exit status {status}, log {run_dir / 'run.log'}"
    ):
        print(log_tail(run_dir / "run.log"))
        return None
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path.home() / "gkyl-runs" / "extsts",
        help="directory for the results of every invocation (default: %(default)s)",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="parallel make jobs (default: half the processors)",
    )
    parser.add_argument("--skip-build", action="store_true", help="do not run make")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="use the CUDA build (cuda-build/) and run the regressions with -g",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        help="earlier run directory to compare the ExtSTS results to",
    )
    parser.add_argument(
        "--mpi",
        type=int,
        metavar="N",
        help="also run ExtSTS on N MPI ranks (z split N ways; N must divide Nz = 8)",
    )
    parser.add_argument(
        "--mpiexec",
        default=os.environ.get("MPIEXEC", "mpirun"),
        help="MPI launcher, may include options (default: $MPIEXEC or mpirun)",
    )
    parser.add_argument(
        "--valgrind",
        action="store_true",
        help=f"also run the unit tests and {VALGRIND_STEPS} ExtSTS steps under Valgrind",
    )
    parser.add_argument(
        "--baseline-tol",
        type=float,
        default=5e-3,
        help="max rel. difference between ExtSTS and SSP-RK3 (default: %(default)s)",
    )
    parser.add_argument(
        "--ref-tol",
        type=float,
        default=1e-12,
        help="max rel. difference from the --ref run (default: %(default)s)",
    )
    parser.add_argument(
        "--mpi-tol",
        type=float,
        default=1e-10,
        help="max rel. difference between MPI and serial ExtSTS (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="list every file that differs"
    )
    args = parser.parse_args()

    config = read_config()
    if config.get("USE_SUNDIALS") != "1":
        sys.exit(
            "Gkeyll is not configured with SUNDIALS: run ./configure --use-sundials=yes ..."
        )
    if args.mpi and config.get("USE_MPI") != "1":
        sys.exit(
            "--mpi needs Gkeyll configured with MPI (./configure --use-mpi=yes ...)"
        )
    if args.valgrind and shutil.which("valgrind") is None:
        sys.exit("--valgrind: valgrind not found on PATH")
    ref_dir = None
    if args.ref:
        ref_dir = args.ref.expanduser().resolve()
        ref_dir = ref_dir / "extsts" if (ref_dir / "extsts").is_dir() else ref_dir
        if not any(ref_dir.glob(f"{EXTSTS_RT}-*.gkyl")):
            sys.exit(f"--ref: no {EXTSTS_RT} output in {ref_dir}")

    build_dir = "cuda-build" if args.gpu else "build"
    gpu_args = ["-g"] if args.gpu else []
    report = Report()

    head = git_output("rev-parse", "--short", "HEAD").strip()
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.run_dir.expanduser().resolve() / f"{stamp}-{head}"
    out_dir.mkdir(parents=True)  # Fails rather than overwrite an earlier invocation.
    write_provenance(out_dir, config, build_dir)
    print(f"Results in {out_dir}", flush=True)

    # 1. Build.
    if not args.skip_build:
        targets = [f"{build_dir}/core/unit/{t}" for t in UNIT_TESTS]
        targets += [
            f"{build_dir}/gyrokinetic/creg/{t}" for t in (EXTSTS_RT, BASELINE_RT)
        ]
        log = out_dir / "build.log"
        status = run_logged(["make", f"-j{args.nproc}", "gyrokinetic"], log, REPO)
        if status == 0:
            status = run_logged(
                ["make", f"-j{args.nproc}", *targets], out_dir / "build_tests.log", REPO
            )
            log = out_dir / "build_tests.log"
        if not report.check(status == 0, "build", f"log {log}"):
            print(log_tail(log))
            return 1

    # 2. Unit tests.
    unit_dir = out_dir / "unit"
    unit_dir.mkdir()
    (unit_dir / build_dir).symlink_to(REPO / build_dir)
    for test in UNIT_TESTS:
        log = unit_dir / f"{test}.log"
        status = run_logged([f"./{build_dir}/core/unit/{test}"], log, unit_dir)
        if not report.check(status == 0, f"unit {test}", f"log {log}"):
            print(log_tail(log))

    # 3. Regression runs.
    extsts_dir = run_regression(
        report, out_dir, build_dir, EXTSTS_RT, "extsts", gpu_args
    )
    baseline_dir = run_regression(
        report, out_dir, build_dir, BASELINE_RT, "ssprk3", gpu_args
    )
    if extsts_dir:
        check_extsts_log(report, "ExtSTS stepper used", extsts_dir / "run.log")

    # 4. Comparisons.
    if extsts_dir and baseline_dir:
        compare_runs(
            report,
            "ExtSTS vs SSP-RK3",
            extsts_dir,
            EXTSTS_RT,
            baseline_dir,
            BASELINE_RT,
            args.baseline_tol,
            args.verbose,
            offset_estimate=True,
        )
    if extsts_dir and ref_dir:
        compare_runs(
            report,
            f"ExtSTS vs reference {ref_dir}",
            extsts_dir,
            EXTSTS_RT,
            ref_dir,
            EXTSTS_RT,
            args.ref_tol,
            args.verbose,
        )
    if args.mpi:
        mpi_args = [*gpu_args, "-M", "-c", str(args.mpi)]
        launcher = [*args.mpiexec.split(), "-np", str(args.mpi)]
        mpi_dir = run_regression(
            report,
            out_dir,
            build_dir,
            EXTSTS_RT,
            f"extsts_mpi{args.mpi}",
            mpi_args,
            launcher,
        )
        if mpi_dir:
            check_extsts_log(
                report, f"ExtSTS stepper used (MPI {args.mpi})", mpi_dir / "run.log"
            )
            if extsts_dir:
                compare_runs(
                    report,
                    f"ExtSTS MPI {args.mpi} vs serial",
                    mpi_dir,
                    EXTSTS_RT,
                    extsts_dir,
                    EXTSTS_RT,
                    args.mpi_tol,
                    args.verbose,
                )

    # 5. Memory checks.
    if args.valgrind:
        valgrind = [
            "valgrind",
            "--leak-check=full",
            "--errors-for-leak-kinds=definite",
            "--error-exitcode=1",
        ]
        vg_dir = out_dir / "valgrind"
        vg_dir.mkdir()
        (vg_dir / build_dir).symlink_to(REPO / build_dir)
        for test in UNIT_TESTS:
            log = vg_dir / f"{test}.log"
            status = run_logged(
                [*valgrind, f"./{build_dir}/core/unit/{test}"], log, vg_dir
            )
            if not report.check(status == 0, f"valgrind {test}", f"log {log}"):
                print(log_tail(log))
        run_regression(
            report,
            vg_dir,
            build_dir,
            EXTSTS_RT,
            "extsts",
            [*gpu_args, f"-s{VALGRIND_STEPS}"],
            valgrind,
            expect_complete=False,
        )

    passed = report.num_checks - len(report.failures)
    print(f"\n{passed} of {report.num_checks} checks passed. Results in {out_dir}")
    for name in report.failures:
        print(f"  failed: {name}")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    sys.exit(main())
