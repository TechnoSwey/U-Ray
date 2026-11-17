"""Shared helpers for reproducible U-Ray versus NumPy benchmarks."""

import math
import os
import platform
import random
import statistics
import time
from importlib import metadata


_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_numpy_threads():
    """Pin NumPy's BLAS libraries to the requested thread count.

    Set URAY_BENCH_NUMPY_THREADS before starting the script to override the
    single-thread default. This makes the default comparison comparable to
    U-Ray's current single-threaded native CPU kernels.
    """
    try:
        threads = int(os.environ.get("URAY_BENCH_NUMPY_THREADS", "1"))
    except ValueError as error:
        raise ValueError("URAY_BENCH_NUMPY_THREADS must be a positive integer") from error
    if threads <= 0:
        raise ValueError("URAY_BENCH_NUMPY_THREADS must be a positive integer")
    for name in _THREAD_ENV_VARS:
        os.environ[name] = str(threads)
    return threads


def run_interleaved(backends, run, iterations, warmups=0, seed=2026,
                    keep_results=False):
    """Warm up and time backend pairs, randomizing which backend runs first."""
    backends = tuple(backends)
    if len(backends) < 2 or len(set(backends)) != len(backends):
        raise ValueError("at least two distinct backends are required")
    if iterations <= 0 or warmups < 0:
        raise ValueError("iterations must be > 0 and warmups must be >= 0")

    measured = {backend: [] for backend in backends}
    last = {backend: None for backend in backends}
    results = {backend: [] for backend in backends}
    rng = random.Random(seed)

    for sweep in range(warmups + iterations):
        order = list(backends)
        rng.shuffle(order)
        for backend in order:
            started = time.perf_counter()
            result = run(backend)
            elapsed = time.perf_counter() - started
            if sweep >= warmups:
                measured[backend].append(elapsed)
                last[backend] = result
                if keep_results:
                    results[backend].append(result)

    summaries = {}
    for backend in backends:
        samples = measured[backend]
        median = statistics.median(samples)
        deviations = [abs(sample - median) for sample in samples]
        summaries[backend] = {
            "samples": samples,
            "median": median,
            "mad": statistics.median(deviations),
            "last": last[backend],
            "results": results[backend],
        }
    return summaries


def prepare_backends(backends, select_backend, seed=2026):
    """Load each backend before timing, in a seeded randomized order."""
    order = list(backends)
    random.Random(seed).shuffle(order)
    elapsed = {}
    for backend in order:
        started = time.perf_counter()
        select_backend(backend)
        elapsed[backend] = time.perf_counter() - started
    return elapsed


def assert_nested_close(expected, actual, *, atol=1e-8, rtol=1e-7, path="$"):
    """Compare full nested numeric outputs and report the first mismatch."""
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(expected, actual, abs_tol=atol, rel_tol=rtol):
            raise AssertionError(
                f"{path}: {expected!r} != {actual!r} "
                f"(atol={atol}, rtol={rtol})"
            )
        return
    if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
        if len(expected) != len(actual):
            raise AssertionError(
                f"{path}: length {len(expected)} != {len(actual)}"
            )
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_nested_close(
                left, right, atol=atol, rtol=rtol, path=f"{path}[{index}]"
            )
        return
    if expected != actual:
        raise AssertionError(f"{path}: {expected!r} != {actual!r}")


def print_environment(*, numpy_threads, seed, iterations, warmups):
    """Print the machine and protocol needed to interpret a timing result."""
    cpu = platform.processor() or platform.machine() or "unknown"
    try:
        numpy_version = metadata.version("numpy")
    except metadata.PackageNotFoundError:
        numpy_version = "not installed"
    print(
        f"Environment: {platform.system()} {platform.release()}, "
        f"Python {platform.python_version()}, CPU={cpu}, "
        f"logical_cpus={os.cpu_count() or 'unknown'}, NumPy={numpy_version}"
    )
    print(
        f"Protocol: NumPy threads={numpy_threads}, randomized paired order "
        f"(seed={seed}), warmups={warmups}, measured iterations={iterations}; "
        "backend initialization is excluded; reported time is the median, "
        "spread is MAD."
    )
