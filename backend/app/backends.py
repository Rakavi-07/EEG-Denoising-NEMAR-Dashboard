"""Compute-backend simulation: CPU (measured) vs GPU end-to-end vs GPU device-resident.

GPU timings are derived from the measured CPU time using the configured speedups.
Swap in real CuPy kernels here to measure on hardware.
"""
from .config import SPEEDUP_E2E, SPEEDUP_DEVICE

try:  # informational only
    import cupy  # noqa: F401
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def simulate(cpu_ms):
    cpu_ms = max(float(cpu_ms), 1e-3)
    e2e, dev = cpu_ms / SPEEDUP_E2E, cpu_ms / SPEEDUP_DEVICE
    return {
        "cpu_ms": cpu_ms,
        "gpu_e2e_ms": e2e,
        "gpu_device_ms": dev,
        "transfer_overhead_ms": e2e - dev,
        "speedup_e2e": SPEEDUP_E2E,
        "speedup_device": SPEEDUP_DEVICE,
    }
