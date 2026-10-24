"""Synthetic 19-channel EEG with simulated MDD vs HC biomarker differences."""
import numpy as np
from .config import CHANNELS, FS, N_SAMPLES

_F = np.fft.rfftfreq(N_SAMPLES, 1 / FS)
_POST = {"P3", "P4", "O1", "O2", "Pz", "T5", "T6"}
_FRONT = {"Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"}
# (centre Hz, sigma Hz, height): delta, theta, alpha, beta
_PEAKS = ((2.0, 1.0, 1.4), (6.0, 1.5, 1.0), (10.0, 1.2, 3.0), (20.0, 4.0, 0.7))


def _amp(g):
    """Amplitude spectrum (C, F) for per-channel band gains g (C, 4)."""
    f = np.maximum(_F, 0.5)
    peaks = sum(g[:, i:i + 1] * h * np.exp(-((f - c) ** 2) / (2 * s ** 2))
                for i, (c, s, h) in enumerate(_PEAKS))
    taper = (_F >= 0.5) / (1 + (_F / 45.0) ** 8)
    return (f ** -0.6) * (1 + peaks) * taper


_REF = _amp(np.ones((1, 4)))
_SCALE = 15.0 / np.sqrt(2 * np.sum(_REF ** 2) / N_SAMPLES ** 2)  # ~15 uV RMS baseline


def _gains(group, rng):
    g = np.ones((len(CHANNELS), 4))
    for i, ch in enumerate(CHANNELS):
        if ch in _POST:
            g[i, 2] *= 1.4
        if ch in _FRONT:
            g[i, 2] *= 0.7
    ix = CHANNELS.index
    if group.upper() == "MDD":
        g[ix("F3"), 2] *= 1.30   # more left-frontal alpha (less left activity) -> lower FAA
        g[ix("F4"), 2] *= 0.90
        g[:, 1] *= 1.35          # theta up
        g[:, 3] *= 0.80          # beta down  -> higher TBR
    else:
        g[ix("F4"), 2] *= 1.12
    return g * rng.lognormal(0, 0.12, size=(1, 4))  # inter-subject variability


def generate_epochs(group, n_epochs, seed=0):
    """Return (n_epochs, 19, 2560) in microvolts."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_epochs, len(CHANNELS), N_SAMPLES))
    for e in range(n_epochs):
        amp = _amp(_gains(group, rng)) * _SCALE
        z = rng.standard_normal(amp.shape) + 1j * rng.standard_normal(amp.shape)
        out[e] = np.fft.irfft(amp * z / np.sqrt(2), n=N_SAMPLES, axis=-1)
    return out
