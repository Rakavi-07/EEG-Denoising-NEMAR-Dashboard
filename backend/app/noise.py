"""Synthetic noise engine implementing Equation 1:  RMS(n) = RMS(r) * 10^(-SNR/20)."""
import numpy as np
from scipy.signal import butter, sosfiltfilt
from .config import FS, NOISE_WEIGHTS, LINE_HZ, EMG_BAND

_EMG_SOS = butter(4, EMG_BAND, btype="band", fs=FS, output="sos")


def rms(x):
    return np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True))


def _unit(x):
    return x / (rms(x) + 1e-12)


def white(shape, rng):
    return _unit(rng.standard_normal(shape))


def line(shape, rng):
    t = np.arange(shape[-1]) / FS
    ph = rng.uniform(0, 2 * np.pi, size=shape[:-1] + (1,))
    return _unit(np.sin(2 * np.pi * LINE_HZ * t + ph))


def emg(shape, rng):
    return _unit(sosfiltfilt(_EMG_SOS, rng.standard_normal(shape), axis=-1))


def add_noise(r, snr_db, rng):
    """r: (C, N) clean. Returns (noisy, noise). SNR is enforced per channel."""
    w = NOISE_WEIGHTS
    mix = w["white"] * white(r.shape, rng) + w["line"] * line(r.shape, rng) + w["emg"] * emg(r.shape, rng)
    n = _unit(mix) * rms(r) * 10 ** (-snr_db / 20)
    return r + n, n
