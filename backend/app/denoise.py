"""Six denoising configurations. All functions take/return (C, N) arrays."""
import numpy as np
import pywt
from scipy.signal import firwin, oaconvolve
from .config import FS, FIR_TAPS, FIR_BAND, WAVELET, LEVEL, ENERGY_RETAIN

_H = firwin(FIR_TAPS, FIR_BAND, pass_zero=False, window="hamming", fs=FS)


def fir(x):
    # linear-phase FIR; 'same' mode compensates the 64-sample group delay
    return oaconvolve(x, _H[None, :], mode="same", axes=-1)


def _soft(d, t):
    return np.sign(d) * np.maximum(np.abs(d) - t, 0.0)


def _universal(d, sigma, n):
    return _soft(d, sigma * np.sqrt(2 * np.log(n)))


def _bayes(d, sigma, n):
    sx = np.sqrt(np.maximum(d.var(axis=-1, keepdims=True) - sigma ** 2, 0.0))
    t = np.where(sx > 0, sigma ** 2 / np.maximum(sx, 1e-12), np.abs(d).max(axis=-1, keepdims=True))
    return _soft(d, t)


def _energy(d, sigma, n):
    """Keep the largest coefficients holding ENERGY_RETAIN of sub-band energy; zero the rest."""
    mag = np.abs(d)
    s = -np.sort(-mag, axis=-1)
    ce = np.cumsum(s ** 2, axis=-1)
    k = (ce < ENERGY_RETAIN * ce[:, -1:]).sum(axis=-1, keepdims=True)
    thr = np.take_along_axis(s, k, axis=-1)
    return np.where(mag >= thr, d, 0.0)


def _wavelet(x, rule):
    n = x.shape[-1]
    c = pywt.wavedec(x, WAVELET, level=LEVEL, mode="symmetric", axis=-1)
    sigma = np.median(np.abs(c[-1]), axis=-1, keepdims=True) / 0.6745  # noise est. from level-1 details
    out = [c[0]] + [rule(d, sigma, n) for d in c[1:]]
    return pywt.waverec(out, WAVELET, mode="symmetric", axis=-1)[..., :n]


DENOISERS = {
    "none": lambda x: x,
    "fir": fir,
    "wav_universal": lambda x: _wavelet(x, _universal),
    "wav_bayes": lambda x: _wavelet(x, _bayes),
    "fir_bayes": lambda x: _wavelet(fir(x), _bayes),
    "fir_energy": lambda x: _wavelet(fir(x), _energy),
}


def apply(key, x):
    return DENOISERS[key](x)
