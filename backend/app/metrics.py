"""Quality metrics and EEG biomarkers."""
import numpy as np
from scipy.signal import welch
from scipy.integrate import trapezoid
from .config import FS, BANDS, CHANNELS

EPS = 1e-12


def band_powers(x):
    """x: (C, N) -> dict band -> (C,) absolute power via Welch PSD."""
    f, p = welch(x, fs=FS, nperseg=512, axis=-1)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f <= hi)
        out[b] = trapezoid(p[..., m], f[m], axis=-1)
    return out


def evaluate(r, y, noisy):
    err = y - r
    sr = np.sum(r ** 2)
    snr_in = 10 * np.log10(sr / (np.sum((noisy - r) ** 2) + EPS))
    snr_out = 10 * np.log10(sr / (np.sum(err ** 2) + EPS))
    res = {
        "snr_in": snr_in,
        "snr_out": snr_out,
        "snr_improvement": snr_out - snr_in,
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "prd": float(100 * np.sqrt(np.sum(err ** 2) / sr)),
        "pearson": float(np.corrcoef(r.ravel(), y.ravel())[0, 1]),
    }
    pr, py = band_powers(r), band_powers(y)
    for b in BANDS:
        res["band_" + b] = float(np.mean(100 * np.abs(py[b] - pr[b]) / (pr[b] + EPS)))
    return {k: float(v) for k, v in res.items()}


def biomarkers(x):
    """FAA = ln(alpha F4) - ln(alpha F3);  TBR = theta/beta at Fz."""
    bp = band_powers(x)
    i3, i4, iz = CHANNELS.index("F3"), CHANNELS.index("F4"), CHANNELS.index("Fz")
    faa = np.log(bp["alpha"][i4] + EPS) - np.log(bp["alpha"][i3] + EPS)
    f, p = welch(x[iz], fs=FS, nperseg=512)
    th = trapezoid(p[(f >= 4) & (f <= 8)], f[(f >= 4) & (f <= 8)])
    be = trapezoid(p[(f >= 13) & (f <= 30)], f[(f >= 13) & (f <= 30)])
    return {"faa": float(faa), "tbr": float(th / (be + EPS))}
