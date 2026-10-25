"""Orchestration: simulation, benchmark and biomarker runs."""
import time
import numpy as np
from scipy.signal import welch
from .config import CONFIGS, CHANNELS, FS, BANDS
from . import eeg_sim, noise, denoise, metrics, backends, nemar


def get_epochs(group, n, seed, source):
    warn = None
    if source == "nemar":
        try:
            return nemar.load_epochs(group, n, seed), None
        except Exception as e:  # fall back gracefully
            warn = "NEMAR load failed (%s); using synthetic data." % e
    return eeg_sim.generate_epochs(group, n, seed), warn


def _r(a, d=2):
    return np.round(a, d).tolist()


def run_simulation(group, snr_db, channel, seed, source):
    clean, warn = get_epochs(group, 1, seed, source)
    r = clean[0]
    rng = np.random.default_rng(seed + 1)
    noisy, _ = noise.add_noise(r, snr_db, rng)
    ch = CHANNELS.index(channel)

    def psd(x):
        f, p = welch(x[ch], fs=FS, nperseg=512)
        m = f <= 60
        return f[m], 10 * np.log10(p[m] + 1e-12)

    f, p_clean = psd(r)
    out = {"t": _r(np.arange(r.shape[-1]) / FS, 3), "clean": _r(r[ch]), "noisy": _r(noisy[ch]),
           "outputs": {}, "metrics": {}, "psd": {"freq": _r(f, 2), "clean": _r(p_clean, 1),
                                                  "noisy": _r(psd(noisy)[1], 1), "outputs": {}},
           "warning": warn}
    for c in CONFIGS:
        y = denoise.apply(c["key"], noisy)
        out["outputs"][c["id"]] = _r(y[ch])
        out["psd"]["outputs"][c["id"]] = _r(psd(y)[1], 1)
        out["metrics"][c["id"]] = metrics.evaluate(r, y, noisy)
    return out


def run_benchmark(snr_levels, n_epochs, groups, seed, source):
    acc, times, warns = {}, {c["id"]: [] for c in CONFIGS}, set()
    for gi, g in enumerate(groups):
        clean, w = get_epochs(g, n_epochs, seed + gi * 1000, source)
        if w:
            warns.add(w)
        for ei in range(clean.shape[0]):
            r = clean[ei]
            for si, snr in enumerate(snr_levels):
                rng = np.random.default_rng(seed * 7919 + gi * 131 + ei * 17 + si)
                noisy, _ = noise.add_noise(r, snr, rng)
                for c in CONFIGS:
                    t0 = time.perf_counter()
                    y = denoise.apply(c["key"], noisy)
                    times[c["id"]].append((time.perf_counter() - t0) * 1e3)
                    acc.setdefault((snr, c["id"]), []).append(metrics.evaluate(r, y, noisy))
    rows = []
    for (snr, cid), lst in acc.items():
        m = {k: float(np.mean([d[k] for d in lst])) for k in lst[0]}
        rows.append({"snr_db": snr, "config_id": cid,
                     "name": next(c["name"] for c in CONFIGS if c["id"] == cid),
                     "snr_out": m["snr_out"], "snr_improvement": m["snr_improvement"],
                     "rmse": m["rmse"], "prd": m["prd"], "pearson": m["pearson"],
                     "band_errors": {b: m["band_" + b] for b in BANDS}})
    timing = [dict(config_id=c["id"], name=c["name"], **backends.simulate(np.median(times[c["id"]])))
              for c in CONFIGS]
    return {"rows": rows, "timing": timing, "warning": "; ".join(warns) or None,
            "meta": {"n_epochs_per_group": n_epochs, "groups": groups, "snr_levels": snr_levels}}


def _cohen_d(a, b):
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2) + 1e-12
    return float((np.mean(a) - np.mean(b)) / sp)


def run_biomarkers(snr_db, n_epochs, seed, source):
    labels = [{"id": 0, "name": "Clean reference", "key": None}] + CONFIGS
    vals = {l["id"]: {"MDD": {"faa": [], "tbr": []}, "HC": {"faa": [], "tbr": []}} for l in labels}
    warns = set()
    for gi, g in enumerate(("MDD", "HC")):
        clean, w = get_epochs(g, n_epochs, seed + gi * 1000, source)
        if w:
            warns.add(w)
        for ei in range(clean.shape[0]):
            r = clean[ei]
            noisy, _ = noise.add_noise(r, snr_db, np.random.default_rng(seed * 31 + gi * 977 + ei))
            for l in labels:
                y = r if l["id"] == 0 else denoise.apply(l["key"], noisy)
                b = metrics.biomarkers(y)
                vals[l["id"]][g]["faa"].append(b["faa"])
                vals[l["id"]][g]["tbr"].append(b["tbr"])
    rows = []
    for l in labels:
        v = vals[l["id"]]
        row = {"config_id": l["id"], "name": l["name"]}
        for g in ("MDD", "HC"):
            for k in ("faa", "tbr"):
                row[g + "_" + k] = float(np.mean(v[g][k]))
                row[g + "_" + k + "_sd"] = float(np.std(v[g][k]))
        row["faa_d"] = _cohen_d(v["HC"]["faa"], v["MDD"]["faa"])   # HC - MDD (expected > 0)
        row["tbr_d"] = _cohen_d(v["MDD"]["tbr"], v["HC"]["tbr"])   # MDD - HC (expected > 0)
        rows.append(row)
    return {"rows": rows, "snr_db": snr_db, "n_epochs": n_epochs, "warning": "; ".join(warns) or None}
