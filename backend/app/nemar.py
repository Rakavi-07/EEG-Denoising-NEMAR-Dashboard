"""Optional loader for the NEMAR nm000114 BIDS dataset via MNE-BIDS."""
import os
import numpy as np
from .config import CHANNELS, FS, EPOCH_SEC

ALIASES = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
_LOOKUP = {c.upper(): c for c in CHANNELS}


def root():
    return os.environ.get("NEMAR_ROOT", "")


def available():
    return bool(root()) and os.path.isdir(root())


def _norm(name):
    n = name.upper().replace("EEG", "").strip(" -_")
    n = n.split("-")[0].split(" ")[0]
    return _LOOKUP.get(ALIASES.get(n, n))


def _is_mdd(v):
    return any(k in str(v).lower() for k in ("mdd", "patient", "depress"))


def load_epochs(group, n_epochs, seed):
    if not available():
        raise RuntimeError("NEMAR_ROOT is not set or not a directory")
    import mne
    import pandas as pd
    from mne_bids import find_matching_paths, read_raw_bids

    r = root()
    parts = pd.read_csv(os.path.join(r, "participants.tsv"), sep="\t")
    gcol = next((c for c in parts.columns if c.lower() in ("group", "diagnosis", "condition")), None)
    if gcol is None:
        raise RuntimeError("participants.tsv has no group/diagnosis/condition column")
    want = group.upper() == "MDD"
    subs = [str(s).replace("sub-", "") for s, v in zip(parts["participant_id"], parts[gcol]) if _is_mdd(v) == want]
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)

    chunks, total = [], 0
    for s in subs:
        paths = find_matching_paths(r, subjects=s, datatypes="eeg",
                                    extensions=[".edf", ".vhdr", ".set", ".bdf", ".fif"])
        for bp in paths:
            raw = read_raw_bids(bp, verbose=False)
            raw.load_data(verbose=False)
            mapping = {c: _norm(c) for c in raw.ch_names if _norm(c)}
            raw.pick(list(mapping))
            raw.rename_channels(mapping)
            if set(raw.ch_names) != set(CHANNELS):
                continue
            raw.reorder_channels(CHANNELS)
            raw.resample(FS, verbose=False)
            ep = mne.make_fixed_length_epochs(raw, duration=EPOCH_SEC, preload=True, verbose=False)
            chunks.append(ep.get_data() * 1e6)  # V -> uV
            total += len(chunks[-1])
            break
        if total >= n_epochs * 3:
            break
    if not chunks:
        raise RuntimeError("no usable subjects/epochs found for group " + group)
    data = np.concatenate(chunks)
    idx = rng.choice(len(data), size=min(n_epochs, len(data)), replace=False)
    return data[idx]
