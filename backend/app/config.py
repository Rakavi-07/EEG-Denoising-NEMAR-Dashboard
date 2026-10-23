FS = 256
EPOCH_SEC = 10
N_SAMPLES = FS * EPOCH_SEC  # 2560

CHANNELS = ["Fp1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "Fp2", "F4",
            "C4", "P4", "O2", "F8", "T4", "T6", "Fz", "Cz", "Pz"]

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 40)}

SNR_LEVELS = [-5, 0, 5, 10]
NOISE_WEIGHTS = {"white": 0.55, "line": 0.25, "emg": 0.40}
LINE_HZ = 50.0
EMG_BAND = (20.0, 100.0)

FIR_TAPS = 129
FIR_BAND = (1.0, 40.0)
WAVELET = "db4"
LEVEL = 4
ENERGY_RETAIN = 0.90

SPEEDUP_E2E = 3.92
SPEEDUP_DEVICE = 18.09

CONFIGS = [
    {"id": 1, "key": "none", "name": "No Denoising"},
    {"id": 2, "key": "fir", "name": "FIR 1-40 Hz"},
    {"id": 3, "key": "wav_universal", "name": "Wavelet Universal"},
    {"id": 4, "key": "wav_bayes", "name": "Wavelet BayesShrink"},
    {"id": 5, "key": "fir_bayes", "name": "FIR + BayesShrink"},
    {"id": 6, "key": "fir_energy", "name": "FIR + Energy 90%"},
]
