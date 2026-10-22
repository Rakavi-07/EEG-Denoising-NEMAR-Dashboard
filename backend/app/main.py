from typing import List, Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from . import config as C, pipeline, backends, nemar

app = FastAPI(title="EEG Denoising & MDD Biomarker API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

Source = Literal["synthetic", "nemar"]
Group = Literal["MDD", "HC"]


class SimReq(BaseModel):
    group: Group = "MDD"
    snr_db: float = 0
    channel: str = "F3"
    seed: int = 7
    source: Source = "synthetic"


class BenchReq(BaseModel):
    snr_levels: List[float] = Field(default_factory=lambda: list(C.SNR_LEVELS))
    groups: List[Group] = Field(default_factory=lambda: ["MDD", "HC"])
    n_epochs: int = Field(4, ge=1, le=30)
    seed: int = 7
    source: Source = "synthetic"


class BioReq(BaseModel):
    snr_db: float = 5
    n_epochs: int = Field(10, ge=3, le=40)
    seed: int = 7
    source: Source = "synthetic"


@app.get("/api/health")
def health():
    return {"status": "ok", "cupy_available": backends.CUPY_AVAILABLE,
            "nemar_root_set": nemar.available()}


@app.get("/api/meta")
def meta():
    return {"channels": C.CHANNELS, "fs": C.FS, "epoch_sec": C.EPOCH_SEC, "n_samples": C.N_SAMPLES,
            "bands": C.BANDS, "snr_levels": C.SNR_LEVELS, "noise_weights": C.NOISE_WEIGHTS,
            "configs": C.CONFIGS, "speedups": {"e2e": C.SPEEDUP_E2E, "device": C.SPEEDUP_DEVICE}}


@app.post("/api/simulate")
def simulate(req: SimReq):
    return pipeline.run_simulation(req.group, req.snr_db, req.channel, req.seed, req.source)


@app.post("/api/benchmark")
def benchmark(req: BenchReq):
    return pipeline.run_benchmark(req.snr_levels, req.n_epochs, req.groups, req.seed, req.source)


@app.post("/api/biomarkers")
def biomarkers(req: BioReq):
    return pipeline.run_biomarkers(req.snr_db, req.n_epochs, req.seed, req.source)
