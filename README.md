# EEG Denoising & MDD Biomarker Dashboard (NEMAR nm000114)

Research dashboard: synthetic noise engine (Eq. 1), six denoising configurations,
quality metrics, FAA / TBR biomarker preservation, and CPU vs GPU compute simulation.

> Research / education tool. Not a diagnostic device. Default data is **synthetic**
> (19-ch 10-20 montage, 256 Hz, 10 s epochs) with simulated MDD-vs-HC differences.
> Real NEMAR data is optional (see step 6).

## 1. Prerequisites
- Python 3.10+  (3.11 recommended)
- Node.js 18+ and npm
- VS Code with the *Python* extension

## 2. Open in VS Code
```
code EEG_Denoising_NEMAR_Dashboard
```
Open two terminals (Terminal > New Terminal, then the split icon).

## 3. Backend (terminal 1)
```
cd backend
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Check http://localhost:8000/api/health and the docs at http://localhost:8000/docs.
In VS Code: Ctrl+Shift+P > "Python: Select Interpreter" > choose `backend/.venv`.

## 4. Frontend (terminal 2)
```
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 (Vite proxies `/api` to port 8000).

## 5. Using the dashboard
1. **Signal Lab** – pick group, input SNR, channel; overlay raw / noisy / denoised traces and PSDs.
2. **Benchmark** – runs all 4 SNR levels x 6 configs x both groups; SNR gain, RMSE, PRD, r, band-power errors.
3. **MDD Biomarkers** – FAA and TBR after each denoiser, MDD vs HC (Cohen's d).
4. **GPU Compute** – measured CPU time vs simulated GPU end-to-end (3.92x) and device-resident (18.09x).

## 6. Optional: real NEMAR nm000114 data
1. Download the BIDS dataset from nemar.org (dataset nm000114) to a local folder.
2. Set the folder before starting the backend:
   - macOS/Linux: `export NEMAR_ROOT=/path/to/nm000114`
   - PowerShell:  `$env:NEMAR_ROOT="C:\data\nm000114"`
3. Choose "NEMAR (MNE-BIDS)" as the data source in the UI.
The loader reads `participants.tsv` (needs a `group`/`diagnosis`/`condition` column),
keeps the 19 montage channels (T7/T8/P7/P8 mapped to T3/T4/T5/T6), resamples to 256 Hz and
cuts 10 s epochs. If anything fails it falls back to synthetic data and shows a warning.
The recorded EEG is treated as the "clean" reference r.

## 7. Tests
```
cd backend
pip install -r requirements-dev.txt
pytest -q
```

## Design notes / assumptions
- **Eq. 1** is applied per channel: `RMS(n) = RMS(r) * 10^(-SNR/20)`. The three components are
  normalised to unit RMS, mixed with weights 0.55 / 0.25 / 0.40 (sum = 1.2), then the mix is
  rescaled so the achieved SNR matches the target exactly.
- FAA = ln(alpha power F4) - ln(alpha power F3); TBR = theta(4-8) / beta(13-30) power at Fz.
  MDD simulation: relatively more left-frontal alpha (lower FAA), higher theta, lower beta.
- Wavelet sigma = median(|d1|)/0.6745. BayesShrink: T = sigma^2 / sigma_x per sub-band.
  Energy retention keeps the largest detail coefficients holding 90 % of sub-band energy.
- GPU numbers are **simulated** from measured CPU time (divide by 3.92 / 18.09). Replace
  `backend/app/backends.py` with real CuPy kernels to measure on hardware.

## Layout
```
backend/   FastAPI service (app/config, eeg_sim, noise, denoise, metrics, backends, nemar, pipeline, main)
frontend/  React + Vite + Tailwind + Recharts + Lucide
```
