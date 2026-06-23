# CLAUDE.md — Coral Survey Pipeline

Developer context. Read before touching any file.

---

## What this is

Web app for reviewing underwater coral survey photos. Dive teams use unmodified Olympus TG-7 cameras; lab staff review photos in a browser, refine SAM segmentation masks, and confirm CIELab metrics + area to a SQLite database.

**Separate from ndvi_analysis** (tray/nursery pipeline). Those repos share concepts but are independent codebases. This repo is for the in-situ survey workflow only.

**Team:** Steven R. Vollmer (PI) + Renata Suarez — Vollmer Lab, Northeastern / FAU  
**Language:** Python only. No R.  
**Style:** concise, no over-engineering, no comments explaining what the code does.

---

## Architecture

```
pipeline/        ← shared modules (no web dependencies)
web/             ← FastAPI app + static HTML/JS
config/          ← YAML config (card dimensions, SAM params)
data/            ← SQLite DB + masks (gitignored)
models/          ← sam_vit_b.pth (gitignored)
```

FastAPI backend on port 8080. Single-page HTML5 frontend. SAM runs on Athene GPU (falls back to CPU). No tkinter — this is a browser app.

---

## Key design decisions

### Card detection
- **WhiBal G7 wallet (53.3 × 85.1 mm)** in lower-left corner of every survey frame
- Card dimensions are config values — never hardcode in pipeline/card.py or anywhere else
- Same detection code must work at any card size (related photomosaic project uses larger WhiBal cards)
- Detection strategy: position prior (lower-left corner) → full-frame fallback → GUI manual click
- Calibration math: sample gray face → measure L* → correction = 49.5 / measured_L*
- WhiBal reference L* = 49.5 (18% gray under D65; from RawWorkflow spectral data)

### Duplicate photo selection
- Dive teams take a/b duplicate shots per colony
- `pipeline/ingest.py:group_duplicates()` pairs them by filename suffix (a/b)
- Sharpest picked by Laplacian variance — operator can override in review UI

### SAM model
- `vit_b` only. vit_h is too slow for interactive review.
- Auto-segment on image load → show top mask → operator clicks to refine
- Prompt points: left-click = foreground, shift-click = background, right-click = undo last

### Database
- SQLite at `data/survey.db`
- Tables: sessions, corals, measurements
- Confirmed corals: status = 'confirmed'; skipped: status = 'skipped'
- Measurements store L*, a*, b*, area_px, scale_mm_px, area_cm2, whibal_correction, mask_path

### Images
- JPGs referenced by path (not copied into the repo)
- Thumbnails resized to max_dim=1200px for canvas display
- Full-res loaded for SAM and metric extraction

---

## What's not here yet

- `pipeline/survey_ingest.py` — full session import with Excel matching, slate detection (needs dive photos)
- `train.py` — SAM fine-tuning script (referenced by training router but not yet written)
- WhiBal detection integration into the review workflow (card.py exists but review.py doesn't call it yet)
- QR code detection (pipeline/card.py:detect_qr() exists but tray pipeline uses it, not survey)

---

## Running locally (dev)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install torch torchvision  # CPU is fine for dev
python -m web.main
```

App at http://localhost:8080. SAM will log "model not found" if weights are missing — the threshold fallback kicks in.

---

## Critical constants

| Constant | Value | Why |
|----------|-------|-----|
| WhiBal L* ref | 49.5 | 18% gray under D65 |
| WhiBal wallet size | 53.3 × 85.1 mm | From RawWorkflow card specs |
| SAM model | vit_b | Speed/accuracy balance |
| Port | 8080 | SSH tunnel default |
