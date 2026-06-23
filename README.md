# Coral Survey Pipeline

Web-based review tool for tracking coral health from underwater photo surveys.
Vollmer Lab — Northeastern / FAU. PI: Steven R. Vollmer. Contact: Renata Suarez.

---

## What this does

Dive teams photograph individual coral colonies in situ using unmodified Olympus TG-7 cameras. A WhiBal G7 gray card in the lower-left frame corner provides per-photo color calibration and scale. This web app lets lab staff review each photo, refine the SAM auto-segmentation mask with a single click, and save CIELab color metrics + colony area to a database.

Over time, confirmed image+mask pairs train a fine-tuned SAM model that improves auto-segmentation on reef backgrounds.

---

## Quick start (Athene server)

```bash
# 1. Clone and set up
git clone https://github.com/VollmerLab/coral-survey.git
cd coral-survey
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 2. Download SAM weights (~375 MB)
mkdir -p models
wget -O models/sam_vit_b.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# 3. Run
python -m web.main
# App is at http://localhost:8080
```

**From outside the university network:**
```bash
ssh -L 8080:localhost:8080 user@athene.neu.edu
# Then open http://localhost:8080 in your browser
```

**Keep it running after SSH logout:**
```bash
nohup python -m web.main > logs/server.log 2>&1 &
echo $!  # note the PID to kill later
```

---

## Project layout

```
coral_survey/
├── pipeline/
│   ├── db.py              ← SQLite: sessions / corals / measurements
│   ├── ingest.py          ← load JPG, EXIF, Laplacian sharpness, duplicate grouping
│   ├── metrics.py         ← CIELab extraction, WhiBal correction factor
│   ├── segment_sam.py     ← SAM vit_b wrapper (auto + prompt)
│   └── card.py            ← WhiBal G7 detection + QR code detection
├── web/
│   ├── main.py            ← FastAPI app (port 8080)
│   ├── routers/
│   │   ├── sessions.py    ← import folder, list sessions
│   │   ├── review.py      ← serve image, SAM inference, confirm, skip
│   │   └── training.py    ← export dataset, trigger training, SSE log
│   └── static/
│       ├── index.html     ← single-page app
│       ├── review.js      ← canvas mask overlay + SAM prompt click handling
│       └── style.css
├── config/
│   └── survey.yaml        ← card dimensions, SAM params, app config
├── data/                  ← survey.db, masks/ (gitignored)
├── models/                ← sam_vit_b.pth (gitignored — download separately)
└── requirements.txt
```

---

## Field protocol summary

- Unmodified TG-7 (standard color — no filter modifications)
- WhiBal G7 wallet card (53.3 × 85.1 mm) held in lower-left corner of every frame, gray face forward, parallel to sensor
- Two shots per colony (a/b); pipeline auto-selects sharpest
- Start/stop slate photos bookend each session
- Species + genotype recorded in Excel per dive

See `docs/PROTOCOL_SURVEY.md` in the tray pipeline repo for full diver instructions.

---

## Card dimensions

Card sizes come from `config/survey.yaml` — never hardcoded. The same detection code is used in a related photomosaic project with larger WhiBal cards.

| Card | Size |
|------|------|
| WhiBal G7 wallet | 53.3 × 85.1 mm |
| CR-80 (tray pipeline ID cards) | 85.6 × 54.0 mm |

---

## SAM model

Using `sam_vit_b` (smallest ViT variant, 375 MB). Balances speed and accuracy for batch review on Athene GPU. Model falls back to CPU if CUDA is unavailable.

Download: `https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`
