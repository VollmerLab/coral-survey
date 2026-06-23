import asyncio
import json
import shutil
import subprocess
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from pipeline.db import db_conn, list_confirmed_pairs

router = APIRouter()

DATASET_DIR = Path(__file__).parent.parent.parent / "data" / "training_dataset"
_training_proc: subprocess.Popen | None = None


@router.get("/training/status")
def training_status():
    with db_conn() as conn:
        pairs = list_confirmed_pairs(conn)
    return {
        "confirmed_pairs": len(pairs),
        "training_running": _training_proc is not None and _training_proc.poll() is None,
    }


class ExportRequest(BaseModel):
    output_dir: str = ""


@router.post("/training/export")
def export_dataset(req: ExportRequest):
    out = Path(req.output_dir) if req.output_dir else DATASET_DIR
    out.mkdir(parents=True, exist_ok=True)
    images_dir = out / "images"
    masks_dir = out / "masks"
    images_dir.mkdir(exist_ok=True)
    masks_dir.mkdir(exist_ok=True)

    with db_conn() as conn:
        pairs = list_confirmed_pairs(conn)

    copied = 0
    manifest = []
    for p in pairs:
        photo = Path(p["best_photo_path"])
        mask = Path(p["mask_path"]) if p["mask_path"] else None
        if not photo.exists():
            continue
        dst_img = images_dir / photo.name
        shutil.copy2(photo, dst_img)
        entry = {
            "image": str(dst_img),
            "mask": None,
            "species": p.get("species"),
            "genotype_id": p.get("genotype_id"),
        }
        if mask and mask.exists():
            dst_mask = masks_dir / mask.name
            shutil.copy2(mask, dst_mask)
            entry["mask"] = str(dst_mask)
        manifest.append(entry)
        copied += 1

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return {"exported": copied, "output_dir": str(out), "manifest": str(manifest_path)}


class TrainRequest(BaseModel):
    dataset_dir: str = ""
    epochs: int = 10


@router.post("/training/start")
def start_training(req: TrainRequest):
    global _training_proc
    if _training_proc is not None and _training_proc.poll() is None:
        return {"status": "already_running"}

    dataset = req.dataset_dir or str(DATASET_DIR)
    script = Path(__file__).parent.parent.parent / "train.py"
    if not script.exists():
        return {"status": "error", "message": "train.py not found"}

    log_path = Path(__file__).parent.parent.parent / "data" / "training.log"
    log_file = open(log_path, "w")
    _training_proc = subprocess.Popen(
        ["python", str(script), "--dataset", dataset, "--epochs", str(req.epochs)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(script.parent),
    )
    return {"status": "started", "pid": _training_proc.pid}


@router.get("/training/log")
async def stream_log():
    log_path = Path(__file__).parent.parent.parent / "data" / "training.log"

    async def event_gen():
        if not log_path.exists():
            yield "data: No log file yet\n\n"
            return
        with open(log_path) as f:
            while True:
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    if _training_proc is None or _training_proc.poll() is not None:
                        yield "data: [done]\n\n"
                        break
                    await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
