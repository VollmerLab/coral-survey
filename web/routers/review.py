import base64
import json
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from pipeline.db import (
    db_conn, get_coral, list_corals, update_coral_status, save_measurement
)
from pipeline.ingest import load_jpg
from pipeline.metrics import sample_region, whibal_correction_factor, area_cm2 as calc_area_cm2
from pipeline.segment_sam import mask_to_png_bytes, png_bytes_to_mask

router = APIRouter()

MASK_DIR = Path(__file__).parent.parent.parent / "data" / "masks"
MASK_DIR.mkdir(parents=True, exist_ok=True)


class PromptRequest(BaseModel):
    points: list[list[int]]   # [[x, y], ...]
    labels: list[int]          # 1=fg, 0=bg


class ConfirmRequest(BaseModel):
    mask_b64: str              # PNG mask as base64
    quality_flag: str = "ok"
    notes: str = ""
    genotype_id: str = ""
    species: str = ""
    scale_mm_px: float | None = None
    whibal_correction: float | None = None


@router.get("/sessions/{session_id}/queue")
def get_queue(session_id: int, status: str = "pending"):
    with db_conn() as conn:
        return list_corals(conn, session_id, status=status)


@router.get("/coral/{coral_id}/image")
def serve_image(coral_id: int):
    with db_conn() as conn:
        coral = get_coral(conn, coral_id)
    if not coral:
        raise HTTPException(404, "Coral not found")
    path = coral.get("best_photo_path") or coral.get("photo_a_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Image file not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/coral/{coral_id}/thumb")
def serve_thumb(coral_id: int, max_dim: int = 1200):
    with db_conn() as conn:
        coral = get_coral(conn, coral_id)
    if not coral:
        raise HTTPException(404, "Coral not found")
    path = coral.get("best_photo_path") or coral.get("photo_a_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Image file not found")

    img = load_jpg(path)
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        u8 = (img * 255).clip(0, 255).astype(np.uint8)
        resized = cv2.resize(u8, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/coral/{coral_id}/auto_mask")
def auto_mask(coral_id: int, request: Request):
    with db_conn() as conn:
        coral = get_coral(conn, coral_id)
    if not coral:
        raise HTTPException(404, "Coral not found")

    path = coral.get("best_photo_path") or coral.get("photo_a_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Image file not found")

    sam = request.app.state.sam
    img = load_jpg(path)

    if sam.available:
        masks = sam.auto_segment(img)
        if masks:
            # Return top 5 masks as PNG b64 for the UI to render
            results = []
            for m in masks[:5]:
                png = mask_to_png_bytes(m["predicted_iou"] > 0 if "predicted_iou" in m else m["segmentation"])
                png = mask_to_png_bytes(m["segmentation"])
                results.append({
                    "mask_b64": base64.b64encode(png).decode(),
                    "area": m["area"],
                    "score": float(m.get("predicted_iou", 0)),
                })
            return {"masks": results, "source": "sam"}
    # Fallback: whole-image foreground
    mask = sam.threshold_fallback(img)
    png = mask_to_png_bytes(mask.astype(bool))
    return {
        "masks": [{"mask_b64": base64.b64encode(png).decode(), "area": int(mask.sum()), "score": 0.5}],
        "source": "threshold",
    }


@router.post("/coral/{coral_id}/prompt")
def prompt_mask(coral_id: int, req: PromptRequest, request: Request):
    with db_conn() as conn:
        coral = get_coral(conn, coral_id)
    if not coral:
        raise HTTPException(404, "Coral not found")

    path = coral.get("best_photo_path") or coral.get("photo_a_path")
    if not path or not Path(path).exists():
        raise HTTPException(404, "Image file not found")

    sam = request.app.state.sam
    img = load_jpg(path)

    if sam.available:
        mask = sam.prompt_segment(img, [tuple(p) for p in req.points], req.labels)
    else:
        mask = sam.threshold_fallback(img).astype(bool)

    if mask is None:
        raise HTTPException(500, "SAM returned no mask")

    png = mask_to_png_bytes(mask)
    return {"mask_b64": base64.b64encode(png).decode()}


@router.post("/coral/{coral_id}/confirm")
def confirm_coral(coral_id: int, req: ConfirmRequest):
    with db_conn() as conn:
        coral = get_coral(conn, coral_id)
        if not coral:
            raise HTTPException(404, "Coral not found")

        path = coral.get("best_photo_path") or coral.get("photo_a_path")
        if not path or not Path(path).exists():
            raise HTTPException(404, "Image file not found")

        # Decode mask
        mask_bytes = base64.b64decode(req.mask_b64)
        mask = png_bytes_to_mask(mask_bytes)

        # Save mask PNG
        mask_path = MASK_DIR / f"coral_{coral_id}.png"
        mask_path.write_bytes(mask_to_png_bytes(mask))

        # Compute metrics
        img = load_jpg(path)
        l_mean, a_mean, b_mean = sample_region(img, mask)
        area_px = int(mask.sum())
        scale = req.scale_mm_px
        a_cm2 = calc_area_cm2(area_px, scale) if scale else None

        # Update coral metadata if provided
        updates = {}
        if req.genotype_id:
            updates["genotype_id"] = req.genotype_id
        if req.species:
            updates["species"] = req.species
        if updates:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            conn.execute(
                f"UPDATE corals SET {set_clause} WHERE id = ?",
                (*updates.values(), coral_id)
            )

        save_measurement(
            conn, coral_id,
            l_mean=l_mean, a_mean=a_mean, b_mean=b_mean,
            area_px=area_px, scale_mm_px=scale, area_cm2=a_cm2,
            whibal_correction=req.whibal_correction,
            quality_flag=req.quality_flag,
            notes=req.notes,
            mask_path=str(mask_path),
        )
        update_coral_status(conn, coral_id, "confirmed")

    return {"coral_id": coral_id, "l_mean": l_mean, "a_mean": a_mean, "b_mean": b_mean,
            "area_px": area_px, "area_cm2": a_cm2}


@router.post("/coral/{coral_id}/skip")
def skip_coral(coral_id: int):
    with db_conn() as conn:
        if not get_coral(conn, coral_id):
            raise HTTPException(404, "Coral not found")
        update_coral_status(conn, coral_id, "skipped")
    return {"coral_id": coral_id, "status": "skipped"}
