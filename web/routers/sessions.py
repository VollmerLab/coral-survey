import re
from pathlib import Path
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipeline.db import db_conn, list_sessions, create_session, insert_coral
from pipeline.ingest import collect_jpgs, group_duplicates, pick_sharpest, exif_datetime

router = APIRouter()

PHOTOS_ROOT = Path("/mnt/onefs/groups/vollmer_lab/photos")

_DATE_RE = re.compile(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})")


def _parse_folder_name(name: str) -> dict:
    """Best-effort parse of folder names like '9.3.2025_Tavernier Inventory_Bleaching'."""
    parts = name.split("_", 1)
    date_str = ""
    remainder = name

    m = _DATE_RE.match(parts[0])
    if m:
        mo, day, yr = m.groups()
        yr = yr if len(yr) == 4 else f"20{yr}"
        date_str = f"{yr}-{int(mo):02d}-{int(day):02d}"
        remainder = parts[1] if len(parts) > 1 else ""

    words = remainder.replace("_", " ").strip().split()
    site = words[0] if words else ""
    label = " ".join(words) if words else name

    return {"name": label, "site": site, "date": date_str}


class ImportRequest(BaseModel):
    folder_path: str
    name: str
    site: str = ""
    session_date: str = ""
    excel_path: str = ""


@router.get("/sessions/discover")
def discover_sessions(root: str = ""):
    search_root = Path(root) if root else PHOTOS_ROOT
    if not search_root.exists():
        return {"root": str(search_root), "folders": []}

    folders = []
    for p in sorted(search_root.iterdir()):
        if not p.is_dir():
            continue
        jpg_count = sum(1 for _ in p.rglob("*.JPG")) + sum(1 for _ in p.rglob("*.jpg"))
        if jpg_count == 0:
            continue
        meta = _parse_folder_name(p.name)
        folders.append({
            "path": str(p),
            "folder_name": p.name,
            "jpg_count": jpg_count,
            **meta,
        })

    return {"root": str(search_root), "folders": folders}


@router.get("/sessions")
def get_sessions():
    with db_conn() as conn:
        return list_sessions(conn)


@router.post("/sessions/import")
def import_session(req: ImportRequest):
    folder = Path(req.folder_path)
    if not folder.exists():
        raise HTTPException(404, f"Folder not found: {req.folder_path}")

    jpgs = collect_jpgs(folder, recursive=True)
    if not jpgs:
        raise HTTPException(400, "No JPG files found in folder")

    session_date = req.session_date or str(date.today())
    excel_path = req.excel_path or None

    with db_conn() as conn:
        session_id = create_session(
            conn,
            name=req.name,
            site=req.site,
            date=session_date,
            folder_path=str(folder),
            excel_path=excel_path,
        )

        groups = group_duplicates(jpgs)
        coral_count = 0
        for base, paths in groups.items():
            best = pick_sharpest(paths)
            if best is None:
                continue
            exif_t = exif_datetime(best)

            kwargs: dict = {
                "best_photo_path": str(best),
                "exif_time": exif_t.isoformat() if exif_t else None,
            }
            if len(paths) >= 2:
                sorted_paths = sorted(str(p) for p in paths)
                kwargs["photo_a_path"] = sorted_paths[0]
                kwargs["photo_b_path"] = sorted_paths[1]
            else:
                kwargs["photo_a_path"] = str(paths[0])

            insert_coral(conn, session_id, **kwargs)
            coral_count += 1

    return {"session_id": session_id, "corals_imported": coral_count}


@router.get("/sessions/{session_id}")
def get_session_detail(session_id: int):
    from pipeline.db import get_session, list_corals
    with db_conn() as conn:
        session = get_session(conn, session_id)
        if not session:
            raise HTTPException(404, "Session not found")
        corals = list_corals(conn, session_id)
        session["corals"] = corals
        return session
