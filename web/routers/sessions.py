from pathlib import Path
from datetime import date

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from pipeline.db import db_conn, list_sessions, create_session, insert_coral
from pipeline.ingest import collect_jpgs, group_duplicates, pick_sharpest, exif_datetime

router = APIRouter()


class ImportRequest(BaseModel):
    folder_path: str
    name: str
    site: str = ""
    session_date: str = ""
    excel_path: str = ""


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
