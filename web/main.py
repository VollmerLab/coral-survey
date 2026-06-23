from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from pipeline.db import init_db
from pipeline.segment_sam import SAMWrapper
from web.routers import sessions, review, training

STATIC_DIR = Path(__file__).parent / "static"
MODEL_PATH = Path(__file__).parent.parent / "models" / "sam_vit_b.pth"


def create_app() -> FastAPI:
    init_db()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = SAMWrapper(MODEL_PATH, device=device)

    app = FastAPI(title="Coral Survey Review")
    app.state.sam = sam

    app.include_router(sessions.router, prefix="/api")
    app.include_router(review.router, prefix="/api")
    app.include_router(training.router, prefix="/api")

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/static/index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web.main:app", host="0.0.0.0", port=8080, reload=False)
