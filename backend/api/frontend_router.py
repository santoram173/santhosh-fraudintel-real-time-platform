"""Serve the frontend dashboard."""
from fastapi import APIRouter
from fastapi.responses import FileResponse
from pathlib import Path

router = APIRouter()
FRONTEND = Path(__file__).parent.parent.parent / "frontend" / "index.html"

@router.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND)
