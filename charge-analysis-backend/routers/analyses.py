"""充电分析相关 API。"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from config import get_settings
from core.dependencies import get_current_user
from database import get_db
from models import AnalysisResult, AnalysisStatus, ChargingAnalysis, User
from schemas import AnalysisListResponse, AnalysisRead, AnalysisRunResponse
from services.analysis_service import get_analysis_service

router = APIRouter(prefix="/api/analyses", tags=["analyses"])
settings = get_settings()
analysis_service = get_analysis_service()


@router.get("", response_model=AnalysisListResponse)
def list_analyses(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> AnalysisListResponse:
    records = (
        db.query(ChargingAnalysis)
        .filter(ChargingAnalysis.user_id == user.id)
        .order_by(ChargingAnalysis.created_at.desc())
        .all()
    )
    return AnalysisListResponse(items=[AnalysisRead.model_validate(r) for r in records])


@router.post("/upload", response_model=AnalysisRead)
async def upload_analysis(
    file: UploadFile = File(...),
    analysis_name: str | None = Form(None),
    description: str | None = Form(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisRead:
    upload_dir = Path(settings.upload_path) / str(user.id)
    upload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(datetime.utcnow().timestamp())
    filename = f"{timestamp}_{file.filename}"
    file_path = upload_dir / filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_size = file_path.stat().st_size

    analysis = ChargingAnalysis(
        name=analysis_name or file.filename,
        description=description,
        file_path=str(file_path),
        file_size=file_size,
        file_type=file.filename.split(".")[-1],
        status=AnalysisStatus.PENDING,
        progress=0.0,
        user_id=user.id,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return AnalysisRead.model_validate(analysis)


@router.post("/{analysis_id}/run", response_model=AnalysisRunResponse)
async def run_analysis(
    analysis_id: int,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisRunResponse:
    analysis = (
        db.query(ChargingAnalysis)
        .filter(ChargingAnalysis.id == analysis_id, ChargingAnalysis.user_id == user.id)
        .first()
    )
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分析不存在")

    if not Path(analysis.file_path).exists():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="分析文件不存在")

    analysis.status = AnalysisStatus.PROCESSING
    analysis.progress = 5.0
    analysis.error_message = None
    db.add(analysis)
    db.commit()

    background_tasks.add_task(analysis_service.run_analysis, analysis_id)
    return AnalysisRunResponse(analysis_id=analysis_id, status="processing")


@router.get("/{analysis_id}", response_model=AnalysisRead)
def get_analysis(analysis_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> AnalysisRead:
    analysis = (
        db.query(ChargingAnalysis)
        .filter(ChargingAnalysis.id == analysis_id, ChargingAnalysis.user_id == user.id)
        .first()
    )
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分析不存在")
    return AnalysisRead.model_validate(analysis)


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_analysis(analysis_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> None:
    analysis = (
        db.query(ChargingAnalysis)
        .filter(ChargingAnalysis.id == analysis_id, ChargingAnalysis.user_id == user.id)
        .first()
    )
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分析不存在")

    file_path = Path(analysis.file_path)
    db.delete(analysis)
    db.commit()

    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            pass


@router.get("/{analysis_id}/results")
def list_results(analysis_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> dict:
    analysis = (
        db.query(ChargingAnalysis)
        .filter(ChargingAnalysis.id == analysis_id, ChargingAnalysis.user_id == user.id)
        .first()
    )
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分析不存在")

    results = (
        db.query(AnalysisResult)
        .filter(AnalysisResult.analysis_id == analysis_id)
        .order_by(AnalysisResult.created_at.asc())
        .all()
    )
    return {
        "analysis": AnalysisRead.model_validate(analysis),
        "results": [
            {
                "id": r.id,
                "analysis_id": r.analysis_id,
                "result_type": r.result_type,
                "title": r.title,
                "content": r.content,
                "metadata": r.metadata,
                "confidence_score": r.confidence_score,
                "created_at": r.created_at.isoformat(),
            }
            for r in results
        ],
    }
