"""充电分析相关 API。"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Body, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from config import get_settings
from core.dependencies import get_current_user
from database import get_db
from models import AnalysisResult, AnalysisStatus, ChargingAnalysis, User
from schemas import (
    AnalysisCancelResponse,
    AnalysisListResponse,
    AnalysisRead,
    AnalysisResultRead,
    AnalysisResultsResponse,
    AnalysisRunRequest,
    AnalysisRunResponse,
    AvailableSignalsResponse,
    SignalInfo,
)
from services.analysis_service import get_analysis_service

router = APIRouter(prefix="/api/analyses", tags=["analyses"])
settings = get_settings()
logger = logging.getLogger(__name__)


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
    analysis_name: str | None = Form(None, example="国标充电日志-2024Q4"),
    description: str | None = Form(None, example="12月现场采集的DC桩充电过程"),
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


@router.get("/signals/available", response_model=AvailableSignalsResponse)
def get_available_signals(user: User = Depends(get_current_user)) -> AvailableSignalsResponse:
    """获取 DBC 文件中所有可用的信号列表（优先使用用户配置的 DBC）。"""
    from services.can_parser import CANLogParser
    from services.dbc_config import resolve_user_dbc_path

    try:
        dbc_path = resolve_user_dbc_path(int(user.id))
        parser = CANLogParser(dbc_file_path=dbc_path) if dbc_path else CANLogParser()
        logger.info(
            "获取可用信号 user_id=%s dbc=%s",
            user.id,
            dbc_path or str(parser.dbc_file_path),
        )

        all_signals = parser.get_available_signals()
        messages = parser.get_available_messages()

        # 构建消息ID到消息名称的映射
        message_id_to_name = {msg["frame_id"]: msg["name"] for msg in messages}

        # 构建信号信息列表
        signal_infos = []
        signal_to_message_ids = parser._signal_to_message_ids

        for signal_name in sorted(all_signals):
            message_ids = signal_to_message_ids.get(signal_name, [])
            if message_ids:
                first_msg_id = message_ids[0]
                message_name = message_id_to_name.get(first_msg_id, "Unknown")
                signal_infos.append(
                    SignalInfo(name=signal_name, message_name=message_name, message_id=hex(first_msg_id))
                )

        return AvailableSignalsResponse(signals=signal_infos, total_count=len(signal_infos))
    except Exception as e:
        logger.exception("获取信号列表失败 user_id=%s err=%s", user.id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"获取信号列表失败: {str(e)}")


@router.get("/dbc/current")
def get_current_dbc(user: User = Depends(get_current_user)) -> dict:
    """获取当前用户配置的 DBC 信息。"""
    from services.dbc_config import dbc_config_to_response, load_user_dbc_config

    cfg = load_user_dbc_config(int(user.id))
    return dbc_config_to_response(cfg)


@router.post("/dbc/upload")
async def upload_dbc(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
) -> dict:
    """上传并设置当前用户用于解析的 DBC 文件。"""
    from services.dbc_config import dbc_config_to_response, get_user_dbc_dir, save_user_dbc_config

    filename = file.filename or "uploaded.dbc"
    if not filename.lower().endswith(".dbc"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持上传 .dbc 文件")

    dbc_dir = get_user_dbc_dir(int(user.id))
    dbc_dir.mkdir(parents=True, exist_ok=True)

    timestamp = int(datetime.utcnow().timestamp())
    safe_name = filename.replace("/", "_").replace("\\", "_")
    dst_path = dbc_dir / f"{timestamp}_{safe_name}"

    try:
        with dst_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.exception("保存 DBC 失败 user_id=%s filename=%s err=%s", user.id, filename, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"保存 DBC 失败: {str(e)}")

    cfg = save_user_dbc_config(int(user.id), dst_path, filename)
    logger.info("DBC 上传成功 user_id=%s filename=%s saved=%s", user.id, filename, str(dst_path))
    return dbc_config_to_response(cfg)


@router.post("/{analysis_id}/run", response_model=AnalysisRunResponse)
async def run_analysis(
    analysis_id: int,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    request: AnalysisRunRequest = Body(
        ...,
        example={"signal_names": ["BatteryVoltage", "ChargeCurrent", "StateOfCharge"]},
    ),
) -> AnalysisRunResponse:
    """运行分析，支持选择特定信号进行解析"""
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

    # 传递信号列表给分析服务
    signal_names = request.signal_names if request.signal_names else None
    # 延迟初始化：避免启动阶段就加载工作流/向量库依赖
    analysis_service = get_analysis_service()
    background_tasks.add_task(analysis_service.run_analysis, analysis_id, signal_names)
    return AnalysisRunResponse(analysis_id=analysis_id, status="processing")


@router.post("/{analysis_id}/cancel", response_model=AnalysisCancelResponse)
def cancel_analysis(
    analysis_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AnalysisCancelResponse:
    """取消分析（总是成功）
    
    无论分析是否在运行中，都会成功返回：
    - 如果任务正在运行，则停止它
    - 如果任务不在运行，则更新数据库状态为已取消
    """
    analysis = (
        db.query(ChargingAnalysis)
        .filter(ChargingAnalysis.id == analysis_id, ChargingAnalysis.user_id == user.id)
        .first()
    )
    if analysis is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="分析不存在")

    # 无论状态如何，都执行取消操作（总是成功）
    # 延迟初始化：避免启动阶段就加载工作流/向量库依赖
    analysis_service = get_analysis_service()
    cancelled = analysis_service.cancel_analysis(analysis_id)
    
    # cancel_analysis 总是返回 True，所以这里总是成功
    return AnalysisCancelResponse(
        message="分析已取消",
        analysis_id=analysis_id,
        was_running=analysis.status == AnalysisStatus.PROCESSING,
    )


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


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT, response_model=None)
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
    
    return None


@router.get("/{analysis_id}/results", response_model=AnalysisResultsResponse)
def list_results(
    analysis_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)
) -> AnalysisResultsResponse:
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
    return AnalysisResultsResponse(
        analysis=AnalysisRead.model_validate(analysis),
        results=[AnalysisResultRead.model_validate(r) for r in results],
    )
