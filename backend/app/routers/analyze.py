"""
영상 분석 라우터 모듈.

YouTube 영상 분석 및 레시피 추출 API를 제공합니다.
"""
import asyncio
import logging
import shutil
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.config import DATA_DIR, JOB_EXPIRE_HOURS, MAX_JOBS
from app.exceptions import (
    RecipeParseError,
    TranscriptionError,
    YouTubeDownloadError,
)
from app.schemas.analyze import (
    AnalyzeRequest,
    AnalyzeResponse,
    JobStatusResponse,
)
from app.services.recipe_parser import parse_recipe
from app.services.transcribe import transcribe_audio
from app.services.youtube import download_video, extract_video_id

# =============================================================================
# 로깅 및 라우터 설정
# =============================================================================
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Analyze"])

# =============================================================================
# 상수
# =============================================================================
MIN_TRANSCRIPT_LENGTH = 20


# =============================================================================
# 작업 관리자 클래스
# =============================================================================
class JobManager:
    """작업 상태 관리 클래스 (메모리 관리 포함)."""

    def __init__(
        self,
        max_jobs: int = MAX_JOBS,
        expire_hours: int = JOB_EXPIRE_HOURS
    ):
        self._jobs: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._max_jobs = max_jobs
        self._expire_hours = expire_hours
        self._lock = asyncio.Lock()

    def create_job(
        self,
        job_id: str,
        url: str,
        video_id: str
    ) -> Dict[str, Any]:
        """새 작업을 생성합니다."""
        job = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "message": "대기 중...",
            "url": url,
            "video_id": video_id,
            "result": None,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        self._jobs[job_id] = job
        self._cleanup_old_jobs()
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """작업을 조회합니다."""
        return self._jobs.get(job_id)

    def update_job(self, job_id: str, **kwargs) -> None:
        """작업 상태를 업데이트합니다."""
        if job_id in self._jobs:
            self._jobs[job_id].update(kwargs)
            self._jobs[job_id]["updated_at"] = datetime.now()

    def delete_job(self, job_id: str) -> bool:
        """작업을 삭제합니다."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            return True
        return False

    def cleanup_job_files(self, job_id: str) -> None:
        """작업 관련 파일을 삭제합니다."""
        job_dir = DATA_DIR / job_id
        if job_dir.exists():
            try:
                shutil.rmtree(job_dir)
                logger.debug(f"작업 파일 삭제: {job_dir}")
            except Exception as e:
                logger.warning(f"작업 파일 삭제 실패: {job_dir}, {e}")

    def get_stats(self) -> Dict[str, Any]:
        """작업 통계를 조회합니다."""
        status_counts: Dict[str, int] = {}
        for job in self._jobs.values():
            status = job.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_jobs": len(self._jobs),
            "max_jobs": self._max_jobs,
            "status_counts": status_counts
        }

    def _cleanup_old_jobs(self) -> None:
        """오래된 작업을 정리합니다."""
        expire_threshold = datetime.now() - timedelta(hours=self._expire_hours)

        expired_jobs = [
            job_id for job_id, job in self._jobs.items()
            if job.get("created_at", datetime.now()) < expire_threshold
        ]

        for job_id in expired_jobs:
            self.cleanup_job_files(job_id)
            del self._jobs[job_id]
            logger.info(f"만료된 작업 삭제: {job_id}")

        while len(self._jobs) > self._max_jobs:
            oldest_job_id = next(iter(self._jobs))
            self.cleanup_job_files(oldest_job_id)
            del self._jobs[oldest_job_id]
            logger.info(f"오래된 작업 삭제 (용량 초과): {oldest_job_id}")


# 전역 작업 관리자
job_manager = JobManager()


# =============================================================================
# 영상 처리 단계별 함수
# =============================================================================
async def _step_download(
    job_id: str,
    url: str,
    job_dir: Path
) -> Dict[str, Any]:
    """
    1단계: 영상 다운로드.

    Args:
        job_id: 작업 ID
        url: YouTube URL
        job_dir: 작업 디렉토리

    Returns:
        비디오 정보 딕셔너리

    Raises:
        Exception: 다운로드 실패 시
    """
    job_manager.update_job(
        job_id,
        status="processing",
        step="download",
        message="영상 다운로드 중...",
        progress=5
    )

    try:
        video_info = await download_video(url, str(job_dir))
    except YouTubeDownloadError as e:
        raise Exception(f"영상 다운로드 실패: {e}")

    job_manager.update_job(
        job_id,
        message="다운로드 완료!",
        progress=25,
        video_info=video_info
    )

    return video_info


async def _step_extract_transcript(
    job_id: str,
    url: str,
    job_dir: Path,
    audio_path: str
) -> tuple[Dict[str, Any], str]:
    """
    2단계: 자막/STT 추출.

    YouTube 자막 먼저 시도, 없으면 Whisper STT 폴백.

    Args:
        job_id: 작업 ID
        url: YouTube URL
        job_dir: 작업 디렉토리
        audio_path: 오디오 파일 경로

    Returns:
        (transcript 딕셔너리, source 문자열) 튜플

    Raises:
        Exception: 텍스트 추출 실패 시
    """
    transcript = None
    transcript_source = None

    # # YouTube 자막 시도 (주석처리 - Whisper 성능 테스트용)
    # job_manager.update_job(
    #     job_id,
    #     step="subtitle",
    #     message="YouTube 자막 확인 중...",
    #     progress=28
    # )
    #
    # try:
    #     subtitle_info = await download_subtitles(url, str(job_dir))
    #     if subtitle_info:
    #         transcript = parse_json3_subtitles(subtitle_info["subtitle_path"])
    #         if transcript and transcript.get("full_text"):
    #             transcript_source = f"youtube_{subtitle_info['language']}"
    #             if subtitle_info["is_auto_generated"]:
    #                 transcript_source += "_auto"
    #             logger.info(
    #                 f"[{job_id[:8]}] YouTube 자막 사용: {transcript_source}"
    #             )
    # except (SubtitleError, Exception) as e:
    #     logger.warning(f"[{job_id[:8]}] YouTube 자막 처리 실패: {e}")

    # Whisper STT 사용 (자막 로직 비활성화)
    if True:  # 항상 Whisper 사용
        job_manager.update_job(
            job_id,
            step="stt",
            message="음성 인식 중... (Whisper AI)",
            progress=35
        )

        try:
            transcript = await transcribe_audio(audio_path)
            transcript_source = "whisper"
            logger.info(f"[{job_id[:8]}] Whisper STT 사용")
        except TranscriptionError as e:
            raise Exception(f"음성 인식 실패: {e}")
    # else:  # 주석처리 - Whisper 성능 테스트용
    #     job_manager.update_job(
    #         job_id,
    #         message="YouTube 자막 사용!",
    #         progress=45
    #     )

    # 텍스트 유효성 최종 확인
    if not _is_valid_transcript(transcript):
        raise Exception(
            "영상에서 텍스트를 추출할 수 없습니다. "
            "음성이나 자막이 포함된 영상인지 확인해주세요."
        )

    job_manager.update_job(
        job_id,
        message="텍스트 추출 완료!",
        progress=50
    )

    return transcript, transcript_source


def _is_valid_transcript(transcript: Optional[Dict[str, Any]]) -> bool:
    """전사 결과가 유효한지 확인합니다."""
    if not transcript:
        return False
    full_text = transcript.get("full_text", "")
    return len(full_text) >= MIN_TRANSCRIPT_LENGTH


async def _step_parse_recipe(
    job_id: str,
    transcript: Dict[str, Any]
) -> Dict[str, Any]:
    """
    3단계: 레시피 파싱.

    Args:
        job_id: 작업 ID
        transcript: 전사 데이터

    Returns:
        레시피 딕셔너리

    Raises:
        Exception: 파싱 실패 시
    """
    job_manager.update_job(
        job_id,
        step="parsing",
        message="GPT-4o로 레시피 분석 중...",
        progress=55
    )

    try:
        recipe = await parse_recipe(transcript)
    except RecipeParseError as e:
        raise Exception(f"레시피 분석 실패: {e}")

    job_manager.update_job(
        job_id,
        message="레시피 분석 완료!",
        progress=90
    )

    return recipe


# =============================================================================
# 메인 처리 함수
# =============================================================================
async def process_video(job_id: str, url: str) -> None:
    """
    비동기로 영상을 처리합니다.

    Args:
        job_id: 작업 ID
        url: YouTube URL
    """
    job_dir = DATA_DIR / job_id
    job_dir.mkdir(exist_ok=True)

    timing: Dict[str, Any] = {}
    total_start = time.time()

    try:
        # 1단계: 영상 다운로드
        step_start = time.time()
        video_info = await _step_download(job_id, url, job_dir)
        timing["download"] = round(time.time() - step_start, 2)
        logger.info(f"[{job_id[:8]}] 다운로드 완료: {timing['download']}초")

        # 2단계: 자막/STT 추출
        step_start = time.time()
        transcript, transcript_source = await _step_extract_transcript(
            job_id, url, job_dir, video_info["audio_path"]
        )
        timing["transcript"] = round(time.time() - step_start, 2)
        timing["transcript_source"] = transcript_source
        logger.info(
            f"[{job_id[:8]}] 텍스트 추출 완료: {timing['transcript']}초 "
            f"(소스: {transcript_source})"
        )

        # 3단계: 레시피 파싱
        step_start = time.time()
        recipe = await _step_parse_recipe(job_id, transcript)
        timing["parsing"] = round(time.time() - step_start, 2)
        logger.info(f"[{job_id[:8]}] 레시피 파싱 완료: {timing['parsing']}초")

        # 총 소요 시간
        timing["total"] = round(time.time() - total_start, 2)
        logger.info(f"[{job_id[:8]}] === 전체 완료: {timing['total']}초 ===")

        # 결과 저장
        job_manager.update_job(
            job_id,
            step="done",
            message="레시피 추출 완료!",
            progress=100,
            status="completed",
            result={
                "recipe": recipe,
                "video_info": video_info,
                "transcript": transcript,
                "timing": timing
            }
        )

    except Exception as e:
        error_message = str(e)
        logger.error(f"[{job_id[:8]}] 처리 오류: {error_message}")

        job_manager.update_job(
            job_id,
            status="failed",
            message=f"오류 발생: {error_message}",
            progress=0
        )


# =============================================================================
# API 엔드포인트
# =============================================================================
@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_video(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks
) -> AnalyzeResponse:
    """
    YouTube URL을 받아 분석을 시작합니다.

    Args:
        request: 분석 요청 (YouTube URL 포함)
        background_tasks: 백그라운드 작업 큐

    Returns:
        작업 ID와 메시지
    """
    url = str(request.url).strip()

    if not url:
        raise HTTPException(
            status_code=400,
            detail="URL이 제공되지 않았습니다."
        )

    video_id = extract_video_id(url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않은 YouTube URL입니다."
        )

    job_id = str(uuid.uuid4())

    job_manager.create_job(job_id, url, video_id)
    logger.info(f"새 작업 생성: {job_id[:8]}, video_id={video_id}")

    background_tasks.add_task(process_video, job_id, url)

    return AnalyzeResponse(job_id=job_id, message="분석을 시작합니다.")


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    작업 상태를 조회합니다.

    Args:
        job_id: 작업 ID

    Returns:
        작업 상태 정보
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail="작업을 찾을 수 없습니다."
        )

    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        video_id=job.get("video_id")
    )


@router.get("/result/{job_id}")
async def get_result(job_id: str) -> Dict[str, Any]:
    """
    분석 결과를 조회합니다.

    Args:
        job_id: 작업 ID

    Returns:
        분석 결과 (레시피, 비디오 정보 등)
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail="작업을 찾을 수 없습니다."
        )

    if job["status"] == "failed":
        raise HTTPException(
            status_code=400,
            detail=job.get("message", "작업이 실패했습니다.")
        )

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="아직 처리가 완료되지 않았습니다."
        )

    return job["result"]


@router.delete("/job/{job_id}")
async def delete_job(job_id: str) -> Dict[str, str]:
    """
    작업을 삭제합니다.

    Args:
        job_id: 작업 ID

    Returns:
        삭제 결과
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail="작업을 찾을 수 없습니다."
        )

    job_manager.cleanup_job_files(job_id)
    job_manager.delete_job(job_id)
    logger.info(f"작업 삭제됨: {job_id[:8]}")

    return {"message": "작업이 삭제되었습니다.", "job_id": job_id}


@router.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """
    작업 통계를 조회합니다.

    Returns:
        작업 통계 정보
    """
    return job_manager.get_stats()
