"""
분석 관련 스키마 모듈.

영상 분석 API의 요청/응답 스키마를 정의합니다.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# =============================================================================
# 요청 스키마
# =============================================================================
class AnalyzeRequest(BaseModel):
    """영상 분석 요청."""

    url: str


# =============================================================================
# 응답 스키마
# =============================================================================
class AnalyzeResponse(BaseModel):
    """분석 시작 응답."""

    job_id: str
    message: str


class JobStatusResponse(BaseModel):
    """작업 상태 응답."""

    job_id: str
    status: str  # pending, processing, completed, failed
    progress: int  # 0-100
    message: str
    video_id: Optional[str] = None


# =============================================================================
# 레시피 관련 스키마
# =============================================================================
class Ingredient(BaseModel):
    """재료."""

    name: str
    amount: str
    unit: str
    note: Optional[str] = None


class RecipeStep(BaseModel):
    """조리 단계."""

    step_number: int
    instruction: str
    timestamp: Optional[int] = None
    duration: Optional[str] = None
    tips: Optional[str] = None


class Recipe(BaseModel):
    """레시피."""

    title: str
    description: Optional[str] = None
    servings: Optional[str] = None
    total_time: Optional[str] = None
    difficulty: Optional[str] = None
    ingredients: List[Ingredient] = []
    steps: List[RecipeStep] = []
    tips: Optional[List[str]] = None


# =============================================================================
# 영상/음성 관련 스키마
# =============================================================================
class VideoInfo(BaseModel):
    """영상 정보."""

    video_id: str
    title: str
    duration: float
    url: Optional[str] = None


class TranscriptSegment(BaseModel):
    """음성 인식 세그먼트."""

    start: float
    end: float
    text: str


class Transcript(BaseModel):
    """음성 인식 결과."""

    full_text: str
    segments: List[TranscriptSegment] = []
    language: Optional[str] = "ko"
    duration: Optional[float] = None


# =============================================================================
# 처리 결과 스키마
# =============================================================================
class Timing(BaseModel):
    """처리 시간."""

    download: Optional[float] = None
    transcript: Optional[float] = None
    parsing: Optional[float] = None
    total: Optional[float] = None


class AnalysisResult(BaseModel):
    """분석 결과."""

    recipe: Dict[str, Any]
    video_info: Dict[str, Any]
    transcript: Dict[str, Any]
    timing: Optional[Dict[str, Any]] = None
