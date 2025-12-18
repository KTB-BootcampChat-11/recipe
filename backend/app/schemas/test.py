"""
테스트 관련 스키마 모듈.

테스트 API의 요청/응답 스키마를 정의합니다.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel


# =============================================================================
# 요청 스키마
# =============================================================================
class TestURLRequest(BaseModel):
    """URL 기반 테스트 요청 (모든 테스트 공통)."""

    url: str


class TestLLMRequest(BaseModel):
    """LLM 테스트 요청 (텍스트 직접 입력)."""

    text: str


class TestFromVideoIdRequest(BaseModel):
    """이전 단계 결과를 video_id로 참조하는 요청."""

    video_id: str


# =============================================================================
# 응답 스키마
# =============================================================================
class TestResponse(BaseModel):
    """테스트 응답 기본."""

    success: bool
    elapsed_time: Optional[float] = None
    error: Optional[str] = None
    video_id: Optional[str] = None


class TestDownloadResponse(TestResponse):
    """다운로드 테스트 응답."""

    video_info: Optional[Dict[str, Any]] = None


class TestSubtitleResponse(TestResponse):
    """자막 테스트 응답."""

    timing: Optional[Dict[str, Any]] = None
    subtitle_info: Optional[Dict[str, Any]] = None
    transcript: Optional[Dict[str, Any]] = None


class TestSTTResponse(TestResponse):
    """STT 테스트 응답."""

    timing: Optional[Dict[str, Any]] = None
    transcript: Optional[Dict[str, Any]] = None


class TestTranscriptResponse(TestResponse):
    """자막+STT 통합 테스트 응답."""

    timing: Optional[Dict[str, Any]] = None
    source: Optional[str] = None  # 'youtube_ko_auto', 'whisper' 등
    transcript: Optional[Dict[str, Any]] = None


class TestLLMResponse(TestResponse):
    """LLM 테스트 응답."""

    timing: Optional[Dict[str, Any]] = None
    recipe: Optional[Dict[str, Any]] = None
    input_segments_count: Optional[int] = None


class TestFullResponse(TestResponse):
    """전체 파이프라인 테스트 응답."""

    timing: Optional[Dict[str, Any]] = None
    summary: Optional[Dict[str, Any]] = None
    transcript: Optional[Dict[str, Any]] = None
    recipe: Optional[Dict[str, Any]] = None
    traceback: Optional[str] = None


# =============================================================================
# 호환성을 위한 별칭
# =============================================================================
TestSTTRequest = TestURLRequest
TestLLMFromSTTRequest = TestFromVideoIdRequest
