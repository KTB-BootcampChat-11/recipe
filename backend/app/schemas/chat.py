"""
채팅 관련 스키마 모듈.

요리 채팅방 API의 요청/응답 스키마를 정의합니다.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class StartSessionRequest(BaseModel):
    """세션 시작 요청."""

    recipe: Dict[str, Any]


class StartSessionResponse(BaseModel):
    """세션 시작 응답."""

    session_id: str
    message: str
    total_steps: int


class ChatMessage(BaseModel):
    """채팅 메시지."""

    role: str
    content: str
    step_number: Optional[int] = None
    image_url: Optional[str] = None


class ChatRequest(BaseModel):
    """채팅 요청."""

    session_id: str
    step_number: int
    message: str
    image_base64: Optional[str] = None


class ChatResponse(BaseModel):
    """채팅 응답."""

    reply: str
    step_info: Dict[str, Any]
    session_status: Dict[str, Any]


class SessionStatus(BaseModel):
    """세션 상태."""

    session_id: str
    recipe_title: str
    current_step: int
    total_steps: int
    completed_steps: List[int]
    progress_percent: int
