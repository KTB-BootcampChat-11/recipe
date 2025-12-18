"""
요리 채팅방 라우터 모듈.

단계별 피드백을 제공하는 채팅 엔드포인트를 제공합니다.
"""
import base64
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from app.config import OPENAI_API_KEY, OPENAI_MODEL_CHAT
from app.prompts import COOKING_ASSISTANT_PROMPT
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SessionStatus,
    StartSessionRequest,
    StartSessionResponse,
)

# =============================================================================
# 라우터 및 클라이언트 설정
# =============================================================================
router = APIRouter(prefix="/api/chat", tags=["Chat"])

# 타임아웃 설정 (채팅 응답 대기 시간 고려)
http_client = httpx.Client(
    timeout=httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)
)
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

# =============================================================================
# 상수
# =============================================================================
MAX_HISTORY_MESSAGES = 6
MAX_TOKENS = 500
MAX_IMAGE_SIZE_MB = 10  # 최대 이미지 크기 (MB)
MAX_IMAGE_SIZE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024
SESSION_EXPIRY_SECONDS = 3600  # 세션 만료 시간 (1시간)
API_MAX_RETRIES = 2  # API 호출 재시도 횟수

# =============================================================================
# 세션 저장소 (메모리)
# =============================================================================
cooking_sessions: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# 헬퍼 함수
# =============================================================================
def _get_session(session_id: str) -> Dict[str, Any]:
    """세션을 조회하고, 없으면 404 에러를 발생시킵니다."""
    if session_id not in cooking_sessions:
        raise HTTPException(
            status_code=404,
            detail="세션을 찾을 수 없습니다."
        )
    return cooking_sessions[session_id]


def _validate_step_number(step_number: int, total_steps: int) -> None:
    """단계 번호 유효성을 검사합니다."""
    if step_number < 1 or step_number > total_steps:
        raise HTTPException(
            status_code=400,
            detail="잘못된 단계 번호입니다."
        )


def _build_system_prompt(
    recipe: Dict[str, Any],
    step: Dict[str, Any],
    step_number: int,
    total_steps: int
) -> str:
    """시스템 프롬프트를 구성합니다."""
    return COOKING_ASSISTANT_PROMPT.format(
        recipe_title=recipe.get("title", "요리"),
        step_number=step_number,
        instruction=step.get("instruction", ""),
        tips=step.get("tips", "없음"),
        difficulty=recipe.get("difficulty", "보통"),
        total_steps=total_steps
    )


def _build_user_content(
    message: str,
    step_number: int,
    image_base64: Optional[str] = None
) -> List[Dict[str, Any]]:
    """사용자 메시지 콘텐츠를 구성합니다."""
    user_content: List[Dict[str, Any]] = []

    if image_base64:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{image_base64}"
            }
        })

    user_content.append({
        "type": "text",
        "text": f"[Step {step_number} 진행 중] {message}"
    })

    return user_content


def _calculate_progress(completed: int, total: int) -> int:
    """진행률을 계산합니다."""
    if total <= 0:
        return 0
    return int((completed / total) * 100)


def _validate_image_size(image_bytes: bytes) -> None:
    """이미지 크기를 검증합니다."""
    if len(image_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"이미지 크기가 너무 큽니다. 최대 {MAX_IMAGE_SIZE_MB}MB까지 허용됩니다."
        )


def _cleanup_expired_sessions() -> None:
    """만료된 세션을 정리합니다."""
    current_time = time.time()
    expired_sessions = [
        sid for sid, session in cooking_sessions.items()
        if current_time - session.get("created_at", 0) > SESSION_EXPIRY_SECONDS
    ]
    for sid in expired_sessions:
        cooking_sessions.pop(sid, None)


def _call_chat_api(
    messages: List[Dict[str, Any]],
    max_retries: int = API_MAX_RETRIES
) -> str:
    """
    OpenAI Chat API를 호출합니다. 재시도 로직 포함.

    Args:
        messages: 메시지 리스트
        max_retries: 최대 재시도 횟수

    Returns:
        AI 응답 텍스트

    Raises:
        HTTPException: API 호출 실패 시
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL_CHAT,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=0.7
            )

            if not response.choices:
                raise APIError("응답에 choices가 없습니다", response=None, body=None)

            return response.choices[0].message.content or ""

        except RateLimitError as e:
            last_error = e
            # Rate limit은 재시도하지 않음
            break

        except APIConnectionError as e:
            last_error = e
            if attempt < max_retries:
                continue

        except APIError as e:
            last_error = e
            if attempt < max_retries:
                continue

        except Exception as e:
            last_error = e
            break

    error_msg = str(last_error)[:100] if last_error else "알 수 없는 오류"
    raise HTTPException(
        status_code=500,
        detail=f"AI 응답 생성 실패: {error_msg}"
    )


# =============================================================================
# 세션 관리 API
# =============================================================================
@router.post("/start", response_model=StartSessionResponse)
async def start_cooking_session(
    request: StartSessionRequest
) -> StartSessionResponse:
    """
    요리 세션을 시작합니다.

    레시피 정보를 받아서 채팅 세션을 생성합니다.
    """
    # 만료된 세션 정리
    _cleanup_expired_sessions()

    # 전체 UUID 사용으로 충돌 방지
    session_id = str(uuid.uuid4())
    recipe = request.recipe
    steps = recipe.get("steps", [])

    cooking_sessions[session_id] = {
        "recipe": recipe,
        "current_step": 1,
        "total_steps": len(steps),
        "steps": steps,
        "completed_steps": [],
        "chat_history": [],
        "ingredients": recipe.get("ingredients", []),
        "created_at": time.time()  # 세션 생성 시간 기록
    }

    return StartSessionResponse(
        session_id=session_id,
        message=f"'{recipe.get('title', '요리')}' 세션이 시작되었습니다!",
        total_steps=len(steps)
    )


@router.get("/session/{session_id}", response_model=SessionStatus)
async def get_session_status(session_id: str) -> SessionStatus:
    """세션 상태를 조회합니다."""
    session = _get_session(session_id)

    completed = len(session["completed_steps"])
    total = session["total_steps"]

    return SessionStatus(
        session_id=session_id,
        recipe_title=session["recipe"].get("title", "요리"),
        current_step=session["current_step"],
        total_steps=total,
        completed_steps=session["completed_steps"],
        progress_percent=_calculate_progress(completed, total)
    )


@router.get("/session/{session_id}/step/{step_number}")
async def get_step_detail(
    session_id: str,
    step_number: int
) -> Dict[str, Any]:
    """특정 단계의 상세 정보를 조회합니다."""
    session = _get_session(session_id)
    steps = session["steps"]

    _validate_step_number(step_number, len(steps))

    step = steps[step_number - 1]

    return {
        "step_number": step_number,
        "instruction": step.get("instruction", ""),
        "tips": step.get("tips", ""),
        "duration": step.get("duration", ""),
        "timestamp": step.get("timestamp", 0),
        "is_completed": step_number in session["completed_steps"],
        "is_current": step_number == session["current_step"]
    }


@router.post("/session/{session_id}/complete-step/{step_number}")
async def complete_step(
    session_id: str,
    step_number: int
) -> Dict[str, Any]:
    """단계를 완료 처리합니다."""
    session = _get_session(session_id)

    if step_number not in session["completed_steps"]:
        session["completed_steps"].append(step_number)

    if step_number < session["total_steps"]:
        session["current_step"] = step_number + 1

    is_finished = len(session["completed_steps"]) == session["total_steps"]

    return {
        "message": f"Step {step_number} 완료!",
        "next_step": session["current_step"],
        "is_finished": is_finished
    }


# =============================================================================
# 채팅 API
# =============================================================================
@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest) -> ChatResponse:
    """
    채팅 메시지를 보내고 AI 응답을 받습니다.

    이미지가 포함되면 GPT-4o Vision으로 분석합니다.
    """
    # 이미지 크기 검증
    if request.image_base64:
        image_bytes = base64.b64decode(request.image_base64)
        _validate_image_size(image_bytes)

    session = _get_session(request.session_id)
    steps = session["steps"]
    step_number = request.step_number

    _validate_step_number(step_number, len(steps))

    step = steps[step_number - 1]
    recipe = session["recipe"]

    # 시스템 프롬프트 구성
    system_prompt = _build_system_prompt(
        recipe, step, step_number, len(steps)
    )

    # 메시지 구성
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt}
    ]

    # 이전 대화 히스토리 추가 (이미지 컨텍스트 포함)
    for msg in session["chat_history"][-MAX_HISTORY_MESSAGES:]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    # 현재 메시지 구성
    user_content = _build_user_content(
        request.message,
        step_number,
        request.image_base64
    )
    messages.append({"role": "user", "content": user_content})

    # 재시도 로직이 포함된 API 호출
    reply = _call_chat_api(messages)

    # 히스토리에 저장 (이미지 포함 시 멀티모달 콘텐츠로 저장)
    if request.image_base64:
        # 이미지가 있으면 멀티모달 형식으로 저장하여 컨텍스트 유지
        session["chat_history"].append({
            "role": "user",
            "content": user_content,  # 멀티모달 콘텐츠 그대로 저장
            "step_number": step_number,
            "has_image": True
        })
    else:
        session["chat_history"].append({
            "role": "user",
            "content": f"[Step {step_number} 진행 중] {request.message}",
            "step_number": step_number,
            "has_image": False
        })

    session["chat_history"].append({
        "role": "assistant",
        "content": reply,
        "step_number": step_number
    })

    session["current_step"] = step_number

    # 세션 활동 시간 갱신
    session["created_at"] = time.time()

    completed = len(session["completed_steps"])
    total = session["total_steps"]

    return ChatResponse(
        reply=reply,
        step_info={
            "step_number": step_number,
            "instruction": step.get("instruction", ""),
            "tips": step.get("tips", ""),
            "is_completed": step_number in session["completed_steps"]
        },
        session_status={
            "current_step": session["current_step"],
            "completed_steps": session["completed_steps"],
            "progress_percent": _calculate_progress(completed, total)
        }
    )


@router.post("/message-with-image")
async def send_message_with_image(
    session_id: str = Form(...),
    step_number: int = Form(...),
    message: str = Form(...),
    image: Optional[UploadFile] = File(None)
) -> ChatResponse:
    """
    이미지 파일과 함께 메시지를 보냅니다.

    multipart/form-data 형식을 사용합니다.
    """
    image_base64 = None

    if image:
        contents = await image.read()
        # 이미지 크기 검증
        _validate_image_size(contents)
        image_base64 = base64.b64encode(contents).decode("utf-8")

    request = ChatRequest(
        session_id=session_id,
        step_number=step_number,
        message=message,
        image_base64=image_base64
    )

    return await send_message(request)


@router.get("/session/{session_id}/history")
async def get_chat_history(session_id: str) -> Dict[str, Any]:
    """채팅 히스토리를 조회합니다."""
    session = _get_session(session_id)

    return {
        "session_id": session_id,
        "recipe_title": session["recipe"].get("title", ""),
        "messages": session["chat_history"]
    }


@router.delete("/session/{session_id}")
async def end_session(session_id: str) -> Dict[str, Any]:
    """세션을 종료합니다."""
    session = _get_session(session_id)
    cooking_sessions.pop(session_id)

    return {
        "message": "세션이 종료되었습니다.",
        "summary": {
            "recipe": session["recipe"].get("title", ""),
            "completed_steps": len(session["completed_steps"]),
            "total_steps": session["total_steps"],
            "total_messages": len(session["chat_history"])
        }
    }
