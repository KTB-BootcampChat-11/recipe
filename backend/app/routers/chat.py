"""
요리 채팅방 라우터 모듈.

단계별 피드백을 제공하는 채팅 엔드포인트를 제공합니다.
"""
import base64
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL_GPT4O
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
client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# 상수
# =============================================================================
MAX_HISTORY_MESSAGES = 6
MAX_TOKENS = 500

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
    session_id = str(uuid.uuid4())[:8]
    recipe = request.recipe
    steps = recipe.get("steps", [])

    cooking_sessions[session_id] = {
        "recipe": recipe,
        "current_step": 1,
        "total_steps": len(steps),
        "steps": steps,
        "completed_steps": [],
        "chat_history": [],
        "ingredients": recipe.get("ingredients", [])
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

    # 이전 대화 히스토리 추가
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

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_GPT4O,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=0.7
        )

        reply = response.choices[0].message.content

        # 히스토리에 저장
        session["chat_history"].append({
            "role": "user",
            "content": request.message,
            "step_number": step_number,
            "has_image": bool(request.image_base64)
        })
        session["chat_history"].append({
            "role": "assistant",
            "content": reply,
            "step_number": step_number
        })

        session["current_step"] = step_number

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

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"AI 응답 생성 실패: {str(e)}"
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
