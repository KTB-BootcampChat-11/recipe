"""
레시피 파싱 서비스 모듈.

GPT-4o를 사용하여 음성 텍스트를 구조화된 레시피로 변환합니다.
"""
import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from app.config import (
    MIN_TRANSCRIPT_LENGTH,
    OPENAI_API_KEY,
    OPENAI_MODEL_GPT4O,
)
from app.exceptions import RecipeParseError
from app.prompts import RECIPE_PARSE_PROMPT

# =============================================================================
# 로깅 및 클라이언트 설정
# =============================================================================
logger = logging.getLogger(__name__)

# 타임아웃 설정 (LLM 응답 대기 시간 고려)
http_client = httpx.Client(
    timeout=httpx.Timeout(connect=30.0, read=180.0, write=30.0, pool=30.0)
)
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

# =============================================================================
# 상수
# =============================================================================
MAX_TRANSCRIPT_LENGTH = 4000  # 토큰 제한 (한글 1자 ≈ 2토큰)
API_TIMEOUT = 180  # 초 (LLM 응답 대기)


# =============================================================================
# 헬퍼 함수
# =============================================================================
def _clean_json_response(text: str) -> str:
    """
    GPT 응답에서 JSON 부분만 추출합니다.

    Args:
        text: GPT 응답 텍스트

    Returns:
        정제된 JSON 문자열
    """
    pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _validate_recipe_data(recipe_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    레시피 데이터 유효성 검사 및 기본값 설정.

    Args:
        recipe_data: 원본 레시피 데이터

    Returns:
        유효성 검사된 레시피 데이터
    """
    defaults = {
        "title": "레시피",
        "description": "",
        "servings": "1인분",
        "total_time": "",
        "difficulty": "보통",
        "ingredients": [],
        "steps": [],
        "tips": []
    }

    for key, default_value in defaults.items():
        if key not in recipe_data or recipe_data[key] is None:
            recipe_data[key] = default_value

    recipe_data["steps"] = _validate_steps(recipe_data.get("steps", []))
    recipe_data["ingredients"] = _validate_ingredients(
        recipe_data.get("ingredients", [])
    )

    return recipe_data


def _validate_steps(steps: List[Any]) -> List[Dict[str, Any]]:
    """
    조리 단계 리스트 유효성 검사.

    Args:
        steps: 원본 조리 단계 리스트

    Returns:
        유효성 검사된 조리 단계 리스트
    """
    validated_steps = []

    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue

        # instruction에서 문자열 '\n'을 실제 줄바꿈으로 변환
        instruction = step.get("instruction", "")
        if isinstance(instruction, str):
            instruction = instruction.replace("\\n", "\n")

        validated_step = {
            "step_number": step.get("step_number", i + 1),
            "instruction": instruction,
            "timestamp": max(0, float(step.get("timestamp", 0))),
            "duration": step.get("duration", ""),
            "details": step.get("details", ""),
            "tips": step.get("tips", "")
        }
        validated_steps.append(validated_step)

    return validated_steps


def _validate_ingredients(ingredients: List[Any]) -> List[Dict[str, str]]:
    """
    재료 리스트 유효성 검사.

    Args:
        ingredients: 원본 재료 리스트

    Returns:
        유효성 검사된 재료 리스트
    """
    validated_ingredients = []

    for ing in ingredients:
        if not isinstance(ing, dict):
            continue

        validated_ing = {
            "name": str(ing.get("name", "")),
            "amount": str(ing.get("amount", "")),
            "unit": str(ing.get("unit", "")),
            "note": str(ing.get("note", ""))
        }

        if validated_ing["name"]:
            validated_ingredients.append(validated_ing)

    return validated_ingredients


def _build_user_message(
    full_text: str,
    segments: List[Dict[str, Any]]
) -> str:
    """
    GPT에 전달할 사용자 메시지를 구성합니다.

    Args:
        full_text: 전체 전사 텍스트
        segments: 타임스탬프가 포함된 세그먼트 리스트

    Returns:
        구성된 사용자 메시지
    """
    segments_text = ""
    for seg in segments:
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        text = seg.get("text", "")
        segments_text += f"[{start:.1f}s - {end:.1f}s]: {text}\n"

    return f"""다음은 요리 영상의 음성 텍스트입니다:

## 전체 텍스트
{full_text}

## 타임스탬프별 세그먼트
{segments_text}

이 내용을 분석하여 구조화된 레시피 JSON을 생성해주세요."""


def _create_empty_recipe(
    title: str = "레시피",
    description: str = "",
    raw_text: str = ""
) -> Dict[str, Any]:
    """
    빈 레시피 구조를 생성합니다.

    Args:
        title: 레시피 제목
        description: 설명
        raw_text: 원본 텍스트

    Returns:
        빈 레시피 구조
    """
    return _validate_recipe_data({
        "title": title,
        "description": description,
        "ingredients": [],
        "steps": [],
        "raw_text": raw_text
    })


# =============================================================================
# API 호출
# =============================================================================
def _call_gpt_api(user_message: str) -> Dict[str, Any]:
    """
    GPT API를 호출하여 레시피를 파싱합니다.

    Args:
        user_message: 사용자 메시지

    Returns:
        파싱된 레시피 데이터

    Raises:
        RecipeParseError: API 응답이 유효하지 않은 경우
        json.JSONDecodeError: JSON 파싱 실패 시
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL_GPT4O,
        messages=[
            {"role": "system", "content": RECIPE_PARSE_PROMPT},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        timeout=API_TIMEOUT
    )

    if not response.choices:
        raise RecipeParseError("GPT 응답에 choices가 없습니다")

    result_text = response.choices[0].message.content

    if not result_text:
        raise RecipeParseError("GPT 응답이 비어있습니다")

    cleaned_text = _clean_json_response(result_text)
    return json.loads(cleaned_text)


# =============================================================================
# 메인 파싱 함수
# =============================================================================
async def parse_recipe(
    transcript_data: Dict[str, Any],
    max_retries: int = 2
) -> Dict[str, Any]:
    """
    GPT-4o를 사용하여 음성 텍스트를 구조화된 레시피로 변환합니다.

    Args:
        transcript_data: 전사 데이터
            - full_text: 전체 텍스트
            - segments: 세그먼트 리스트

        max_retries: API 호출 실패 시 재시도 횟수

    Returns:
        구조화된 레시피 데이터

    Raises:
        RecipeParseError: 파싱 실패 시 (내부적으로 처리됨)
    """
    full_text = transcript_data.get("full_text", "").strip()

    # 입력 검증
    if not full_text or len(full_text) < MIN_TRANSCRIPT_LENGTH:
        logger.warning(
            f"전사 텍스트가 너무 짧습니다: {len(full_text)}자 "
            f"(최소 {MIN_TRANSCRIPT_LENGTH}자)"
        )
        return _create_empty_recipe(
            description="음성 인식 결과가 너무 짧습니다."
        )

    # 토큰 제한 체크
    if len(full_text) > MAX_TRANSCRIPT_LENGTH:
        logger.warning(
            f"전사 텍스트가 너무 깁니다: {len(full_text)}자, "
            f"{MAX_TRANSCRIPT_LENGTH}자로 자름"
        )
        full_text = full_text[:MAX_TRANSCRIPT_LENGTH]

    segments = transcript_data.get("segments", [])
    user_message = _build_user_message(full_text, segments)

    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            print(f"[LLM] 레시피 파싱 시도 {attempt + 1}/{max_retries + 1}, 모델: {OPENAI_MODEL_GPT4O}")
            logger.info(
                f"[LLM] 레시피 파싱 시도 {attempt + 1}/{max_retries + 1}, "
                f"모델: {OPENAI_MODEL_GPT4O}"
            )

            recipe_data = _call_gpt_api(user_message)
            recipe_data = _validate_recipe_data(recipe_data)

            print(f"[LLM] 레시피 파싱 성공: {recipe_data.get('title', 'unknown')}")
            logger.info(
                f"[LLM] 레시피 파싱 성공: {recipe_data.get('title', 'unknown')}"
            )
            return recipe_data

        except json.JSONDecodeError as e:
            last_error = e
            logger.warning(f"[LLM] JSON 파싱 실패 (시도 {attempt + 1}): {e}")

        except RateLimitError as e:
            last_error = e
            logger.error(f"[LLM] API 할당량 초과: {e}")
            break

        except APIConnectionError as e:
            last_error = e
            print(f"[LLM] API 연결 오류 (시도 {attempt + 1}): {e}")
            logger.warning(f"[LLM] API 연결 오류 (시도 {attempt + 1}): {e}")

        except APIError as e:
            last_error = e
            logger.error(f"[LLM] OpenAI API 오류: {e}")
            break

        except RecipeParseError as e:
            last_error = e
            logger.warning(f"[LLM] 레시피 파싱 오류 (시도 {attempt + 1}): {e}")

        except Exception as e:
            last_error = e
            logger.error(f"[LLM] 예상치 못한 오류 (시도 {attempt + 1}): {e}")

    # 모든 재시도 실패 시 기본 구조 반환
    logger.error(f"[LLM] 레시피 파싱 최종 실패: {last_error}")

    error_description = str(last_error)[:100] if last_error else "알 수 없는 오류"

    return _create_empty_recipe(
        title="레시피 (파싱 실패)",
        description=f"레시피 분석 중 오류가 발생했습니다: {error_description}",
        raw_text=full_text
    )
