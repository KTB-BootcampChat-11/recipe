"""
공통 유틸리티 함수 모듈.

여러 라우터와 서비스에서 공통으로 사용하는 헬퍼 함수들을 제공합니다.
"""
from typing import Any, Dict, Optional, TypeVar

from fastapi import HTTPException

T = TypeVar("T")


def get_or_404(
    collection: Dict[str, T],
    key: str,
    resource_name: str = "리소스"
) -> T:
    """
    딕셔너리에서 키로 값을 조회하고, 없으면 404 에러를 발생시킵니다.

    Args:
        collection: 조회할 딕셔너리
        key: 찾을 키
        resource_name: 에러 메시지에 표시할 리소스 이름

    Returns:
        찾은 값

    Raises:
        HTTPException: 키가 없을 때 404 에러
    """
    if key not in collection:
        raise HTTPException(
            status_code=404,
            detail=f"{resource_name}을(를) 찾을 수 없습니다."
        )
    return collection[key]


def format_timestamp(seconds: float) -> str:
    """
    초를 MM:SS 형식으로 변환합니다.

    Args:
        seconds: 초 단위 시간

    Returns:
        MM:SS 형식 문자열
    """
    if seconds < 0:
        seconds = 0

    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_duration(seconds: float) -> str:
    """
    초를 사람이 읽기 좋은 형식으로 변환합니다.

    Args:
        seconds: 초 단위 시간

    Returns:
        "X분 Y초" 또는 "X초" 형식 문자열
    """
    if seconds < 0:
        seconds = 0

    if seconds < 60:
        return f"{int(seconds)}초"

    minutes = int(seconds // 60)
    secs = int(seconds % 60)

    if secs == 0:
        return f"{minutes}분"
    return f"{minutes}분 {secs}초"


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    텍스트를 지정된 길이로 자릅니다.

    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        suffix: 잘렸을 때 붙일 접미사

    Returns:
        잘린 텍스트
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def safe_get(
    data: Dict[str, Any],
    key: str,
    default: Optional[Any] = None
) -> Any:
    """
    딕셔너리에서 안전하게 값을 가져옵니다.
    None 값도 기본값으로 대체합니다.

    Args:
        data: 딕셔너리
        key: 키
        default: 기본값

    Returns:
        값 또는 기본값
    """
    value = data.get(key)
    return value if value is not None else default


def calculate_progress(completed: int, total: int) -> int:
    """
    진행률을 퍼센트로 계산합니다.

    Args:
        completed: 완료된 수
        total: 전체 수

    Returns:
        0-100 사이의 퍼센트 값
    """
    if total <= 0:
        return 0
    return min(100, int((completed / total) * 100))
