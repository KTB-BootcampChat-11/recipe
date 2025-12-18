import json
from datetime import datetime
from app.config import LOG_DIR


def save_log(log_type: str, data: dict, video_id: str = None) -> str:
    """
    STT/LLM 응답을 JSON 파일로 저장

    Args:
        log_type: 로그 타입 (stt, llm, llm_from_stt, full)
        data: 저장할 데이터
        video_id: 비디오 ID (optional)

    Returns:
        저장된 파일 경로
    """
    # 로그 디렉토리 없으면 생성
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{video_id}_" if video_id else ""
    filename = f"{prefix}{log_type}_{timestamp}.json"
    filepath = LOG_DIR / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return str(filepath)
