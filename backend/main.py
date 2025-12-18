"""
애플리케이션 진입점.

실행: uvicorn main:app --reload
"""
import logging

from app.main import app

# =============================================================================
# 로그 필터링
# =============================================================================


class StatusEndpointFilter(logging.Filter):
    """상태 폴링 엔드포인트 로그 필터."""

    def filter(self, record: logging.LogRecord) -> bool:
        """'/api/status/' 요청은 로그에서 제외합니다."""
        return "/api/status/" not in record.getMessage()


# uvicorn access 로거에 필터 적용
logging.getLogger("uvicorn.access").addFilter(StatusEndpointFilter())

# =============================================================================
# 메인 실행
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
