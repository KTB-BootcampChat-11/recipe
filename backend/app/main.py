"""
FastAPI 애플리케이션 모듈.

쇼츠 레시피 정리기 API 서버를 구성합니다.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CORS_ORIGINS, DATA_DIR
from app.routers import (
    analyze_router,
    chat_router,
    health_router,
    test_router,
)


# =============================================================================
# 앱 생명주기 관리
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 생명주기를 관리합니다."""
    # Startup
    DATA_DIR.mkdir(exist_ok=True)
    yield
    # Shutdown (cleanup if needed)


# =============================================================================
# 앱 팩토리
# =============================================================================
def create_app() -> FastAPI:
    """FastAPI 앱을 생성합니다."""
    application = FastAPI(
        title="쇼츠 레시피 정리기",
        description="YouTube 쇼츠에서 레시피를 추출하고 정리합니다",
        version="1.0.0",
        lifespan=lifespan
    )

    # CORS 설정
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 라우터 등록
    application.include_router(health_router)
    application.include_router(analyze_router)
    application.include_router(test_router)
    application.include_router(chat_router)

    return application


# =============================================================================
# 앱 인스턴스
# =============================================================================
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
