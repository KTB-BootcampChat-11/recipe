from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {"status": "healthy"}
