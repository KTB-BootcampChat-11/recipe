"""
통합 예외 클래스 모듈.

모든 서비스에서 사용하는 예외를 중앙 관리합니다.
"""
from typing import Optional


class RecipeAnalysisError(Exception):
    """레시피 분석 기본 예외."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class YouTubeDownloadError(RecipeAnalysisError):
    """YouTube 다운로드 실패."""

    pass


class SubtitleError(RecipeAnalysisError):
    """자막 처리 실패."""

    pass


class TranscriptionError(RecipeAnalysisError):
    """음성 인식 실패."""

    pass


class AudioFileError(TranscriptionError):
    """오디오 파일 관련 에러."""

    pass


class RecipeParseError(RecipeAnalysisError):
    """레시피 파싱 실패."""

    pass


class SessionNotFoundError(RecipeAnalysisError):
    """세션을 찾을 수 없음."""

    pass


class JobNotFoundError(RecipeAnalysisError):
    """작업을 찾을 수 없음."""

    pass
