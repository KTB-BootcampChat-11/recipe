"""
테스트 라우터 모듈.

개발 및 디버깅용 API 엔드포인트를 제공합니다.
"""
import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from app.config import DATA_DIR
from app.schemas.test import (
    TestFromVideoIdRequest,
    TestLLMRequest,
    TestURLRequest,
)
from app.utils.logger import save_log
from app.services.recipe_parser import parse_recipe
from app.services.transcribe import transcribe_audio
from app.services.youtube import (
    download_subtitles,
    download_video,
    extract_video_id,
    parse_json3_subtitles,
)

# =============================================================================
# 라우터 설정
# =============================================================================
router = APIRouter(prefix="/api/test", tags=["Test"])

# =============================================================================
# 상수
# =============================================================================
MIN_TRANSCRIPT_LENGTH = 20
TEXT_PREVIEW_LENGTH = 500


# =============================================================================
# 캐시 헬퍼 함수
# =============================================================================
def _get_test_dir(video_id: str) -> Path:
    """테스트 디렉토리 경로를 반환합니다."""
    return DATA_DIR / f"test_{video_id}"


def _load_cached_result(video_id: str, stage: str) -> Optional[Dict[str, Any]]:
    """저장된 단계별 결과를 로드합니다."""
    cache_file = _get_test_dir(video_id) / f"{stage}_result.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _save_cached_result(
    video_id: str,
    stage: str,
    result: Dict[str, Any]
) -> None:
    """단계별 결과를 저장합니다."""
    test_dir = _get_test_dir(video_id)
    test_dir.mkdir(exist_ok=True)
    cache_file = test_dir / f"{stage}_result.json"
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def _truncate_text(text: str, max_length: int = TEXT_PREVIEW_LENGTH) -> str:
    """텍스트를 지정된 길이로 자릅니다."""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text


def _format_transcript_preview(
    transcript: Dict[str, Any]
) -> Dict[str, Any]:
    """전사 결과 미리보기 형식으로 변환합니다."""
    full_text = transcript.get("full_text", "")
    return {
        "full_text": _truncate_text(full_text),
        "full_text_length": len(full_text),
        "language": transcript.get("language"),
        "duration": transcript.get("duration"),
        "segments_count": len(transcript.get("segments", [])),
        "segments_preview": transcript.get("segments", [])[:5]
    }


# =============================================================================
# 1단계: 다운로드
# =============================================================================
@router.post("/download")
async def test_download(request: TestURLRequest) -> Dict[str, Any]:
    """
    1단계: YouTube 다운로드 테스트.

    - 입력: YouTube URL
    - 출력: video_id, video_path, audio_path
    - 다음 단계: /subtitle, /stt, /transcript
    """
    start = time.time()

    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않은 YouTube URL입니다."
        )

    test_dir = _get_test_dir(video_id)
    test_dir.mkdir(exist_ok=True)

    try:
        video_info = await download_video(request.url, str(test_dir))
        elapsed = round(time.time() - start, 2)

        result = {
            "video_id": video_info.get("video_id"),
            "title": video_info.get("title"),
            "duration": video_info.get("duration"),
            "video_path": video_info.get("video_path"),
            "audio_path": video_info.get("audio_path"),
            "url": request.url
        }

        _save_cached_result(video_id, "download", result)

        return {
            "success": True,
            "elapsed_time": elapsed,
            "video_id": video_id,
            "video_info": result,
            "next_steps": [
                f"POST /api/test/subtitle (video_id: {video_id})",
                f"POST /api/test/stt (video_id: {video_id})",
                f"POST /api/test/transcript (video_id: {video_id})"
            ]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": round(time.time() - start, 2)
        }


# =============================================================================
# 2단계: 텍스트 추출 (자막/STT)
# =============================================================================
@router.post("/subtitle")
async def test_subtitle(request: TestURLRequest) -> Dict[str, Any]:
    """
    2단계-A: YouTube 자막 테스트 (자막만 시도).

    - 입력: YouTube URL
    - 출력: 자막 정보, transcript
    """
    start = time.time()
    timing: Dict[str, Any] = {}

    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않은 YouTube URL입니다."
        )

    test_dir = _get_test_dir(video_id)
    test_dir.mkdir(exist_ok=True)

    try:
        # 자막 다운로드
        subtitle_start = time.time()
        subtitle_info = await download_subtitles(request.url, str(test_dir))
        timing["subtitle_download"] = round(time.time() - subtitle_start, 2)

        if not subtitle_info:
            return {
                "success": False,
                "video_id": video_id,
                "error": "사용 가능한 자막이 없습니다",
                "timing": timing,
                "suggestion": "POST /api/test/stt 로 Whisper STT를 시도하세요"
            }

        # 자막 파싱
        parse_start = time.time()
        transcript = parse_json3_subtitles(subtitle_info["subtitle_path"])
        timing["subtitle_parse"] = round(time.time() - parse_start, 2)
        timing["total"] = round(time.time() - start, 2)

        if not transcript:
            return {
                "success": False,
                "video_id": video_id,
                "error": "자막 파싱 실패",
                "subtitle_info": subtitle_info,
                "timing": timing
            }

        source = f"youtube_{subtitle_info['language']}"
        if subtitle_info["is_auto_generated"]:
            source += "_auto"

        _save_cached_result(video_id, "transcript", {
            "source": source,
            **transcript
        })

        return {
            "success": True,
            "video_id": video_id,
            "timing": timing,
            "subtitle_info": {
                "language": subtitle_info["language"],
                "is_auto_generated": subtitle_info["is_auto_generated"],
                "path": subtitle_info["subtitle_path"]
            },
            "transcript": _format_transcript_preview(transcript),
            "next_step": f"POST /api/test/llm with video_id: {video_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "elapsed_time": round(time.time() - start, 2)
        }


@router.post("/stt")
async def test_stt(request: TestURLRequest) -> Dict[str, Any]:
    """
    2단계-B: Whisper STT 테스트 (다운로드 + STT).

    - 입력: YouTube URL
    - 출력: transcript
    """
    start = time.time()
    timing: Dict[str, Any] = {}

    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않은 YouTube URL입니다."
        )

    test_dir = _get_test_dir(video_id)
    test_dir.mkdir(exist_ok=True)

    try:
        # 캐시된 다운로드 결과 확인
        cached_download = _load_cached_result(video_id, "download")

        if cached_download and cached_download.get("audio_path"):
            audio_path = cached_download["audio_path"]
            timing["download"] = "cached"
        else:
            download_start = time.time()
            video_info = await download_video(request.url, str(test_dir))
            timing["download"] = round(time.time() - download_start, 2)
            audio_path = video_info["audio_path"]
            _save_cached_result(video_id, "download", video_info)

        # STT
        stt_start = time.time()
        transcript = await transcribe_audio(audio_path)
        timing["stt"] = round(time.time() - stt_start, 2)
        timing["total"] = round(time.time() - start, 2)

        _save_cached_result(video_id, "transcript", {
            "source": "whisper",
            **transcript
        })

        result = {
            "success": True,
            "video_id": video_id,
            "timing": timing,
            "transcript": _format_transcript_preview(transcript),
            "next_step": f"POST /api/test/llm with video_id: {video_id}"
        }

        save_log("stt", result, video_id)
        return result
    except Exception as e:
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "elapsed_time": round(time.time() - start, 2)
        }


@router.post("/transcript")
async def test_transcript(request: TestURLRequest) -> Dict[str, Any]:
    """
    2단계-C: 자막+STT 통합 테스트 (자막 우선, STT 폴백).

    - 입력: YouTube URL
    - 출력: transcript (자막 또는 STT)
    - 실제 서비스와 동일한 로직
    """
    start = time.time()
    timing: Dict[str, Any] = {}

    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않은 YouTube URL입니다."
        )

    test_dir = _get_test_dir(video_id)
    test_dir.mkdir(exist_ok=True)

    transcript = None
    source = None

    try:
        # # 1. YouTube 자막 시도 (주석처리 - Whisper 성능 테스트용)
        # subtitle_start = time.time()
        # try:
        #     subtitle_info = await download_subtitles(request.url, str(test_dir))
        #     if subtitle_info:
        #         transcript = parse_json3_subtitles(subtitle_info["subtitle_path"])
        #         full_text = transcript.get("full_text", "") if transcript else ""
        #         if transcript and len(full_text) >= MIN_TRANSCRIPT_LENGTH:
        #             source = f"youtube_{subtitle_info['language']}"
        #             if subtitle_info["is_auto_generated"]:
        #                 source += "_auto"
        # except Exception:
        #     pass
        # timing["subtitle_attempt"] = round(time.time() - subtitle_start, 2)

        # 2. Whisper STT 사용 (자막 로직 비활성화)
        if True:  # 항상 Whisper 사용
            cached_download = _load_cached_result(video_id, "download")
            if cached_download and cached_download.get("audio_path"):
                audio_path = cached_download["audio_path"]
                timing["download"] = "cached"
            else:
                download_start = time.time()
                video_info = await download_video(request.url, str(test_dir))
                timing["download"] = round(time.time() - download_start, 2)
                audio_path = video_info["audio_path"]
                _save_cached_result(video_id, "download", video_info)

            stt_start = time.time()
            transcript = await transcribe_audio(audio_path)
            timing["stt"] = round(time.time() - stt_start, 2)
            source = "whisper"

        timing["total"] = round(time.time() - start, 2)

        if not transcript or not transcript.get("full_text"):
            return {
                "success": False,
                "video_id": video_id,
                "error": "텍스트 추출 실패",
                "timing": timing
            }

        _save_cached_result(video_id, "transcript", {
            "source": source,
            **transcript
        })

        return {
            "success": True,
            "video_id": video_id,
            "source": source,
            "timing": timing,
            "transcript": _format_transcript_preview(transcript),
            "next_step": f"POST /api/test/llm with video_id: {video_id}"
        }
    except Exception as e:
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "elapsed_time": round(time.time() - start, 2)
        }


# =============================================================================
# 3단계: LLM 레시피 파싱
# =============================================================================
@router.post("/llm")
async def test_llm_from_video(
    request: TestFromVideoIdRequest
) -> Dict[str, Any]:
    """
    3단계: LLM 레시피 파싱 테스트 (이전 단계 결과 사용).

    - 입력: video_id (이전 단계에서 받은 것)
    - 출력: 구조화된 레시피
    """
    start = time.time()

    video_id = request.video_id

    cached_transcript = _load_cached_result(video_id, "transcript")
    if not cached_transcript:
        return {
            "success": False,
            "video_id": video_id,
            "error": (
                "저장된 transcript가 없습니다. "
                "먼저 /subtitle, /stt, 또는 /transcript를 실행하세요."
            )
        }

    try:
        recipe = await parse_recipe(cached_transcript)
        elapsed = round(time.time() - start, 2)

        _save_cached_result(video_id, "recipe", recipe)

        result = {
            "success": True,
            "video_id": video_id,
            "elapsed_time": elapsed,
            "input_source": cached_transcript.get("source", "unknown"),
            "input_text_length": len(cached_transcript.get("full_text", "")),
            "recipe": recipe
        }

        save_log("llm", result, video_id)
        return result
    except Exception as e:
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "elapsed_time": round(time.time() - start, 2)
        }


@router.post("/llm-direct")
async def test_llm_direct(request: TestLLMRequest) -> Dict[str, Any]:
    """
    LLM 테스트 (텍스트 직접 입력).

    - 입력: 텍스트 직접 입력
    - 출력: 구조화된 레시피
    """
    start = time.time()

    test_transcript = {
        "full_text": request.text,
        "segments": [{"start": 0, "end": 10, "text": request.text}],
        "language": "ko",
        "duration": 60
    }

    try:
        recipe = await parse_recipe(test_transcript)
        elapsed = round(time.time() - start, 2)

        return {
            "success": True,
            "elapsed_time": elapsed,
            "input_text_length": len(request.text),
            "recipe": recipe
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "elapsed_time": round(time.time() - start, 2)
        }


# =============================================================================
# 전체 파이프라인 & 유틸리티
# =============================================================================
@router.post("/full")
async def test_full_pipeline(request: TestURLRequest) -> Dict[str, Any]:
    """
    전체 파이프라인 테스트 (동기 실행).

    다운로드 -> 자막/STT -> LLM
    """
    start = time.time()
    timing: Dict[str, Any] = {}

    video_id = extract_video_id(request.url)
    if not video_id:
        raise HTTPException(
            status_code=400,
            detail="유효하지 않은 YouTube URL입니다."
        )

    test_dir = _get_test_dir(video_id)
    test_dir.mkdir(exist_ok=True)

    try:
        # 1. 다운로드
        step_start = time.time()
        video_info = await download_video(request.url, str(test_dir))
        timing["download"] = round(time.time() - step_start, 2)
        _save_cached_result(video_id, "download", video_info)

        # 2. 자막/STT (자막 우선)
        step_start = time.time()
        transcript = None
        source = None

        try:
            subtitle_info = await download_subtitles(request.url, str(test_dir))
            if subtitle_info:
                transcript = parse_json3_subtitles(subtitle_info["subtitle_path"])
                full_text = transcript.get("full_text", "") if transcript else ""
                if transcript and len(full_text) >= MIN_TRANSCRIPT_LENGTH:
                    source = f"youtube_{subtitle_info['language']}"
                    if subtitle_info["is_auto_generated"]:
                        source += "_auto"
        except Exception:
            pass

        if not transcript or not transcript.get("full_text"):
            transcript = await transcribe_audio(video_info["audio_path"])
            source = "whisper"

        timing["transcript"] = round(time.time() - step_start, 2)
        timing["transcript_source"] = source
        _save_cached_result(video_id, "transcript", {"source": source, **transcript})

        # 3. LLM 파싱
        step_start = time.time()
        recipe = await parse_recipe(transcript)
        timing["llm_parsing"] = round(time.time() - step_start, 2)
        _save_cached_result(video_id, "recipe", recipe)

        timing["total"] = round(time.time() - start, 2)

        result = {
            "success": True,
            "video_id": video_id,
            "timing": timing,
            "summary": {
                "video_title": video_info.get("title"),
                "video_duration": video_info.get("duration"),
                "transcript_source": source,
                "transcript_length": len(transcript.get("full_text", "")),
                "segments_count": len(transcript.get("segments", [])),
                "recipe_title": recipe.get("title"),
                "ingredients_count": len(recipe.get("ingredients", [])),
                "steps_count": len(recipe.get("steps", []))
            },
            "transcript": transcript,
            "recipe": recipe
        }

        save_log("full", result, video_id)
        return result
    except Exception as e:
        return {
            "success": False,
            "video_id": video_id,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "elapsed_time": round(time.time() - start, 2)
        }


@router.get("/cache/{video_id}")
async def get_cached_results(video_id: str) -> Dict[str, Any]:
    """저장된 테스트 결과를 조회합니다."""
    test_dir = _get_test_dir(video_id)

    if not test_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="해당 video_id의 테스트 결과가 없습니다."
        )

    results: Dict[str, Any] = {}
    for stage in ["download", "transcript", "recipe"]:
        cached = _load_cached_result(video_id, stage)
        if cached:
            results[stage] = cached

    return {
        "video_id": video_id,
        "cached_stages": list(results.keys()),
        "results": results
    }


@router.delete("/cache/{video_id}")
async def clear_cache(video_id: str) -> Dict[str, Any]:
    """테스트 캐시를 삭제합니다."""
    test_dir = _get_test_dir(video_id)

    if not test_dir.exists():
        raise HTTPException(
            status_code=404,
            detail="해당 video_id의 테스트 결과가 없습니다."
        )

    shutil.rmtree(test_dir)
    return {
        "success": True,
        "message": f"video_id={video_id} 캐시가 삭제되었습니다."
    }
