"""
YouTube 다운로드 서비스 모듈.

YouTube 영상 다운로드, 오디오 추출, 자막 처리 기능을 제공합니다.
"""
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import yt_dlp

from app.exceptions import YouTubeDownloadError

logger = logging.getLogger(__name__)


# =============================================================================
# 예외 클래스
# =============================================================================

class VideoNotFoundError(YouTubeDownloadError):
    """영상을 찾을 수 없는 경우."""

    pass


class VideoUnavailableError(YouTubeDownloadError):
    """영상이 비공개이거나 삭제된 경우."""

    pass


# =============================================================================
# 상수
# =============================================================================

SUPPORTED_VIDEO_EXTENSIONS = ["mp4", "webm", "mkv", "mov", "avi"]
MAX_VIDEO_DURATION = 180  # 3분 (쇼츠는 보통 60초 이하)
SUBTITLE_LANG_PRIORITY = ["ko", "en", "ja"]


# =============================================================================
# 헬퍼 함수
# =============================================================================

def _create_progress_hook(label: str):
    """다운로드 진행률 로깅용 hook (10% 단위로만 출력)."""
    last_logged = [0]

    def hook(d: Dict) -> None:
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            if total > 0:
                percent = int(downloaded / total * 100)
                if percent >= last_logged[0] + 10:
                    last_logged[0] = (percent // 10) * 10
                    speed = d.get("speed")
                    speed_str = f"{speed / 1024 / 1024:.1f}MB/s" if speed else "..."
                    logger.info(f"[{label}] 다운로드 {percent}% ({speed_str})")
        elif d["status"] == "finished":
            logger.info(f"[{label}] 다운로드 완료")

    return hook


def _get_ydl_base_opts() -> Dict:
    """기본 yt-dlp 옵션을 반환합니다."""
    return {
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
    }


def _handle_download_error(error: Exception, context: str) -> None:
    """다운로드 에러를 적절한 예외로 변환합니다."""
    error_msg = str(error).lower()

    if "private" in error_msg or "unavailable" in error_msg:
        raise VideoUnavailableError("영상이 비공개이거나 삭제되었습니다.")
    elif "not found" in error_msg or "404" in error_msg:
        raise VideoNotFoundError("영상을 찾을 수 없습니다.")
    else:
        raise YouTubeDownloadError(f"{context}: {error}")


# =============================================================================
# 메인 함수
# =============================================================================

def extract_video_id(url: str) -> Optional[str]:
    """
    YouTube URL에서 video ID를 추출합니다.

    Args:
        url: YouTube URL

    Returns:
        비디오 ID 또는 None
    """
    if not url:
        return None

    patterns = [
        r"(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]+)",
        r"(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]+)",
        r"(?:youtu\.be\/)([a-zA-Z0-9_-]+)",
        r"(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


async def get_video_info(url: str) -> Optional[Dict]:
    """
    영상 정보만 조회합니다 (다운로드 없이).

    Args:
        url: YouTube URL

    Returns:
        영상 정보 딕셔너리 또는 None
    """
    video_id = extract_video_id(url)
    if not video_id:
        return None

    loop = asyncio.get_event_loop()
    ydl_opts = {**_get_ydl_base_opts(), "extract_flat": False}

    try:
        def get_info():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        info = await loop.run_in_executor(None, get_info)

        if info:
            return {
                "video_id": info.get("id"),
                "title": info.get("title"),
                "duration": info.get("duration"),
                "thumbnail": info.get("thumbnail"),
                "channel": info.get("channel"),
                "view_count": info.get("view_count"),
            }

    except Exception as e:
        logger.error(f"영상 정보 조회 실패: {e}")

    return None


async def download_video(url: str, output_dir: str) -> Dict:
    """
    YouTube 쇼츠 영상과 오디오를 다운로드합니다.

    Args:
        url: YouTube URL
        output_dir: 출력 디렉토리

    Returns:
        다운로드 결과 정보 딕셔너리

    Raises:
        YouTubeDownloadError: 다운로드 실패 시
        VideoNotFoundError: 영상을 찾을 수 없는 경우
        VideoUnavailableError: 영상이 비공개/삭제된 경우
    """
    video_id = extract_video_id(url)
    if not video_id:
        raise VideoNotFoundError(f"유효하지 않은 YouTube URL입니다: {url}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    # 1. 영상 정보 가져오기
    info = await _fetch_video_info(url, loop)
    actual_video_id = info.get("id", video_id)
    title = info.get("title", "untitled")
    duration = info.get("duration", 0)

    logger.info(f"영상 정보: {title} ({duration}초)")

    if duration and duration > MAX_VIDEO_DURATION:
        logger.warning(
            f"영상이 너무 깁니다: {duration}초 (최대 {MAX_VIDEO_DURATION}초)"
        )

    # 2. 영상 다운로드
    await _download_video_file(url, output_dir, actual_video_id, loop)

    # 3. 오디오 추출
    audio_path = os.path.join(output_dir, f"{actual_video_id}.mp3")
    await _extract_audio(url, output_dir, actual_video_id, loop)

    # 4. 파일 확인
    video_path = _find_video_file(output_dir, actual_video_id)

    if not video_path:
        raise YouTubeDownloadError(
            f"다운로드된 비디오 파일을 찾을 수 없습니다: {actual_video_id}"
        )

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise YouTubeDownloadError(
            f"오디오 파일을 찾을 수 없거나 비어있습니다: {audio_path}"
        )

    logger.info(f"다운로드 완료: video={video_path}, audio={audio_path}")

    return {
        "video_path": video_path,
        "audio_path": audio_path,
        "video_id": actual_video_id,
        "title": title,
        "duration": duration,
        "url": url,
    }


async def _fetch_video_info(url: str, loop) -> Dict:
    """영상 정보를 가져옵니다."""
    logger.info("영상 정보 조회 중...")

    ydl_opts = {**_get_ydl_base_opts(), "extract_flat": False}

    try:
        def get_info():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        info = await loop.run_in_executor(None, get_info)

        if not info:
            raise VideoNotFoundError("영상 정보를 가져올 수 없습니다.")

        return info

    except yt_dlp.utils.DownloadError as e:
        _handle_download_error(e, "영상 정보 조회 실패")
    except Exception as e:
        raise YouTubeDownloadError(f"영상 정보 조회 중 오류: {e}")


async def _download_video_file(
    url: str,
    output_dir: str,
    video_id: str,
    loop
) -> None:
    """영상 파일을 다운로드합니다."""
    logger.info(f"영상 다운로드 중: {video_id}")

    ydl_opts = {
        **_get_ydl_base_opts(),
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, f"{video_id}.%(ext)s"),
        "noprogress": True,
        "progress_hooks": [_create_progress_hook("영상")],
    }

    try:
        def download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        await loop.run_in_executor(None, download)

    except yt_dlp.utils.DownloadError as e:
        raise YouTubeDownloadError(f"영상 다운로드 실패: {e}")
    except Exception as e:
        raise YouTubeDownloadError(f"영상 다운로드 중 오류: {e}")


async def _extract_audio(
    url: str,
    output_dir: str,
    video_id: str,
    loop
) -> None:
    """영상에서 오디오를 추출합니다."""
    logger.info(f"오디오 추출 중: {video_id}")

    ydl_opts = {
        **_get_ydl_base_opts(),
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "outtmpl": os.path.join(output_dir, f"{video_id}.%(ext)s"),
        "noprogress": True,
        "progress_hooks": [_create_progress_hook("오디오")],
    }

    try:
        def download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        await loop.run_in_executor(None, download)

    except yt_dlp.utils.DownloadError as e:
        raise YouTubeDownloadError(f"오디오 추출 실패: {e}")
    except Exception as e:
        raise YouTubeDownloadError(f"오디오 추출 중 오류: {e}")


def _find_video_file(output_dir: str, video_id: str) -> Optional[str]:
    """다운로드된 비디오 파일을 찾습니다."""
    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        path = os.path.join(output_dir, f"{video_id}.{ext}")
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    return None


# =============================================================================
# 자막 관련 함수
# =============================================================================

async def download_subtitles(url: str, output_dir: str) -> Optional[Dict]:
    """
    YouTube 영상의 자막을 다운로드합니다.

    Args:
        url: YouTube URL
        output_dir: 출력 디렉토리

    Returns:
        자막 정보 딕셔너리 또는 None
    """
    video_id = extract_video_id(url)
    if not video_id:
        return None

    loop = asyncio.get_event_loop()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 자막 정보 조회
    info = await _fetch_subtitle_info(url, loop)
    if not info:
        return None

    # 언어 선택
    subtitles = info.get("subtitles", {})
    auto_captions = info.get("automatic_captions", {})
    selected_lang, is_auto = _select_subtitle_language(subtitles, auto_captions)

    if not selected_lang:
        logger.info("사용 가능한 자막 없음")
        return None

    # 자막 다운로드
    subtitle_path = os.path.join(output_dir, f"{video_id}.{selected_lang}.json3")

    success = await _download_subtitle_file(
        url, output_dir, video_id, selected_lang, is_auto, loop
    )

    if not success or not os.path.exists(subtitle_path):
        logger.warning(f"자막 파일을 찾을 수 없음: {subtitle_path}")
        return None

    logger.info(f"자막 다운로드 완료: {subtitle_path}")

    return {
        "subtitle_path": subtitle_path,
        "language": selected_lang,
        "is_auto_generated": is_auto,
    }


async def _fetch_subtitle_info(url: str, loop) -> Optional[Dict]:
    """자막 정보를 조회합니다."""
    ydl_opts = {**_get_ydl_base_opts(), "extract_flat": False}

    try:
        def get_info():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                return ydl.extract_info(url, download=False)

        return await loop.run_in_executor(None, get_info)

    except Exception as e:
        logger.warning(f"자막 정보 조회 실패: {e}")
        return None


def _select_subtitle_language(
    subtitles: Dict,
    auto_captions: Dict
) -> tuple[Optional[str], bool]:
    """우선순위에 따라 자막 언어를 선택합니다."""
    # 1. 수동 자막 우선
    for lang in SUBTITLE_LANG_PRIORITY:
        if lang in subtitles:
            logger.info(f"수동 자막 발견: {lang}")
            return lang, False

    # 2. 자동 생성 자막
    for lang in SUBTITLE_LANG_PRIORITY:
        if lang in auto_captions:
            logger.info(f"자동 생성 자막 발견: {lang}")
            return lang, True

    # 3. 기본 언어
    if auto_captions:
        lang = next(iter(auto_captions.keys()))
        logger.info(f"기본 자동 자막 사용: {lang}")
        return lang, True

    if subtitles:
        lang = next(iter(subtitles.keys()))
        logger.info(f"기본 수동 자막 사용: {lang}")
        return lang, False

    return None, False


async def _download_subtitle_file(
    url: str,
    output_dir: str,
    video_id: str,
    lang: str,
    is_auto: bool,
    loop
) -> bool:
    """자막 파일을 다운로드합니다."""
    ydl_opts = {
        **_get_ydl_base_opts(),
        "skip_download": True,
        "writesubtitles": not is_auto,
        "writeautomaticsub": is_auto,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3",
        "outtmpl": os.path.join(output_dir, video_id),
    }

    try:
        def download():
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        await loop.run_in_executor(None, download)
        return True

    except Exception as e:
        logger.warning(f"자막 다운로드 실패: {e}")
        return False


def parse_json3_subtitles(subtitle_path: str) -> Optional[Dict]:
    """
    JSON3 형식의 자막 파일을 파싱합니다.

    Args:
        subtitle_path: JSON3 자막 파일 경로

    Returns:
        전사 결과 딕셔너리 또는 None
    """
    try:
        with open(subtitle_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"자막 파일 읽기 실패: {e}")
        return None

    events = data.get("events", [])
    if not events:
        logger.warning("자막 이벤트가 없습니다")
        return None

    segments = _parse_subtitle_events(events)

    if not segments:
        logger.warning("파싱된 자막 세그먼트가 없습니다")
        return None

    # 중복 제거
    cleaned_segments = _remove_duplicate_segments(segments)
    full_text = " ".join([seg["text"] for seg in cleaned_segments])

    # duration 계산 (마지막 세그먼트의 end 시간)
    duration = cleaned_segments[-1]["end"] if cleaned_segments else 0

    logger.info(
        f"자막 파싱 완료: {len(cleaned_segments)}개 세그먼트, {len(full_text)}자"
    )

    return {
        "full_text": full_text,
        "segments": cleaned_segments,
        "language": "ko",  # YouTube 자막은 언어 선택 후 다운로드하므로
        "duration": duration,
    }


def _parse_subtitle_events(events: List[Dict]) -> List[Dict]:
    """자막 이벤트를 세그먼트로 변환합니다."""
    segments = []

    for event in events:
        segs = event.get("segs", [])
        if not segs:
            continue

        start_ms = event.get("tStartMs", 0)
        duration_ms = event.get("dDurationMs", 0)
        start_sec = start_ms / 1000.0
        end_sec = (start_ms + duration_ms) / 1000.0

        text_parts = []
        for seg in segs:
            utf8_text = seg.get("utf8", "")
            if utf8_text and utf8_text.strip() and utf8_text != "\n":
                text_parts.append(utf8_text)

        if text_parts:
            text = "".join(text_parts).strip()
            if text and text not in ("[Music]", "[음악]"):
                segments.append({
                    "start": round(start_sec, 2),
                    "end": round(end_sec, 2),
                    "text": text,
                })

    return segments


def _remove_duplicate_segments(segments: List[Dict]) -> List[Dict]:
    """연속된 중복 세그먼트를 제거합니다."""
    cleaned = []
    prev_text = ""

    for seg in segments:
        if seg["text"] != prev_text:
            cleaned.append(seg)
            prev_text = seg["text"]

    return cleaned
