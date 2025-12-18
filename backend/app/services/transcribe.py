"""
음성 인식(STT) 서비스 모듈.

OpenAI Whisper API를 사용하여 오디오를 텍스트로 변환합니다.
하이브리드 모드: gpt-4o-transcribe (정확도) + whisper-1 (타임스탬프) 병합
"""
import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import httpx
from openai import OpenAI

from app.config import (
    MAX_AUDIO_FILE_SIZE,
    OPENAI_API_KEY,
    SUPPORTED_AUDIO_FORMATS,
)
from app.exceptions import AudioFileError, TranscriptionError
from app.prompts import COOKING_PROMPT

# =============================================================================
# 로깅 및 클라이언트 설정
# =============================================================================
logger = logging.getLogger(__name__)

# 타임아웃 설정 (오디오 파일 업로드 + 처리 시간 고려)
# connect: 연결 타임아웃, read: 응답 대기 타임아웃
http_client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0))
client = OpenAI(api_key=OPENAI_API_KEY, http_client=http_client)

# =============================================================================
# 상수
# =============================================================================
MODEL_ACCURATE = "gpt-4o-transcribe"  # 정확한 텍스트용
MODEL_TIMESTAMP = "whisper-1"          # 타임스탬프용


# =============================================================================
# 텍스트 처리 패턴 (pre-compiled)
# =============================================================================
# 문장 종결 패턴 (한국어)
SENTENCE_ENDINGS = re.compile(
    r"(요|다|죠|네요|세요|해요|하세요|합니다|됩니다|입니다|있어요|없어요|"
    r"주세요|드세요|넣으세요|볶으세요|썰어주세요|구워주세요|끓여주세요|"
    r"거든요|잖아요|대요|래요|냐고요|는데요|어요|아요|"
    r"고요|구요|나요|까요|ㄹ까요|을까요|"
    r"니다|ㅂ니다|습니다|"
    r"거예요|건데요|세요|네요|죠|어|야)[\.\!\?]?$"
)


# =============================================================================
# 파일 유효성 검사
# =============================================================================
def _validate_audio_file(audio_path: str) -> None:
    """
    오디오 파일 유효성 검사.

    Args:
        audio_path: 오디오 파일 경로

    Raises:
        AudioFileError: 파일이 유효하지 않은 경우
    """
    path = Path(audio_path)

    if not path.exists():
        raise AudioFileError(
            f"오디오 파일을 찾을 수 없습니다: {audio_path}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioFileError(f"오디오 파일이 비어있습니다: {audio_path}")

    if file_size > MAX_AUDIO_FILE_SIZE:
        raise AudioFileError(
            f"오디오 파일이 너무 큽니다: {file_size / 1024 / 1024:.1f}MB "
            f"(최대 {MAX_AUDIO_FILE_SIZE / 1024 / 1024}MB)"
        )

    if path.suffix.lower() not in SUPPORTED_AUDIO_FORMATS:
        raise AudioFileError(
            f"지원하지 않는 오디오 형식입니다: {path.suffix} "
            f"(지원 형식: {', '.join(SUPPORTED_AUDIO_FORMATS)})"
        )


# =============================================================================
# 텍스트 정제
# =============================================================================
def _clean_transcript_text(text: str) -> str:
    """
    전사 텍스트 정제 (반복, 필러 단어, 오인식 패턴 제거).

    Args:
        text: 원본 텍스트

    Returns:
        정제된 텍스트
    """
    if not text:
        return ""

    # 필러 단어 제거
    text = re.sub(r"\b(음+|어+|그+|아+|에+)\.{0,3}\s*", "", text)

    # 반복되는 감탄사 제거
    text = re.sub(r"\b(네네|아아|오오|와와|음음|어어)\b", "", text)

    # 의미없는 반복 패턴 제거
    text = re.sub(r"\b(\w+)(\s+\1){2,}\b", r"\1", text)

    # Whisper 오인식 패턴 교정 (유튜브 관련)
    text = re.sub(r"구독\s*좋아요\s*알림", "", text)
    text = re.sub(r"구독과\s*좋아요", "", text)

    # 배경음악 인식 오류 제거
    text = re.sub(r"♪+|♫+|🎵+|🎶+", "", text)
    text = re.sub(
        r"\[음악\]|\[배경음악\]|\[BGM\]",
        "",
        text,
        flags=re.IGNORECASE
    )

    # 숫자+단위 정규화
    text = re.sub(r"(\d+)\s*스푼", r"\1스푼", text)
    text = re.sub(r"(\d+)\s*큰술", r"\1큰술", text)
    text = re.sub(r"(\d+)\s*작은술", r"\1작은술", text)
    text = re.sub(r"(\d+)\s*분", r"\1분", text)
    text = re.sub(r"(\d+)\s*초", r"\1초", text)
    text = re.sub(r"(\d+)\s*그램", r"\1g", text)
    text = re.sub(r"(\d+)\s*g", r"\1g", text)
    text = re.sub(r"(\d+)\s*ml", r"\1ml", text, flags=re.IGNORECASE)

    # 연속된 공백/줄바꿈 정리
    text = re.sub(r"\s+", " ", text)

    # 문장 부호 정리
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)

    return text.strip()


# =============================================================================
# 문장 분리
# =============================================================================
def _split_into_sentences_from_segments(
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Whisper API 세그먼트를 문장 단위로 정리합니다.

    Args:
        segments: Whisper API 세그먼트 리스트

    Returns:
        정리된 세그먼트 리스트
    """
    if not segments:
        return []

    result = []
    for seg in segments:
        text = seg.get("text", "").strip()
        if text:
            result.append({
                "start": round(seg.get("start", 0), 2),
                "end": round(seg.get("end", 0), 2),
                "text": _clean_transcript_text(text)
            })

    return result


def _split_text_into_sentences(
    full_text: str,
    duration: float
) -> List[Dict[str, Any]]:
    """
    전체 텍스트를 문장 단위로 분리합니다 (타임스탬프 없는 경우).

    Args:
        full_text: 전체 텍스트
        duration: 오디오 길이

    Returns:
        세그먼트 리스트
    """
    if not full_text:
        return []

    # 문장 종결 패턴으로 분리
    sentences = re.split(r'(?<=[.!?요다죠])\s+', full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [{"start": 0, "end": duration, "text": full_text}]

    # 균등하게 타임스탬프 배분
    time_per_sentence = duration / len(sentences) if sentences else 0
    result = []

    for i, sentence in enumerate(sentences):
        result.append({
            "start": round(i * time_per_sentence, 2),
            "end": round((i + 1) * time_per_sentence, 2),
            "text": _clean_transcript_text(sentence)
        })

    return result


# =============================================================================
# OpenAI Whisper API 호출
# =============================================================================
def _transcribe_accurate(audio_path: str, language: str = "ko") -> Dict[str, Any]:
    """
    gpt-4o-transcribe 모델로 정확한 텍스트를 얻습니다.

    Args:
        audio_path: 오디오 파일 경로
        language: 언어 코드

    Returns:
        전사 결과 (text만 포함)
    """
    print("[STT] gpt-4o-transcribe API 호출 중...")
    logger.info(f"[STT] gpt-4o-transcribe API 호출: {audio_path}")

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=MODEL_ACCURATE,
            file=audio_file,
            language=language,
            response_format="json",
            prompt=COOKING_PROMPT
        )

    return response


def _transcribe_with_timestamps(
    audio_path: str,
    language: str = "ko"
) -> Dict[str, Any]:
    """
    whisper-1 모델로 타임스탬프가 포함된 세그먼트를 얻습니다.

    Args:
        audio_path: 오디오 파일 경로
        language: 언어 코드

    Returns:
        전사 결과 (segments, duration 포함)
    """
    print("[STT] whisper-1 API 호출 중 (타임스탬프)...")
    logger.info(f"[STT] whisper-1 API 호출 (타임스탬프용): {audio_path}")

    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=MODEL_TIMESTAMP,
            file=audio_file,
            language=language,
            response_format="verbose_json",
            prompt=COOKING_PROMPT
        )

    return response


def _merge_transcripts(
    accurate_text: str,
    timestamp_response: Any
) -> Dict[str, Any]:
    """
    gpt-4o-transcribe 텍스트와 whisper-1 타임스탬프를 병합합니다.

    전략:
    - full_text: gpt-4o-transcribe의 정확한 텍스트 사용
    - segments: whisper-1의 타임스탬프 구조를 유지하되,
                각 세그먼트 텍스트를 gpt-4o-transcribe 텍스트에서 매칭

    Args:
        accurate_text: gpt-4o-transcribe에서 얻은 정확한 텍스트
        timestamp_response: whisper-1에서 얻은 타임스탬프 응답

    Returns:
        병합된 전사 결과
    """
    # whisper-1 응답에서 세그먼트와 duration 추출
    segments = []
    if hasattr(timestamp_response, 'segments') and timestamp_response.segments:
        segments = timestamp_response.segments

    duration = timestamp_response.duration if hasattr(
        timestamp_response, 'duration'
    ) else 0

    if not segments:
        # 세그먼트가 없으면 전체 텍스트를 하나의 세그먼트로
        return {
            "segments": [{"start": 0, "end": duration, "text": accurate_text}],
            "duration": duration
        }

    # 정확한 텍스트를 문장 단위로 분리
    accurate_sentences = re.split(r'(?<=[.!?요다죠])\s*', accurate_text)
    accurate_sentences = [s.strip() for s in accurate_sentences if s.strip()]

    # whisper-1 세그먼트 텍스트도 문장 단위로 분리하여 매핑 준비
    whisper_sentences = []
    for seg in segments:
        text = seg.text if hasattr(seg, 'text') else seg.get('text', '')
        whisper_sentences.append({
            "start": seg.start if hasattr(seg, 'start') else seg.get('start', 0),
            "end": seg.end if hasattr(seg, 'end') else seg.get('end', 0),
            "text": text.strip()
        })

    # 병합 전략: whisper-1의 타임스탬프 구조 유지, 텍스트는 비율에 맞게 배분
    merged_segments = []

    if len(accurate_sentences) == len(whisper_sentences):
        # 문장 수가 같으면 1:1 매핑
        for i, ws in enumerate(whisper_sentences):
            merged_segments.append({
                "start": round(ws["start"], 2),
                "end": round(ws["end"], 2),
                "text": accurate_sentences[i]
            })
    else:
        # 문장 수가 다르면 whisper-1 타임스탬프에 gpt-4o 텍스트를 비율 배분
        total_whisper_len = sum(len(ws["text"]) for ws in whisper_sentences)

        if total_whisper_len == 0:
            # whisper 텍스트가 비어있으면 균등 배분
            time_per_sentence = duration / len(accurate_sentences)
            for i, sent in enumerate(accurate_sentences):
                merged_segments.append({
                    "start": round(i * time_per_sentence, 2),
                    "end": round((i + 1) * time_per_sentence, 2),
                    "text": sent
                })
        else:
            # whisper 세그먼트 길이 비율에 따라 accurate 텍스트 배분
            accurate_full = " ".join(accurate_sentences)
            accurate_idx = 0

            for ws in whisper_sentences:
                # 이 세그먼트에 할당할 문자 수 계산
                ratio = len(ws["text"]) / total_whisper_len
                chars_to_assign = int(len(accurate_full) * ratio)

                # 문장 경계에서 끊기
                end_idx = min(accurate_idx + chars_to_assign, len(accurate_full))

                # 마지막이 아니면 문장 종결 위치 찾기
                if end_idx < len(accurate_full):
                    # 가장 가까운 문장 종결 찾기
                    for delim in ['. ', '! ', '? ', '요 ', '다 ', '죠 ']:
                        pos = accurate_full.find(delim, end_idx - 20, end_idx + 20)
                        if pos != -1:
                            end_idx = pos + len(delim)
                            break

                segment_text = accurate_full[accurate_idx:end_idx].strip()
                accurate_idx = end_idx

                if segment_text:
                    merged_segments.append({
                        "start": round(ws["start"], 2),
                        "end": round(ws["end"], 2),
                        "text": segment_text
                    })

            # 남은 텍스트가 있으면 마지막 세그먼트에 추가
            if accurate_idx < len(accurate_full) and merged_segments:
                remaining = accurate_full[accurate_idx:].strip()
                if remaining:
                    merged_segments[-1]["text"] += " " + remaining

    return {
        "segments": merged_segments,
        "duration": duration
    }


# =============================================================================
# 메인 전사 함수
# =============================================================================
async def transcribe_audio(
    audio_path: str,
    language: str = "ko",
    use_vad: bool = False  # API 모드에서는 VAD 미사용
) -> Dict[str, Any]:
    """
    하이브리드 방식으로 오디오를 텍스트로 변환합니다.

    두 모델을 병렬로 호출하여 결과를 병합:
    - gpt-4o-transcribe: 정확한 텍스트
    - whisper-1: 타임스탬프/세그먼트

    Args:
        audio_path: 오디오 파일 경로
        language: 언어 코드 (기본: ko)
        use_vad: VAD 사용 여부 (API 모드에서는 무시됨)

    Returns:
        전사 결과 딕셔너리:
        - full_text: 전체 텍스트 (gpt-4o-transcribe)
        - segments: 세그먼트 리스트 (타임스탬프는 whisper-1, 텍스트는 병합)
        - language: 언어 코드
        - duration: 오디오 길이

    Raises:
        TranscriptionError: 음성 인식 실패 시
        AudioFileError: 오디오 파일이 유효하지 않은 경우
    """
    _validate_audio_file(audio_path)

    print("[STT] 하이브리드 음성 인식 시작...")
    logger.info(f"[STT] 하이브리드 음성 인식 시작: {audio_path}")

    try:
        # 두 API를 병렬로 호출
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # gpt-4o-transcribe (정확한 텍스트)
            accurate_future = loop.run_in_executor(
                executor,
                _transcribe_accurate,
                audio_path,
                language
            )
            # whisper-1 (타임스탬프)
            timestamp_future = loop.run_in_executor(
                executor,
                _transcribe_with_timestamps,
                audio_path,
                language
            )

            # 두 결과 대기
            accurate_response, timestamp_response = await asyncio.gather(
                accurate_future,
                timestamp_future
            )

        # 정확한 텍스트 추출
        accurate_text = accurate_response.text if hasattr(
            accurate_response, 'text'
        ) else ""

        # 텍스트 정제
        cleaned_text = _clean_transcript_text(accurate_text)

        # 두 결과 병합
        merged = _merge_transcripts(cleaned_text, timestamp_response)

        # 세그먼트 텍스트도 정제
        cleaned_segments = []
        for seg in merged["segments"]:
            cleaned_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": _clean_transcript_text(seg["text"])
            })

        print(f"[STT] 하이브리드 음성 인식 완료: {len(cleaned_text)}자, {len(cleaned_segments)}개 세그먼트")
        logger.info(
            f"[STT] 하이브리드 음성 인식 완료: {len(cleaned_text)}자, "
            f"{len(cleaned_segments)}개 세그먼트"
        )

        return {
            "full_text": cleaned_text,
            "segments": cleaned_segments,
            "language": language,
            "duration": merged["duration"]
        }

    except AudioFileError:
        raise

    except Exception as e:
        logger.error(f"[STT] 음성 인식 중 오류: {e}")
        raise TranscriptionError(f"음성 인식 중 오류가 발생했습니다: {e}")
