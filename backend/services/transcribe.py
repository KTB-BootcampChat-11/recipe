"""
음성 인식(STT) 서비스 모듈.

Whisper API와 Silero VAD를 사용하여 오디오를 텍스트로 변환합니다.
"""
import io
import logging
import re
import warnings

# torchaudio deprecation 경고 숨기기
warnings.filterwarnings("ignore", message=".*torchaudio.*deprecated.*")
warnings.filterwarnings("ignore", message=".*sox_effects.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from openai import APIConnectionError, APIError, OpenAI, RateLimitError

from app.config import (
    MAX_AUDIO_FILE_SIZE,
    OPENAI_API_KEY,
    OPENAI_MODEL_WHISPER,
    SUPPORTED_AUDIO_FORMATS,
    VAD_MAX_SEGMENT_DURATION,
    VAD_MIN_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION,
    VAD_SAMPLE_RATE,
    VAD_SPEECH_PAD_MS,
)
from app.exceptions import AudioFileError, TranscriptionError
from app.prompts import COOKING_PROMPT

# =============================================================================
# 로깅 및 클라이언트 설정
# =============================================================================
logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

# =============================================================================
# VAD 모델 관리 (싱글톤)
# =============================================================================
_vad_model = None
_vad_utils = None


def _get_vad_model():
    """VAD 모델 싱글톤 로드 (torch hub 사용)."""
    global _vad_model, _vad_utils
    if _vad_model is None:
        logger.info("Silero VAD 모델 로딩 (torch hub)...")
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True
        )
        logger.info("Silero VAD 모델 로딩 완료")
    return _vad_model, _vad_utils


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

# 문장 구분자 패턴 (쉼표, 접속어 등)
SENTENCE_CONNECTORS = re.compile(
    r"(,\s*|그리고\s+|그런데\s+|그래서\s+|그러면\s+|다음에\s+|그다음에\s+|"
    r"먼저\s+|일단\s+|그러고\s+나서\s+|이제\s+)"
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
# VAD (Voice Activity Detection)
# =============================================================================
def _detect_speech_segments(audio_path: str) -> List[Dict[str, float]]:
    """
    VAD를 사용하여 오디오에서 발화 구간을 감지합니다.

    Args:
        audio_path: 오디오 파일 경로

    Returns:
        발화 구간 리스트 [{'start': float, 'end': float}, ...]
    """
    model, utils = _get_vad_model()
    (get_speech_timestamps, _, read_audio, _, _) = utils

    wav = read_audio(audio_path, sampling_rate=VAD_SAMPLE_RATE)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=VAD_SAMPLE_RATE,
        min_speech_duration_ms=int(VAD_MIN_SPEECH_DURATION * 1000),
        min_silence_duration_ms=int(VAD_MIN_SILENCE_DURATION * 1000),
        speech_pad_ms=VAD_SPEECH_PAD_MS,
        return_seconds=True
    )

    if not speech_timestamps:
        logger.warning("VAD: 발화 구간을 감지하지 못했습니다")
        return []

    segments = _split_long_segments(speech_timestamps)
    logger.info(f"VAD: {len(segments)}개 발화 구간 감지")
    return segments


def _split_long_segments(
    timestamps: List[Dict[str, float]]
) -> List[Dict[str, float]]:
    """
    긴 발화 구간을 최대 길이로 분할합니다.

    Args:
        timestamps: VAD 타임스탬프 리스트

    Returns:
        분할된 세그먼트 리스트
    """
    segments = []
    for ts in timestamps:
        start = ts["start"]
        end = ts["end"]
        duration = end - start

        if duration > VAD_MAX_SEGMENT_DURATION:
            current = start
            while current < end:
                seg_end = min(current + VAD_MAX_SEGMENT_DURATION, end)
                segments.append({"start": current, "end": seg_end})
                current = seg_end
        else:
            segments.append({"start": start, "end": end})

    return segments


def _extract_audio_segment(
    audio_path: str,
    start: float,
    end: float
) -> io.BytesIO:
    """
    오디오 파일에서 특정 구간을 추출합니다.

    Args:
        audio_path: 원본 오디오 파일 경로
        start: 시작 시간 (초)
        end: 끝 시간 (초)

    Returns:
        WAV 형식의 오디오 바이트 버퍼
    """
    waveform, sample_rate = torchaudio.load(audio_path)

    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    segment = waveform[:, start_sample:end_sample]

    buffer = io.BytesIO()
    torchaudio.save(buffer, segment, sample_rate, format="wav")
    buffer.seek(0)

    return buffer


# =============================================================================
# Whisper API 호출
# =============================================================================
def _transcribe_segment(
    audio_bytes: io.BytesIO,
    language: str = "ko"
) -> Any:
    """
    오디오 세그먼트를 Whisper API로 전사합니다.

    Args:
        audio_bytes: 오디오 바이트 버퍼
        language: 언어 코드

    Returns:
        Whisper API 응답
    """
    audio_bytes.name = "segment.wav"

    response = client.audio.transcriptions.create(
        model=OPENAI_MODEL_WHISPER,
        file=audio_bytes,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        language=language,
        prompt=COOKING_PROMPT
    )

    return response


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
def _split_into_sentences(response: Any) -> List[Dict[str, Any]]:
    """
    Word 타임스탬프를 사용하여 문장 단위로 세그먼트를 분할합니다.

    Args:
        response: Whisper API 응답

    Returns:
        세그먼트 리스트
    """
    words = getattr(response, "words", None)

    if not words:
        return _extract_segments_from_response(response)

    sentences = []
    current_words = []
    current_start = None

    for i, word in enumerate(words):
        word_text = getattr(word, "word", "").strip()
        word_start = getattr(word, "start", 0)
        word_end = getattr(word, "end", 0)

        if not word_text:
            continue

        if current_start is None:
            current_start = word_start

        current_words.append(word_text)

        is_sentence_end = _check_sentence_end(
            word_text, word_end, words, i, len(current_words)
        )

        if is_sentence_end:
            sentence_text = _join_words(current_words)
            if sentence_text:
                sentences.append({
                    "start": round(current_start, 2),
                    "end": round(word_end, 2),
                    "text": sentence_text
                })
            current_words = []
            current_start = None

    # 남은 단어들 처리
    if current_words:
        sentence_text = _join_words(current_words)
        if sentence_text and words:
            sentences.append({
                "start": round(current_start or 0, 2),
                "end": round(getattr(words[-1], "end", 0), 2),
                "text": sentence_text
            })

    if not sentences:
        return _extract_segments_from_response(response)

    return sentences


def _check_sentence_end(
    word_text: str,
    word_end: float,
    words: List[Any],
    index: int,
    word_count: int
) -> bool:
    """
    문장 종결 조건을 체크합니다.

    Args:
        word_text: 현재 단어 텍스트
        word_end: 현재 단어 종료 시간
        words: 전체 단어 리스트
        index: 현재 인덱스
        word_count: 현재까지 누적 단어 수

    Returns:
        문장 종결 여부
    """
    # 문장 종결 패턴 매칭
    if SENTENCE_ENDINGS.search(word_text):
        return True

    # 긴 pause 감지 (0.8초 이상)
    if index < len(words) - 1:
        next_start = getattr(words[index + 1], "start", 0)
        if next_start - word_end > 0.8:
            return True

    # 문장이 너무 길면 강제 분리 (15단어 이상)
    if word_count >= 15:
        return True

    return False


def _join_words(words: List[str]) -> str:
    """
    단어 리스트를 자연스러운 문장으로 연결합니다.

    Args:
        words: 단어 리스트

    Returns:
        연결된 문장
    """
    if not words:
        return ""

    text = " ".join(words)

    # 조사 앞 불필요한 공백 제거
    text = re.sub(
        r"\s+(을|를|이|가|은|는|에|의|로|으로|와|과|도|만|까지|부터|에서)\b",
        r"\1",
        text
    )

    # 숫자와 단위 사이 공백 제거
    text = re.sub(
        r"(\d+)\s+(분|초|g|ml|개|장|큰술|작은술|스푼|컵)",
        r"\1\2",
        text
    )

    return text.strip()


def _extract_segments_from_response(response: Any) -> List[Dict[str, Any]]:
    """
    Whisper 응답에서 세그먼트를 추출합니다.

    Args:
        response: Whisper API 응답

    Returns:
        세그먼트 리스트
    """
    segments = []

    if hasattr(response, "segments") and response.segments:
        for segment in response.segments:
            text = getattr(segment, "text", "").strip()
            if text:
                segments.append({
                    "start": getattr(segment, "start", 0),
                    "end": getattr(segment, "end", 0),
                    "text": text
                })

    return segments


# =============================================================================
# 메인 전사 함수
# =============================================================================
async def transcribe_audio(
    audio_path: str,
    language: str = "ko",
    use_vad: bool = True
) -> Dict[str, Any]:
    """
    VAD + Whisper API를 사용하여 오디오를 텍스트로 변환합니다.

    Args:
        audio_path: 오디오 파일 경로
        language: 언어 코드 (기본: ko)
        use_vad: VAD 사용 여부 (기본: True)

    Returns:
        전사 결과 딕셔너리:
        - full_text: 전체 텍스트
        - segments: 세그먼트 리스트
        - language: 언어 코드
        - duration: 오디오 길이

    Raises:
        TranscriptionError: 음성 인식 실패 시
        AudioFileError: 오디오 파일이 유효하지 않은 경우
    """
    _validate_audio_file(audio_path)

    logger.info(f"음성 인식 시작: {audio_path} (VAD: {use_vad})")

    try:
        if use_vad:
            return await _transcribe_with_vad(audio_path, language)
        else:
            return await _transcribe_full_audio(audio_path, language)

    except (RateLimitError, APIConnectionError, APIError) as e:
        logger.error(f"API 오류: {e}")
        raise TranscriptionError(f"음성 인식 API 오류: {e}")

    except AudioFileError:
        raise

    except Exception as e:
        logger.error(f"음성 인식 중 예상치 못한 오류: {e}")
        raise TranscriptionError(f"음성 인식 중 오류가 발생했습니다: {e}")


async def _transcribe_with_vad(
    audio_path: str,
    language: str
) -> Dict[str, Any]:
    """VAD를 사용하여 발화 구간별로 전사합니다."""
    speech_segments = _detect_speech_segments(audio_path)

    if not speech_segments:
        logger.warning("VAD: 발화 구간이 없습니다. 전체 오디오로 폴백합니다.")
        return await _transcribe_full_audio(audio_path, language)

    all_segments = []
    all_texts = []
    total_duration = 0

    for i, seg in enumerate(speech_segments):
        logger.info(
            f"VAD 구간 {i + 1}/{len(speech_segments)}: "
            f"{seg['start']:.1f}s - {seg['end']:.1f}s"
        )

        audio_buffer = _extract_audio_segment(
            audio_path, seg["start"], seg["end"]
        )
        response = _transcribe_segment(audio_buffer, language)

        segment_text = getattr(response, "text", "") or ""

        if segment_text.strip():
            cleaned_text = _clean_transcript_text(segment_text)
            all_texts.append(cleaned_text)

            sub_segments = _split_into_sentences(response)

            # 원본 오디오 기준으로 타임스탬프 오프셋 적용
            for sub_seg in sub_segments:
                sub_seg["start"] = round(seg["start"] + sub_seg["start"], 2)
                sub_seg["end"] = round(seg["start"] + sub_seg["end"], 2)
                sub_seg["text"] = _clean_transcript_text(sub_seg["text"])

            all_segments.extend(sub_segments)

        total_duration = max(total_duration, seg["end"])

    full_text = " ".join(all_texts)

    logger.info(
        f"음성 인식 완료 (VAD): {len(full_text)}자, "
        f"{len(all_segments)}개 세그먼트"
    )

    return {
        "full_text": full_text,
        "segments": all_segments,
        "language": language,
        "duration": total_duration
    }


async def _transcribe_full_audio(
    audio_path: str,
    language: str
) -> Dict[str, Any]:
    """전체 오디오를 한 번에 전사합니다 (VAD 미사용 폴백)."""
    with open(audio_path, "rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=OPENAI_MODEL_WHISPER,
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
            language=language,
            prompt=COOKING_PROMPT
        )

    if not response:
        raise TranscriptionError("음성 인식 응답이 비어있습니다")

    full_text = getattr(response, "text", "") or ""

    if not full_text.strip():
        logger.warning("음성 인식 결과가 비어있습니다 (무음 또는 인식 실패)")
        return {
            "full_text": "",
            "segments": [],
            "language": language,
            "duration": getattr(response, "duration", 0)
        }

    cleaned_text = _clean_transcript_text(full_text)
    segments = _split_into_sentences(response)

    for seg in segments:
        seg["text"] = _clean_transcript_text(seg["text"])

    logger.info(
        f"음성 인식 완료: {len(cleaned_text)}자, {len(segments)}개 세그먼트"
    )

    return {
        "full_text": cleaned_text,
        "segments": segments,
        "language": getattr(response, "language", language),
        "duration": getattr(response, "duration", 0)
    }
