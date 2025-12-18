"""
프롬프트 관리 모듈.

GPT 및 Whisper API에서 사용하는 프롬프트를 관리합니다.
"""
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


def _load_prompt(filename: str) -> str:
    """프롬프트 파일을 로드합니다."""
    filepath = _PROMPTS_DIR / filename
    return filepath.read_text(encoding="utf-8")


# Whisper 힌트 프롬프트 (한국어 요리 용어)
COOKING_PROMPT = _load_prompt("cooking.txt")

# GPT 레시피 파싱 프롬프트
RECIPE_PARSE_PROMPT = _load_prompt("recipe.txt")

# GPT 요리 어시스턴트 시스템 프롬프트
COOKING_ASSISTANT_PROMPT = _load_prompt("assistant.txt")
