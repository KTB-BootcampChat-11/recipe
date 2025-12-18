# 오늘 뭐먹지 - 쇼츠 레시피 정리기

YouTube Shorts에서 레시피를 자동으로 추출하고, AI 요리 어시스턴트와 함께 요리할 수 있는 서비스입니다.

## 주요 기능

### 1. 레시피 추출
- YouTube Shorts URL을 입력하면 자동으로 레시피 분석
- YouTube 자막 우선 사용 (없으면 Whisper STT 폴백)
- GPT-4o로 재료/조리순서 구조화

### 2. AI 요리 어시스턴트 (Chat)
- 추출된 레시피 기반 단계별 요리 가이드
- GPT-4o Vision으로 요리 사진 피드백
- 실시간 질문/답변 지원
- 요리 완료 시 홈으로 복귀

## 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                           │
├─────────────────────────────────────────────────────────────────┤
│  Routers: analyze.py, chat.py, test.py                          │
├─────────────────────────────────────────────────────────────────┤
│  Services:                                                       │
│  ┌──────────┐ ┌────────────┐ ┌──────────────┐                   │
│  │ youtube  │ │ transcribe │ │ recipe_parser│                   │
│  │ (yt-dlp) │ │ (Whisper)  │ │  (GPT-4o)    │                   │
│  └──────────┘ └────────────┘ └──────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### 레시피 추출 파이프라인

```
YouTube URL
    │
    ▼
┌──────────────┐
│ 1. Download  │ yt-dlp로 영상/오디오 다운로드
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│ 2. Subtitle  │────▶│ 2b. Whisper  │ 자막 없으면 STT 폴백
│  (YouTube)   │     │    (STT)     │
└──────┬───────┘     └──────┬───────┘
       │                    │
       └────────┬───────────┘
                │
                ▼
       ┌──────────────┐
       │ 3. GPT-4o    │ 레시피 구조화 (재료, 단계, 타임스탬프)
       │   Parsing    │
       └──────────────┘
```

## 기술 스택

### Backend
- FastAPI (MVC 패턴)
- OpenAI Whisper API (STT)
- OpenAI GPT-4o (레시피 파싱 + Vision)
- yt-dlp (YouTube 다운로드 + 자막)

### Frontend
- React 18
- TypeScript
- Tailwind CSS
- React Router
- Vite

## 사전 요구사항

- Python 3.10+
- Node.js 18+
- OpenAI API Key

## 설치 및 실행

### 1. 백엔드 설정

```bash
cd backend

# 가상환경 생성 및 활성화
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에 OPENAI_API_KEY 입력

# 서버 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. 프론트엔드 설정

```bash
cd frontend

# 패키지 설치
npm install

# 개발 서버 실행
npm run dev
```

### 3. 접속

브라우저에서 http://localhost:5173 접속

## 사용 방법

### 레시피 추출
1. YouTube Shorts URL을 입력 (예: https://youtube.com/shorts/xxxxx)
2. "레시피 추출" 버튼 클릭
3. 진행 상황 확인 (다운로드 → 자막/STT → 파싱)
4. 분석 완료 후 레시피 및 타임라인 확인

### AI 요리 어시스턴트
1. 레시피 추출 완료 후 "요리 시작하기" 버튼 클릭
2. 채팅방에서 단계별 요리 진행
3. 사진을 찍어 보내면 AI가 피드백 제공
4. "다음 단계" 버튼으로 진행
5. 모든 단계 완료 시 홈으로 복귀

## API 엔드포인트

### 레시피 분석
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/analyze` | YouTube URL 분석 시작 |
| GET | `/api/status/{job_id}` | 작업 상태 확인 |
| GET | `/api/result/{job_id}` | 분석 결과 조회 |

### 채팅 (요리 어시스턴트)
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/chat/start` | 요리 세션 시작 |
| GET | `/api/chat/session/{session_id}` | 세션 상태 조회 |
| POST | `/api/chat/message` | 메시지 전송 (이미지 포함 가능) |
| POST | `/api/chat/session/{id}/complete-step/{n}` | 단계 완료 처리 |

### 테스트 (개발용)
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/api/test/download` | 다운로드 테스트 |
| POST | `/api/test/subtitle` | 자막 테스트 |
| POST | `/api/test/stt` | STT 테스트 |
| POST | `/api/test/transcript` | 자막+STT 통합 테스트 |
| POST | `/api/test/llm` | LLM 파싱 테스트 (video_id) |
| POST | `/api/test/full` | 전체 파이프라인 테스트 |
| GET | `/api/test/cache/{video_id}` | 캐시된 결과 조회 |

## 프로젝트 구조

```
ktb-short/
├── backend/
│   ├── main.py                    # 앱 진입점
│   ├── app/
│   │   ├── main.py                # FastAPI 앱 팩토리
│   │   ├── config.py              # 설정
│   │   ├── routers/               # 컨트롤러 (MVC)
│   │   │   ├── analyze.py         # 레시피 분석 API
│   │   │   ├── chat.py            # 채팅 API (GPT-4o Vision)
│   │   │   ├── health.py          # 헬스 체크
│   │   │   └── test.py            # 테스트 API
│   │   ├── schemas/               # DTO (MVC)
│   │   │   ├── analyze.py
│   │   │   └── test.py
│   │   └── utils/                 # 유틸리티
│   │       └── logger.py
│   ├── services/                  # 비즈니스 로직
│   │   ├── youtube.py             # YouTube 다운로드 + 자막
│   │   ├── transcribe.py          # Whisper STT
│   │   └── recipe_parser.py       # GPT-4o 레시피 파싱
│   ├── data/                      # 데이터 저장소
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.tsx               # React 진입점 + 라우팅
│   │   ├── App.tsx                # 메인 페이지
│   │   ├── pages/
│   │   │   └── ChatRoom.tsx       # 채팅 페이지
│   │   ├── components/
│   │   │   ├── VideoPlayer.tsx
│   │   │   └── RecipeResult.tsx
│   │   └── api/
│   │       └── index.ts           # API 클라이언트
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

## 라이선스

MIT
