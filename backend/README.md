# Recipe Backend

## 실행 방법

### 로컬 실행

```bash
# 가상환경 생성 및 활성화
python3.11 -m venv .venv
source .venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정

# 서버 실행
uvicorn app.main:app --reload --port 8000
```

### Docker 실행

```bash
# 환경변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정

# 빌드 및 실행
docker-compose up -d --build

# 로그 확인
docker-compose logs -f

# 중지
docker-compose down
```

### API 문서

http://localhost:8000/docs
