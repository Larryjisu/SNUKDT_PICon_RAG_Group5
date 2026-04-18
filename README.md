# Jiyu Kim Persona Server + PiCon Evaluation Wrapper

이 저장소는 **내가 직접 작성/수정한 부분만** 남긴 최소 공개용 버전입니다.

포함된 내용:
- `rag_persona_server.py`: 김지유 페르소나 RAG 서버
- `run_eval.py`: PiCon 평가 실행 래퍼
- `.env.example`: 필요한 환경 변수 예시
- `NOTICE.md`: 원본 PiCon 의존성 및 출처 안내

포함하지 않은 내용:
- 원본 `PiCon` 프레임워크 전체 소스
- 웹 인터뷰 프론트엔드
- 예제 데이터, 결과 로그, 실험 부산물

## 왜 이렇게 분리했나

`PiCon`은 원본 프레임워크이고, 이 저장소의 핵심은 그 위에서 동작하는 **김지유 페르소나 서버**와 **평가 실행 래퍼**입니다.
그래서 공개 저장소에는 원본 전체를 다시 올리지 않고, **내 기여분만 남기는 구조**로 정리했습니다.

## 폴더 구조

```text
jiyu_picon_integration_minimal/
├── README.md
├── NOTICE.md
├── .gitignore
├── .env.example
├── requirements.txt
├── rag_persona_server.py
└── run_eval.py
```

## 사전 준비

### 1) 가상환경

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) 이 저장소용 의존성 설치

```bash
pip install -r requirements.txt
```

### 3) PiCon 별도 준비

아래 둘 중 하나로 준비하세요.

#### 방법 A. 로컬에 PiCon clone 후 editable install

```bash
git clone <PICON_REPO_URL>
cd picon
pip install -e .
```

그 다음 다시 이 저장소 루트로 와서 실행하면 됩니다.

#### 방법 B. 설치하지 않고 로컬 경로만 지정

PiCon 소스 루트 경로를 환경 변수로 넘길 수 있습니다.

```bash
export PICON_SOURCE_DIR=/absolute/path/to/picon
```

`PICON_SOURCE_DIR`는 **`pyproject.toml`이 있는 PiCon 저장소 루트**를 가리켜야 합니다.

## 환경 변수

`.env.example`를 `.env`로 복사해서 사용하세요.

최소 필요 값:
- `OPENAI_API_KEY`

선택 값:
- `SERPER_API_KEY`
- `GOOGLE_GEOCODE`
- `REDIS_URL`

## 실행

### 서버 단독 실행

```bash
python rag_persona_server.py --port 8001 --model gpt-4o
```

### 평가 실행

```bash
python run_eval.py --turns 30 --sessions 2 --model gpt-4o
```

`run_eval.py`가 하는 일:
1. 로컬 persona server 실행
2. 서버 준비 확인
3. `PiCon`의 `run()` 호출
4. 결과 JSON 및 요약 저장

결과는 기본적으로 `outputs/jiyu_kim/` 아래 저장됩니다.

## 주의

- 이 저장소만으로는 `PiCon` 평가가 바로 돌아가지 않습니다.
- 반드시 **원본 PiCon을 별도 설치하거나 경로로 연결**해야 합니다.
- 공개 포트폴리오 용도에서는 이 구조가 더 안전하고, 내가 한 작업도 더 분명하게 보입니다.

## 내 기여 범위

- 김지유 페르소나 지식 베이스 및 응답 서버 구성
- PiCon 평가 실행용 래퍼 작성
- 프로젝트 구조 최소화 및 공개용 정리
