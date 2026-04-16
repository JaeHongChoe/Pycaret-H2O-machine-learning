# DOAC Gastrointestinal Bleeding Prediction

위장관 출혈(`GIB`) 예측을 위한 **노트북 + Python 스크립트 분리형 정리 저장소**입니다.

이 저장소는 의료 정형 데이터 전처리, baseline 비교, AutoML 학습, 최종 모델 선택 과정을
한 곳에서 보기 쉽게 정리한 프로젝트입니다.

현재 기준으로:

- 원본 분석 흐름은 `doac.ipynb`
- 재실행 가능한 분리형 파이프라인은 `scripts/`, `src/`
- 최종 채택 모델 방향은 `H2O AutoML`
- 추가 노트북은 `archive/`로 이동한 실험 기록

즉, 이 저장소는 **`doac.ipynb`를 기준으로 흐름을 유지하되,
핵심 로직을 `.py`로 분리해 재현성과 가독성을 높인 구조**라고 보면 됩니다.

## 빠른 시작

### 1. 환경 설치

```bash
pip install -r requirements.txt
```

### 2. H2O AutoML 실행

```bash
python scripts/run_h2o_automl.py \
    --data-path /absolute/path/to/dataset_0914.xlsx
```

### 3. PyCaret baseline 실행

```bash
python scripts/run_pycaret_baseline.py \
    --data-path /absolute/path/to/dataset_0914.xlsx
```

위 예시의 Excel 파일은 저장소에 포함되어 있지 않으며,
의료 데이터 특성상 로컬에서 별도로 준비한 파일 경로를 넣어 실행하는 전제를 가집니다.

기본 결과는 `outputs/` 아래에 저장됩니다.

## 이 저장소의 핵심

- 의료 정형 데이터 기반 `GIB` 이진 분류
- 결측치 확인 및 범주형 결측 행 정리
- 불필요 변수 제거 후 train/test 분리
- PyCaret 기반 baseline 모델 비교
- H2O AutoML 기반 최종 학습 및 평가
- 결과물 CSV / JSON 저장 구조 추가

## 왜 이렇게 정리했는가

기존 `doac.ipynb`에는 아래 흐름이 한 노트북 안에 모두 들어 있었습니다.

1. Excel 데이터 로드
2. 결측치 확인 및 처리
3. feature 제거
4. train/test 분리
5. PyCaret baseline 비교
6. H2O AutoML 학습 및 평가

이 구조는 실험 기록으로는 충분하지만,
GitHub 포트폴리오나 재실행 관점에서는 로직이 노트북 셀에 흩어져 있다는 단점이 있었습니다.

그래서 이번 정리에서는:

- 노트북은 **분석 참조본**
- `src/`는 **핵심 로직**
- `scripts/`는 **실행 진입점**
- `archive/`는 **보조 실험 기록**

으로 역할을 분리했습니다.

## 메인 노트북에서 하는 일

`doac.ipynb`는 아래 흐름을 포함합니다.

### 1. 데이터 로드

- `dataset_0914.xlsx`를 읽어옵니다.
- 식별용 컬럼 `number`를 제거합니다.

### 2. 결측치 확인 및 처리

- `check_missing_col()`로 결측 컬럼을 확인합니다.
- `handle_na()`로 범주형 결측치가 있는 행을 제거합니다.

### 3. 변수 정리

노트북 기준 제거 대상:

- `Mortality`
- `Intracranial hemorrhage`
- `D6`

### 4. 데이터 분리

- 타깃 `GIB`를 `1/2 -> 1/0` 형태로 정리합니다.
- `train_test_split(..., test_size=0.3, random_state=15)`로 train/test를 분리합니다.

### 5. PyCaret baseline

- `setup()`
- `compare_models(sort='AUC')`
- `et`, `gbc`, `lr` 생성
- `tune_model()`
- `blend_models()`

이 단계는 **baseline 비교와 후보 탐색**이 목적입니다.

### 6. H2O AutoML

- `train_data`를 다시 `train/valid`로 분리
- `max_runtime_secs = 60 * 60 * 16`
- `exclude_algos = ['DRF', 'GLM']`
- validation / test 성능 확인

현재 저장소에서 최종적으로 채택한 방향은 **H2O AutoML**입니다.

## 새로 추가한 Python 구조

```text
Pycaret-H2O-machine-learning/
├── README.md
├── requirements.txt
├── doac.ipynb
├── archive/
│   ├── README.md
│   ├── DOAC_new.ipynb
│   └── DOAC_mljar.ipynb
├── scripts/
│   ├── run_h2o_automl.py
│   └── run_pycaret_baseline.py
└── src/
    └── doac_pipeline/
        ├── __init__.py
        ├── h2o_automl.py
        ├── metrics.py
        ├── preprocess.py
        ├── pycaret_baseline.py
        └── utils.py
```

### `src/doac_pipeline/preprocess.py`

전처리와 데이터 분리를 담당합니다.

포함 내용:
- Excel 로드
- 결측 컬럼 확인
- 범주형 결측 행 제거
- 기본 drop 컬럼 제거
- `GIB` 라벨 정리
- train/test, train/valid 분리
- 전처리 요약 JSON 생성용 메타데이터 정리

### `src/doac_pipeline/pycaret_baseline.py`

PyCaret baseline 비교 흐름을 함수형으로 분리한 모듈입니다.

포함 내용:
- `setup()`
- `compare_models()`
- 후보 모델 생성
- 튜닝
- 앙상블
- test 예측 저장용 결과 생성

### `src/doac_pipeline/h2o_automl.py`

최종 채택한 H2O AutoML 파이프라인을 분리한 모듈입니다.

포함 내용:
- H2O 클러스터 초기화
- train/valid frame 생성
- AutoML 학습
- validation / test metric 추출
- leaderboard / prediction / variable importance 저장용 결과 생성
- 예측 컬럼명 표준화 및 클러스터 종료 처리

### `scripts/run_pycaret_baseline.py`

PyCaret baseline을 CLI로 실행하는 진입점입니다.

예시:

```bash
python scripts/run_pycaret_baseline.py \
    --data-path /absolute/path/to/dataset_0914.xlsx
```

기본 산출물:
- `outputs/pycaret_baseline/preprocess_summary.json`
- `outputs/pycaret_baseline/compare_models.csv`
- `outputs/pycaret_baseline/test_metrics.csv`
- `outputs/pycaret_baseline/test_predictions.csv`
- `outputs/pycaret_baseline/run_summary.json`

주요 옵션:
- `--drop-column`: 추가로 제외할 feature 지정
- `--id-column`: 식별용 컬럼 지정
- `--tune-iterations`: 튜닝 반복 횟수 조정

### `scripts/run_h2o_automl.py`

최종 H2O AutoML 파이프라인을 CLI로 실행하는 진입점입니다.

예시:

```bash
python scripts/run_h2o_automl.py \
    --data-path /absolute/path/to/dataset_0914.xlsx
```

기본 산출물:
- `outputs/h2o_automl/preprocess_summary.json`
- `outputs/h2o_automl/leaderboard.csv`
- `outputs/h2o_automl/validation_metrics.json`
- `outputs/h2o_automl/test_metrics.json`
- `outputs/h2o_automl/validation_predictions.csv`
- `outputs/h2o_automl/test_predictions.csv`
- `outputs/h2o_automl/variable_importance.csv`
- `outputs/h2o_automl/run_summary.json`

주요 옵션:
- `--drop-column`: 추가로 제외할 feature 지정
- `--id-column`: 식별용 컬럼 지정
- `--exclude-algo`: 제외할 H2O 알고리즘 지정
- `--max-runtime-secs`: AutoML 탐색 시간 조정

## 처리 흐름

```text
Excel 데이터 로드
    ↓
식별 컬럼 제거
    ↓
결측 컬럼 확인 및 범주형 결측 행 제거
    ↓
불필요 변수 제거
    ↓
GIB 라벨 정리
    ↓
train / test 분리
    ↓
PyCaret baseline 비교
    ↓
H2O AutoML 재학습
    ↓
validation / test 성능 확인
    ↓
최종 모델 방향 결정
```

## 추천 탐색 순서

1. 저장소 전체 맥락은 이 README
2. 원본 분석 흐름은 `doac.ipynb`
3. 재실행 가능한 파이프라인은 `scripts/run_h2o_automl.py`
4. baseline 비교는 `scripts/run_pycaret_baseline.py`
5. 과거 실험 흔적은 `archive/`

## 산출물 해석

### `preprocess_summary.json`

- 원본/정리 후 데이터 shape
- 제거된 컬럼 목록
- train/test 타깃 분포
- 남은 feature 목록

### `leaderboard.csv`

- H2O AutoML이 validation 기준으로 정렬한 후보 모델 목록

### `validation_metrics.json`, `test_metrics.json`

- 최종 leader 모델의 accuracy / precision / recall / F1 / AUC 요약

### `test_predictions.csv`

- 샘플별 예측 라벨과 score 확인용 파일

## 실행 전제

현재 저장소에는 실행에 필요한 실제 데이터 파일이 포함되어 있지 않습니다.

그 이유는 본 프로젝트가 **의료 데이터 기반 예측 과제**였고,
환자 정보 및 민감정보 보호 이슈로 원본 데이터를 공개 저장소에 포함할 수 없기 때문입니다.
즉, 데이터 부재는 정리 누락이 아니라 **의도적인 비공개 처리**에 가깝습니다.

노트북/스크립트에서 참조하는 대표 파일:
- `dataset_0914.xlsx`

따라서 이 저장소는
**즉시 재실행 가능한 완성형 패키지라기보다, 분석 흐름·코드 구조·실험 방향을 공유하기 위한 정리본**에 가깝습니다.

공개 범위에는 아래만 포함합니다.

- 전처리/학습/평가 코드
- 노트북 기반 분석 흐름
- 실행용 스크립트와 결과 저장 구조
- 실험 정리 문서

공개 범위에서 제외한 항목은 아래와 같습니다.

- 원본 의료 데이터
- 환자 단위 레코드
- 외부 검증 데이터
- 민감한 중간 산출물

## 주요 라이브러리

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `openpyxl`
- `pycaret`
- `h2o`
- `xgboost`
- `lightgbm`
- `catboost`

## 메모

- 최종 채택 모델 방향은 `H2O AutoML`입니다.
- `PyCaret` 구간은 baseline 비교와 후보 탐색 용도로 보는 것이 자연스럽습니다.
- `DOAC_new.ipynb`, `DOAC_mljar.ipynb`는 `archive/`로 이동한 보조 실험 기록입니다.
- 추후 더 정리하려면, feature selection 근거와 external validation 흐름을 별도 문서로 분리하는 것이 좋습니다.
