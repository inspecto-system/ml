# ml

ML-трек Inspecto: подсказка категории дефекта по фотографии (ADR-016). Методика, наборы данных и результаты — в [`platform/docs/ML_DOCUMENTATION.md`](https://github.com/inspecto-system/platform/blob/main/docs/ML_DOCUMENTATION.md).

| Каталог | Что внутри |
|---|---|
| `training/` | Аудит наборов данных, обучение и оценка модели (пакет `inspecto_ml`) |
| `service/` | ml-service — подсказки категорий по фото (пакет `ml_service`, контракт `platform/api/ml.yaml`), образ — `Dockerfile` в корне |

Наборы данных, собственные фотографии и веса моделей в git не попадают (`.gitignore`): они хранятся только на ПК владельца в каталоге `data/`.

## Проверки

Проект управляется [uv](https://docs.astral.sh/uv/): версия uv закреплена в `pyproject.toml`, версия Python — в `.python-version`, точные версии зависимостей — в `uv.lock`. CI — общий шаблон `platform/.github/workflows/python.yml`.

```sh
cd training
uv sync --locked
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest --cov
```

Без локального Python — в контейнере (Git Bash, из корня репозитория):

```sh
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)":/repo -w /repo/training \
  -e UV_PYTHON_DOWNLOADS=never -e UV_PROJECT_ENVIRONMENT=/venv python:3.12-slim \
  sh -c 'pip install -q uv==0.13.7 && uv sync --locked && uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest --cov'
```

## Данные

Архивы скачиваются вручную с официальных страниц (ссылки и лицензии — в ML_DOCUMENTATION.md) и кладутся без изменений:

| Набор | Файл |
|---|---|
| BD3 (Kaggle, нужен вход) | `data/raw/bd3/archive.zip` |
| SDNET2018 (сайт USU, проверка Cloudflare в браузере) | `data/raw/sdnet2018/DATA_Maguire_20180517_ALL.zip` |
| FBD (GitHub, две части архива) | `data/raw/fbd/train-test-split.zip` и `.z01` |

FBD — многотомный zip, Python его не читает; массивы извлекаются 7-Zip в `data/interim/fbd/`:

```sh
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)/data":/data alpine:3.22 \
  sh -c 'apk add -q 7zip && cd /data/interim/fbd && 7z x -y /data/raw/fbd/train-test-split.zip'
```

## Аудит наборов (ML-01)

Считает классы, размеры и форматы, битые файлы, точные копии (SHA-256) и почти одинаковые кадры (разностный хеш по двум направлениям, 128 бит, расстояние Хэмминга ≤ 4) внутри класса, между классами и между наборами. Архивы читаются как есть; отчёт — JSON с отпечатками исходных файлов.

```sh
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)":/repo -w /repo/training \
  -e UV_PYTHON_DOWNLOADS=never -e UV_PROJECT_ENVIRONMENT=/venv python:3.12-slim \
  sh -c 'pip install -q uv==0.13.7 && uv sync --locked --no-dev && uv run --no-dev python -m inspecto_ml.audit \
    --bd3 ../data/raw/bd3/archive.zip --sdnet ../data/raw/sdnet2018/DATA_Maguire_20180517_ALL.zip \
    --fbd ../data/interim/fbd/train-test-split --out ../data/reports/audit.json'
```

Метки FBD хранятся числами 0–3 без описания порядка; соответствие (0 — spalling, 1 — crack, 2 — stain, 3 — vegetation) восстановлено по числу изображений каждого класса из README FBD и проверено просмотром примеров.

## Разбиение (ML-03)

Таблица соответствия классов (ML-02) — `training/src/inspecto_ml/class_mapping.toml`. Разбиение убирает повторы и противоречивые копии, держит почти одинаковые кадры в одной части, делит каждую категорию 70/15/15 (обучение / проверка / тест) с фиксированным seed; FBD целиком — в часть `external` (проверка на другом домене). Манифест хранит версию — SHA-256 записей; результаты ссылаются на неё.

```sh
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)":/repo -w /repo/training \
  -e UV_PYTHON_DOWNLOADS=never -e UV_PROJECT_ENVIRONMENT=/venv python:3.12-slim \
  sh -c 'pip install -q uv==0.13.7 && uv sync --locked --no-dev && uv run --no-dev python -m inspecto_ml.split_cli \
    --bd3 ../data/raw/bd3/archive.zip --sdnet ../data/raw/sdnet2018/DATA_Maguire_20180517_ALL.zip \
    --fbd ../data/interim/fbd/train-test-split --out ../data/splits/split.json'
```

## Baseline (ML-03)

Мажоритарный класс и логистическая регрессия на признаках замороженной ResNet-50 (ImageNet, torchvision `IMAGENET1K_V2`; веса скачиваются один раз в `TORCH_HOME`). Признаки кэшируются по версии разбиения в `data/features`.

```sh
MSYS_NO_PATHCONV=1 docker run --rm -v "$(pwd -W)":/repo -w /repo/training -e TORCH_HOME=/repo/data/cache/torch \
  -e UV_PYTHON_DOWNLOADS=never -e UV_PROJECT_ENVIRONMENT=/venv python:3.12-slim \
  sh -c 'pip install -q uv==0.13.7 && uv sync --locked --no-dev && uv run --no-dev python -m inspecto_ml.baseline_cli \
    --bd3 ../data/raw/bd3/archive.zip --sdnet ../data/raw/sdnet2018/DATA_Maguire_20180517_ALL.zip \
    --fbd ../data/interim/fbd/train-test-split --split ../data/splits/split.json \
    --features ../data/features --out ../data/runs/baseline.json'
```

## ml-service (T-601)

FastAPI + ONNX Runtime (CPU). `POST /internal/v1/classify {photoId}` — принимает только сервисный токен core-service со scope `ml.classify` (RS256, JWKS core, issuer, audience `inspecto-internal`), сам скачивает вариант `preview` фото из media-service своим токеном (клиент `ml-service`, scope `photos.read`) и возвращает до трёх категорий; «нет дефекта» или уверенность ниже порога модели — пустой список. Ошибки — RFC 9457 (`application/problem+json`), заголовок `X-Request-Id`. `GET /health/ready` — `503`, пока модель не загружена.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `ML_PORT` | `8084` | Порт |
| `ML_CORE_URL` | `http://127.0.0.1:8081` | core-service: JWKS и выдача сервисного токена |
| `ML_CORE_CLIENT_SECRET` | — (обязательна, ≥ 32 символов) | Секрет клиента `ml-service` в core |
| `ML_JWKS_URL` | `{ML_CORE_URL}/api/v1/.well-known/jwks.json` | Ключи подписи core |
| `ML_JWT_ISSUER`, `ML_JWT_INTERNAL_AUDIENCE` | `inspecto-core`, `inspecto-internal` | Проверка токена |
| `ML_MEDIA_URL` | `http://127.0.0.1:8082` | media-service |
| `ML_MODEL_DIR` | `/models/current` | Каталог с `model.onnx` и `model.json` (`inspecto_ml.export_cli`) |

Модель в образ не входит — каталог монтируется (только для чтения):

```sh
docker build -t inspecto/ml-service .
MSYS_NO_PATHCONV=1 docker run --rm -p 8084:8084 -e ML_CORE_CLIENT_SECRET=... \
  -v "$(pwd -W)/data/models/resnet50-logreg-917dcdb8":/models/current:ro inspecto/ml-service
```

Проверки — как у `training/`, из каталога `service/`.
