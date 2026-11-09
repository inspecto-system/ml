# ml

ML-трек Inspecto: подсказка категории дефекта по фотографии (ADR-016). Методика, наборы данных и результаты — в [`platform/docs/ML_DOCUMENTATION.md`](https://github.com/inspecto-system/platform/blob/main/docs/ML_DOCUMENTATION.md).

| Каталог | Что внутри |
|---|---|
| `training/` | Аудит наборов данных, обучение и оценка модели (пакет `inspecto_ml`) |

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
