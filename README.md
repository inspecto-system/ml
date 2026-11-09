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
