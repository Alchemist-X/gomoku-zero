# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm AS builder

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv

RUN python -m venv "${VIRTUAL_ENV}"
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==2.13.0"

WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install '.[cloud]'


FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    WEB_CONCURRENCY=1 \
    GOMOKU_CONFIG=/app/configs/production.json \
    GOMOKU_STATIC_DIR=/app/static

RUN groupadd --gid 10001 gomoku \
    && useradd --uid 10001 --gid gomoku --create-home --shell /usr/sbin/nologin gomoku

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY configs ./configs
COPY static ./static
COPY deploy ./deploy
RUN chmod 0755 /app/deploy/*.sh \
    && chown -R gomoku:gomoku /app

USER gomoku
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn gomoku_zero.api:app --host 0.0.0.0 --port \"${PORT:-8080}\" --workers \"${WEB_CONCURRENCY:-1}\""]
