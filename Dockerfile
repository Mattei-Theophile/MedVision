# Image autonome, pour Home Assistant « Container » / « Core » ou tout serveur Docker.
# Prérequis : générer uv.lock avec `uv lock` et le versionner.
FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dépendances d'abord (couche mise en cache), puis le code.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    MEDVISION_GESTURES_DIR=/data/gestures \
    MEDVISION_MODEL_PATH=/data/models/hand_landmarker.task
VOLUME /data

ENTRYPOINT ["medvision"]
CMD ["run"]
