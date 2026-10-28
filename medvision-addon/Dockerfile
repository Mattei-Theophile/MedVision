ARG BUILD_FROM=python:3.12-slim-bookworm
FROM ${BUILD_FROM}

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

# Bibliothèques système requises par OpenCV / MediaPipe.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 git \
    && rm -rf /var/lib/apt/lists/*

# Le code de l'add-on est installé depuis le dépôt GitHub, à la version indiquée dans build.yaml.
ARG MEDVISION_REF=master
RUN uv pip install --system --no-cache \
    "medvision @ git+https://github.com/Mattei-Theophile/MedVision@${MEDVISION_REF}"

CMD ["python", "-m", "medvision.addon"]
