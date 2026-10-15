FROM python:3.11-slim

ARG PROJECT_EXTRAS=vision,fl,dev,serve,eval

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONPATH=/workspace/src \
    HOME=/home/agrifed

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        gcc \
        git \
        libglib2.0-0 \
        libgomp1 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

ARG USER_UID=1001
ARG USER_GID=1001

RUN groupadd --gid 1000 agrifed 2>/dev/null || true \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash agrifed 2>/dev/null || true \
    && if [ "${USER_UID}" != "1000" ]; then \
        groupadd --gid ${USER_GID} agrifed-user 2>/dev/null || true && \
        useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash agrifed-user 2>/dev/null || true ; \
    fi \
    && mkdir -p /workspace /home/agrifed/.cache \
    && chown -R 1000:1000 /workspace /home/agrifed

COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests

RUN python -m pip install --upgrade pip setuptools wheel \
    && if [ -n "${PROJECT_EXTRAS}" ]; then python -m pip install ".[${PROJECT_EXTRAS}]"; else python -m pip install .; fi

USER agrifed

CMD ["bash"]