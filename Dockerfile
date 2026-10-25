# Hunt Autopsy — API + web app. CPU image: everything after the event layer runs on CPU.
# Perception (YOLO / CoTracker3 / TAPNet) runs offline on a GPU box and writes cached episodes.
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY events events
COPY sim sim
COPY explain explain
COPY narrate narrate
COPY football football
COPY agents agents
COPY app app
COPY evals evals
COPY ingest ingest
COPY perception perception
# WITH_VISION=1 (default) adds CPU PyTorch + Ultralytics so uploaded clips can be analysed in the container.
ARG WITH_VISION=1
RUN pip install uv && uv pip install --system ".[azure,video]" && \
    if [ "$WITH_VISION" = "1" ]; then \
      uv pip install --system torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
      uv pip install --system ultralytics "lap>=0.5.12" && \
      python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"; \
    fi
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
# Pre-train and cache the explainers at build time so cold starts are fast.
RUN python -c "from agents.models import shot_autopsy, hunt_library; shot_autopsy(); hunt_library()"
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
