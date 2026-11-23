FROM pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime

ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PYTHONUNBUFFERED=1
WORKDIR /opt/zero-systems

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-cache-dir ".[all]"

ENTRYPOINT ["zero-systems"]
