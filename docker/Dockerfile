# Dockerfile
FROM python:3.11-slim

WORKDIR /
COPY flfm/ app/flfm
COPY flfm/tests/data/ /app/data
COPY pyproject.toml /app/pyproject.toml

COPY requirements/build.txt /build.txt
COPY requirements/prd.txt /prd.txt
COPY requirements/pytorch.txt /pytorch.txt

RUN pip3 install -r prd.txt
RUN pip3 install -r build.txt

RUN pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Build.
RUN pip3 install app/
