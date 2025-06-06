FROM cnstark/pytorch:2.2.1-py3.10.15-cuda12.1.0-ubuntu22.04

LABEL maintainer="mockvox"
LABEL version="mockvox-20250513"
LABEL description="Docker image for MockVox"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg aria2 git && \
    rm -rf /var/lib/apt/lists/* 

WORKDIR /mockvox
COPY ./src/ /mockvox/src/
COPY ./Docker/ /mockvox/Docker/
COPY ./.env.sample ./pyproject.toml /mockvox/
RUN pip install --no-cache-dir -e .

EXPOSE 5000