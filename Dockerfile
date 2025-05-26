FROM cnstark/pytorch:2.2.1-py3.10.15-cuda12.1.0-ubuntu22.04

LABEL maintainer="18702837579@163.com"
LABEL version="mockvox-20250513"
LABEL description="Docker image for MockVox"

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg aria2 git && \
    rm -rf /var/lib/apt/lists/* 
     

WORKDIR /mockvox
COPY . /mockvox   
RUN pip install -e .[dev] 
# ARG IMAGE_TYPE=chinese

# RUN if [ "$IMAGE_TYPE" == "chinese" ]; then \
#         chmod +x /mockvox/Docker/chineseDownload.sh && \
#         /mockvox/Docker/chineseDownload.sh && \
#         python /mockvox/Docker/chineseDownload.py;\
#     fi

VOLUME /mockvox/pretrained    

EXPOSE 5000

CMD ["python", "/mockvox/src/mockvox/main.py"]