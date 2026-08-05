FROM python:3.14-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-c", "import torch, cv2, mmcv, mmengine, mmdet; print('Hello World from Docker!'); print('PyTorch:', torch.__version__); print('OpenCV:', cv2.__version__); print('MMDetection:', mmdet.__version__)"]
