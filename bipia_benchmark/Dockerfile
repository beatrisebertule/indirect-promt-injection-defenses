FROM python:3.10-slim

WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ git jq

# Copy BIPIA (overridden by volume in Compose, but needed for build process)
COPY BIPIA /app/BIPIA

WORKDIR /app/BIPIA
# Strip vLLM -> container will not host models locally, vLLM takes ages to install
RUN sed -i '/vllm/Id' pyproject.toml || true
RUN sed -i '/vllm/Id' setup.py || true
RUN find . -type f -name "*.py" -exec sed -i '/import vllm/d' {} +
RUN find . -type f -name "*.py" -exec sed -i '/from vllm/d' {} +

RUN pip install --no-cache-dir -e .
RUN pip install requests pyyaml pandas tqdm torch

WORKDIR /app
COPY benchmark_defense.py /app/benchmark_defense.py

CMD ["python", "-u", "benchmark_defense.py"]
