FROM python:3.12-slim

WORKDIR /app

# Install agentdojo
RUN pip install agentdojo

# Copy the patched agent_pipeline.py
COPY agent_pipeline.py /usr/local/lib/python3.12/site-packages/agentdojo/agent_pipeline/agent_pipeline.py

# Default command
CMD ["python", "-m", "agentdojo.scripts.benchmark", "--model", "LOCAL", "--model-id", "qwen3.6:35b", "-s", "workspace"]