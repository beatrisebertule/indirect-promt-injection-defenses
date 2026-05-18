FROM python:3.12-slim

WORKDIR /app

RUN pip install agentdojo

# Patch the agent_pipeline.py with your custom local model support
COPY agent_pipeline.py /usr/local/lib/python3.12/site-packages/agentdojo/agent_pipeline/agent_pipeline.py

# Copy defense files
COPY defenses.py /usr/local/lib/python3.12/site-packages/agentdojo/agent_pipeline/defenses.py

# Default command - no defense, no attack
CMD ["python", "-m", "agentdojo.scripts.benchmark", "--model", "LOCAL", "--model-id", "qwen3.6:35b", "-s", "workspace"]