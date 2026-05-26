FROM python:3.12-slim

WORKDIR /app

RUN pip install agentdojo

COPY agent_pipeline.py /usr/local/lib/python3.12/site-packages/agentdojo/agent_pipeline/agent_pipeline.py
COPY local_llm.py /usr/local/lib/python3.12/site-packages/agentdojo/agent_pipeline/llms/local_llm.py
COPY defenses.py /usr/local/lib/python3.12/site-packages/agentdojo/agent_pipeline/defenses.py

ENTRYPOINT ["python", "-m", "agentdojo.scripts.benchmark"]
