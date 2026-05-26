# How to Run agentDojo on Docker

Build the docker:\
`docker build -t yourusername/agentdojo-benchmark .`

Run the benchmark with your own API keys (not attack, no defense):\
`docker run -e OPENAI_API_KEY="sk-xxx" -e OPENAI_BASE_URL="https://chat.science.ru.nl/ollama/v1" bbertule/agentdojo-benchmark --model LOCAL --model-id gemma4:31b -s workspace`

Run the benchmark with your own API keys (attack, no defense):\
`docker run -e OPENAI_API_KEY="sk-xxx" -e OPENAI_BASE_URL="https://chat.science.ru.nl/ollama/v1" bbertule/agentdojo-benchmark --model LOCAL --model-id gemma4:31b -s workspace --attack tool_knowledge`

Run the benchmark with your own API keys (no attack, defense):\
`docker run -e OPENAI_API_KEY="sk-xxx" -e OPENAI_BASE_URL="https://chat.science.ru.nl/ollama/v1" bbertule/agentdojo-benchmark --model LOCAL --model-id gemma4:31b -s workspace --defense delimiting`

Run the benchmark with your own API keys (attack, defense):\
`docker run -e OPENAI_API_KEY="sk-xxx" -e OPENAI_BASE_URL="https://chat.science.ru.nl/ollama/v1" bbertule/agentdojo-benchmark --model LOCAL --model-id gemma4:31b -s workspace --attack tool_knowledge --defense delimiting`

# How to run agentDojo with no Docker 
Install agentDojo:\
`pip install agentdojo`

For local (uni) model use change `agent_pipeline.py`:\
```
elif provider == "local":
	client = openai.OpenAI(
		api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
		base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
		timeout=120,
	)
	if model_id is None:
		model_id = _get_local_model_id(port)
	logging.info(f"Using local model: {model_id}")
	logging.info(f"Using tool delimiter: {tool_delimiter}")
	llm = LocalLLM(client, model_id, tool_delimiter=tool_delimiter)
```

# Run agentDojo with on user task (not the whole task suite):\
Run on one user task `user_task_0`:\
`python -m agentdojo.scripts.benchmark --model LOCAL --model-id qwen3.6:35b -s workspace -f -ut user_task_0`


# Additional Variables
There are four task suites - worksapce, slack, travel, and banking.

There are many attacks suites...

Run `python -m agentdojo.scripts.benchmark --help` to see other options (all attack suites).

