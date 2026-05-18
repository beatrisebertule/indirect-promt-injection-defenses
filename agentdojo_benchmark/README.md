# How to Run agentDojo

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

# Additional Variables
There are four task suites - worksapce, slack, travel, and banking.

There are many attacks suites...

Run `python -m agentdojo.scripts.benchmark --help` to see other options (all attack suites).