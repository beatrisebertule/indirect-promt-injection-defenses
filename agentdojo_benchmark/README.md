# How to Run agentDojo

Build the docker:\
`docker build -t yourusername/agentdojo-benchmark .`

Run the benchmark with your own API keys:\
`docker run -e OPENAI_API_KEY="sk-xxx" -e OPENAI_BASE_URL="https://chat.science.ru.nl/ollama/v1" yourusername/agentdojo-benchmark`

# Additional Commands
Change the Docker file to run these.

To run with no attack:\
`python -m agentdojo.scripts.benchmark --model LOCAL --model-id qwen3.6:35b -s workspace`

To run with attack:\
`python -m agentdojo.scripts.benchmark --model LOCAL --model-id qwen3.6:35b -s workspace --attack tool_knowledge `

# Additional Variables
There are four task suites - worksapce, slack, travel, and banking.

There are many attacks suites...

Run `python -m agentdojo.scripts.benchmark --help` to see other options (all attack suites).