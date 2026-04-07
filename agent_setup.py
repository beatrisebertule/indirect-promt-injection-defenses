from mistralai.client import Mistral

client = Mistral(api_key="ukdDF1MXzg1NtOv0pdI5MmVyYBbbXj9t")

inputs = [
    {"role":"user","content":"Hello!"}
]

response = client.beta.conversations.start(
    agent_id="ag_019d2096209770d6a272f4fcf189750b",
    agent_version=0,
    inputs=inputs,
)

print(response)