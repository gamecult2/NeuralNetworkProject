from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "$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"
)

completion = client.chat.completions.create(
  model="mistralai/mistral-large",
  messages=[{"role":"user","content":""}],
  temperature=0.5,
  top_p=1,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

