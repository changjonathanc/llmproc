import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.count_tokens(
    betas=["token-counting-2024-11-01"],
    model="claude-3-5-sonnet-20241022",
    system="You are a scientist",
    messages=[{"role": "user", "content": "Hello, Claude"}],
)

print(response)

