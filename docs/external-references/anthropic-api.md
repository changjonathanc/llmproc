Vertex AI
Vertex AI API
Anthropic’s Claude models are now generally available through Vertex AI.

The Vertex API for accessing Claude is nearly-identical to the Messages API and supports all of the same options, with two key differences:

In Vertex, model is not passed in the request body. Instead, it is specified in the Google Cloud endpoint URL.
In Vertex, anthropic_version is passed in the request body (rather than as a header), and must be set to the value vertex-2023-10-16.
Vertex is also supported by Anthropic’s official client SDKs. This guide will walk you through the process of making a request to Claude on Vertex AI in either Python or TypeScript.

Note that this guide assumes you have already have a GCP project that is able to use Vertex AI. See using the Claude 3 models from Anthropic for more information on the setup required, as well as a full walkthrough.

​
Install an SDK for accessing Vertex AI
First, install Anthropic’s client SDK for your language of choice.


Python

TypeScript

pip install -U google-cloud-aiplatform "anthropic[vertex]"
​
Accessing Vertex AI
​
Model Availability
Note that Anthropic model availability varies by region. Search for “Claude” in the Vertex AI Model Garden or go to Use Claude 3 for the latest information.

​
API model names
Model	Vertex AI API model name
Claude 3 Haiku	claude-3-haiku@20240307
Claude 3 Sonnet	claude-3-sonnet@20240229
Claude 3 Opus (Public Preview)	claude-3-opus@20240229
Claude 3.5 Haiku	claude-3-5-haiku@20241022
Claude 3.5 Sonnet	claude-3-5-sonnet-v2@20241022
Claude 3.7 Sonnet	claude-3-7-sonnet@20250219
​
Making requests
Before running requests you may need to run gcloud auth application-default login to authenticate with GCP.

The following examples shows how to generate text from Claude 3 Haiku on Vertex AI:


Python

TypeScript

cURL

from anthropic import AnthropicVertex

project_id = "MY_PROJECT_ID"
# Where the model is running. e.g. us-central1 or europe-west4 for haiku
region = "MY_REGION"

client = AnthropicVertex(project_id=project_id, region=region)

message = client.messages.create(
    model="claude-3-7-sonnet@20250219",
    max_tokens=100,
    messages=[
        {
            "role": "user",
            "content": "Hey Claude!",
        }
    ],
)
print(message)
