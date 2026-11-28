# Azure

* Go to https://portal.azure.com/ and log in (or sign up if you haven't already)
* Access https://ai.azure.com/
* Create a project / resource (or use an existing one)
* Copy and API key

## Embeddings

The provider reuses the OpenAI implementation, so it also implements
`fantasy.EmbeddingModelProvider`. Pass the name of your embedding deployment
as the model ID. Azure deployments only accept text inputs.

OpenRouter, Vercel and the OpenAI-compatible provider are built the same way
and expose the same method. Whether their backends serve `/embeddings` depends
on the service, and a missing endpoint comes back as a 404 error.
