# Google Provider

This document describes how to get an API keys for Google Gemini and Vertex.

## Gemini

Simply navigate to [this page](https://aistudio.google.com/apikey) in the
Google AI Studio and create a new API key.

## Vertex

### Install `gcloud`

Install the `gcloud` command line tool. Install via Homebrew, Nix, or download
it from [here](https://cloud.google.com/sdk/docs/install).

```bash
# Homebrew
brew install --cask google-cloud-sdk

# Nix
nix-env -iA nixpkgs.google-cloud-sdk
```

### Authenticate

Then authenticate with your Google account:

```bash
gcloud auth login
```

### Create And Setup Project

Navigate here to create a new project if you haven't already:
https://console.cloud.google.com/projectcreate

Alternatively, you can create a new project via the command line:

```bash
gcloud projects create {YOUR_PROJECT_ID} --name="{YOUR_PROJECT_NAME}"
```

Set the project on your machine:

```bash
gcloud config set project {YOUR_PROJECT_ID}
```

Enable the Vertex AI API:

```bash
gcloud services enable aiplatform.googleapis.com
```

### Setup Env

Finally, you need to run this command to ensure that libraries will be able to
find your credentials.

```bash
gcloud auth application-default login
```

## Embeddings

The provider implements `fantasy.EmbeddingModelProvider` on both the Gemini API
and Vertex. There is no default model, so pass a model ID.

```go
provider, err := google.New(google.WithGeminiAPIKey(apiKey))
if err != nil {
	return err
}
embeddings, ok := provider.(fantasy.EmbeddingModelProvider)
if !ok {
	return errors.New("provider has no embedding models")
}
model, err := embeddings.EmbeddingModel(ctx, "gemini-embedding-2")
if err != nil {
	return err
}

response, err := model.Embed(ctx, fantasy.EmbeddingCall{
	// One vector per input, in the same order.
	Inputs: fantasy.TextEmbeddingInputs("first text", "second text"),
	// Optional: shorten the vectors.
	Dimensions: new(int64(768)),
})
```

All inputs go out in one request. The request is not split and not retried.

### Models and modalities

What a model accepts is decided by the service. The provider only checks that a
file is an image, audio, video or PDF, and it never keeps a table of models, so
an unsupported combination comes back as an error from the service.

| | `gemini-embedding-2` | `gemini-embedding-001` |
|---|---|---|
| Inputs | text, images, audio, video, PDF | text only |
| Several parts in one input | one vector for the whole input | not applicable |
| Task type | rejected, write the task into the text | supported |
| Dimensions | 128 to 3072, shortened vectors are normalized | 128 to 3072, shortened vectors are not normalized |

An input with several parts is how you embed an image together with a caption:

```go
input := fantasy.EmbeddingInput{Parts: []fantasy.MessagePart{
	fantasy.TextPart{Text: "a red bicycle"},
	fantasy.FilePart{Data: png, MediaType: "image/png"},
}}
```

`gemini-embedding-001` takes a task type through the provider options. The title
only has an effect with `TaskTypeRetrievalDocument`.

```go
call := fantasy.EmbeddingCall{
	Inputs: fantasy.TextEmbeddingInputs("a document"),
	ProviderOptions: google.NewEmbeddingProviderOptions(&google.EmbeddingProviderOptions{
		TaskType: new(google.TaskTypeRetrievalDocument),
		Title:    new("Release notes"),
	}),
}
```

### Vertex

- `DocumentOCR` and `AudioTrackExtraction` are sent on the Vertex `:embedContent`
  and `:batchEmbedContents` routes. The Gemini Developer API rejects these
  options before sending a request.
- If the final segment of the model ID contains neither `gemini` nor `maas`
  (case-sensitive), the SDK uses `:predict`, as for `text-embedding-005`.
  This route uses only the text of the first part of each input. The response
  warns about ignored later parts and about either option above if set.
- A warning identifies each input the service reports as truncated. On
  `:predict`, usage is empty unless statistics are available for every input.
- Models whose final name segment contains `maas` embed one input per request.
