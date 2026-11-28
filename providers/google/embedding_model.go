package google

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"strings"

	"charm.land/fantasy"
	"google.golang.org/genai"
)

type embeddingModel struct {
	modelID  string
	provider string
	client   *genai.Client
	backend  genai.Backend
}

var _ fantasy.EmbeddingModelProvider = (*provider)(nil)

// EmbeddingModel implements fantasy.EmbeddingModelProvider. The Gemini API and
// Vertex AI backends are both supported, and the SDK routes each request by
// model ID. There is no default model.
//
// What a model accepts is decided by the service, not by this package.
// gemini-embedding-2 embeds text, images, audio, video and PDFs, turns the
// parts of one input into one vector, and rejects a task type.
// gemini-embedding-001 embeds text only and takes a task type.
//
// On Vertex AI the SDK sends models whose ID has neither "gemini" nor "maas",
// such as text-embedding-005, to the :predict endpoint. For those it only reads
// the text of the first part of each input and drops the other parts without an
// error, and it fails on an input whose first part is a file.
func (a *provider) EmbeddingModel(ctx context.Context, modelID string) (fantasy.EmbeddingModel, error) {
	client, err := a.newClient(ctx)
	if err != nil {
		return nil, err
	}
	return &embeddingModel{
		modelID:  modelID,
		provider: a.options.name,
		client:   client,
		backend:  a.options.backend,
	}, nil
}

// Model implements fantasy.EmbeddingModel.
func (m *embeddingModel) Model() string {
	return m.modelID
}

// Provider implements fantasy.EmbeddingModel.
func (m *embeddingModel) Provider() string {
	return m.provider
}

// Embed implements fantasy.EmbeddingModel. All inputs go out in one request,
// which is neither split nor retried, and the vectors are returned as the
// service made them, without normalizing shortened ones.
func (m *embeddingModel) Embed(ctx context.Context, call fantasy.EmbeddingCall) (*fantasy.EmbeddingResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, err
	}
	options, err := embeddingProviderOptions(call)
	if err != nil {
		return nil, err
	}
	// NOTE: The SDK would fail these on the Gemini API backend with a plain
	// error, which would not look like an invalid argument.
	if m.backend != genai.BackendVertexAI {
		if options.DocumentOCR != nil {
			return nil, invalidEmbeddingArgument("the document_ocr option needs the Vertex AI backend (google.WithVertex)")
		}
		if options.AudioTrackExtraction != nil {
			return nil, invalidEmbeddingArgument("the audio_track_extraction option needs the Vertex AI backend (google.WithVertex)")
		}
	}
	if call.Dimensions != nil && *call.Dimensions > math.MaxInt32 {
		return nil, invalidEmbeddingArgument("dimensions must be at most %d, got %d", math.MaxInt32, *call.Dimensions)
	}
	contents, err := embeddingContents(call.Inputs)
	if err != nil {
		return nil, err
	}

	config := &genai.EmbedContentConfig{
		DocumentOcr:          options.DocumentOCR,
		AudioTrackExtraction: options.AudioTrackExtraction,
	}
	if options.TaskType != nil {
		config.TaskType = *options.TaskType
	}
	if options.Title != nil {
		config.Title = *options.Title
	}
	if call.Dimensions != nil {
		config.OutputDimensionality = new(int32(*call.Dimensions))
	}

	// NOTE: The SDK sends the request once because the client has no retry
	// options, so retrying is left to the caller.
	response, err := m.client.Models.EmbedContent(withEmbeddingCallUA(ctx, call), m.modelID, contents, config)
	if err != nil {
		return nil, toProviderErr(err)
	}

	return m.toEmbeddingResponse(response, len(contents))
}

func embeddingProviderOptions(call fantasy.EmbeddingCall) (*EmbeddingProviderOptions, error) {
	v, ok := call.ProviderOptions[Name]
	if !ok {
		return &EmbeddingProviderOptions{}, nil
	}
	options, ok := v.(*EmbeddingProviderOptions)
	if !ok {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "google provider options should be *google.EmbeddingProviderOptions"}
	}
	return options, nil
}

// embeddingContents turns every input into one content, so the service returns
// one vector per input. The role is left empty, as embeddings have no turns.
func embeddingContents(inputs []fantasy.EmbeddingInput) ([]*genai.Content, error) {
	contents := make([]*genai.Content, len(inputs))
	for i, input := range inputs {
		parts := make([]*genai.Part, len(input.Parts))
		for j, part := range input.Parts {
			if text, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
				parts[j] = &genai.Part{Text: text.Text}
				continue
			}
			file, ok := fantasy.AsMessagePart[fantasy.FilePart](part)
			if !ok {
				// NOTE: EmbeddingCall.Validate already rejected other part types,
				// so this only guards against that check changing.
				return nil, invalidEmbeddingArgument("input %d part %d: unsupported part type, use text or files", i, j)
			}
			// NOTE: This is the only check on files. Whether the model can embed
			// a given kind of file is left to the service, so gemini-embedding-001
			// still gets images and the service answers with its own error.
			if !isEmbeddingMediaType(file.MediaType) {
				return nil, invalidEmbeddingArgument("input %d part %d: media type %q is not supported, use image/*, audio/*, video/* or application/pdf", i, j, file.MediaType)
			}
			parts[j] = &genai.Part{InlineData: &genai.Blob{Data: file.Data, MIMEType: file.MediaType}}
		}
		contents[i] = &genai.Content{Parts: parts}
	}
	return contents, nil
}

func isEmbeddingMediaType(mediaType string) bool {
	mediaType = strings.ToLower(mediaType)
	return strings.HasPrefix(mediaType, "image/") ||
		strings.HasPrefix(mediaType, "audio/") ||
		strings.HasPrefix(mediaType, "video/") ||
		mediaType == "application/pdf"
}

func (m *embeddingModel) toEmbeddingResponse(response *genai.EmbedContentResponse, inputs int) (*fantasy.EmbeddingResponse, error) {
	if len(response.Embeddings) != inputs {
		return nil, invalidEmbeddingResponse("expected %d embeddings, got %d", inputs, len(response.Embeddings))
	}

	embeddings := make([][]float32, inputs)
	var statisticsTokens int64
	for i, embedding := range response.Embeddings {
		if embedding == nil {
			return nil, invalidEmbeddingResponse("embedding %d is missing", i)
		}
		if len(embedding.Values) == 0 {
			return nil, invalidEmbeddingResponse("embedding %d is empty", i)
		}
		embeddings[i] = embedding.Values
		if embedding.Statistics != nil {
			statisticsTokens += int64(embedding.Statistics.TokenCount)
		}
	}

	// NOTE: Vertex models routed to :predict report no usage metadata, only the
	// token count of every input, so the counts are summed. When the service
	// reports neither, the usage stays empty instead of being estimated.
	usage := fantasy.Usage{InputTokens: statisticsTokens, TotalTokens: statisticsTokens}
	if metadata := response.UsageMetadata; metadata != nil {
		usage = fantasy.Usage{
			InputTokens: int64(metadata.PromptTokenCount),
			TotalTokens: int64(cmp.Or(metadata.TotalTokenCount, metadata.PromptTokenCount)),
		}
	}

	return &fantasy.EmbeddingResponse{
		Embeddings: embeddings,
		// NOTE: The response carries no model, so this is the requested one.
		Model: m.modelID,
		Usage: usage,
	}, nil
}

func invalidEmbeddingArgument(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid argument", Message: fmt.Sprintf(format, args...)}
}

func invalidEmbeddingResponse(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid response", Message: fmt.Sprintf(format, args...)}
}
