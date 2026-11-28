package openai

import (
	"cmp"
	"context"
	"fmt"
	"math"

	"charm.land/fantasy"
	"github.com/charmbracelet/openai-go"
	"github.com/charmbracelet/openai-go/packages/param"
)

// NOTE: the endpoint rejects more inputs than this in one request. Splitting
// larger batches is left to the caller.
const maxEmbeddingInputs = 2048

type embeddingModel struct {
	modelID  string
	provider string
	client   openai.Client
}

var _ fantasy.EmbeddingModelProvider = (*provider)(nil)

func newEmbeddingModel(modelID, provider string, client openai.Client) embeddingModel {
	return embeddingModel{modelID: modelID, provider: provider, client: client}
}

// Model implements fantasy.EmbeddingModel.
func (m embeddingModel) Model() string {
	return m.modelID
}

// Provider implements fantasy.EmbeddingModel.
func (m embeddingModel) Provider() string {
	return m.provider
}

// NOTE: callers choose batch boundaries and retry policy. Embed does not split
// batches, and SDK retries are disabled by default. Negotiating request
// compression can still resend a batch.
func (m embeddingModel) Embed(ctx context.Context, call fantasy.EmbeddingCall) (*fantasy.EmbeddingResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, err
	}
	options, err := embeddingProviderOptions(call)
	if err != nil {
		return nil, err
	}
	if len(call.Inputs) > maxEmbeddingInputs {
		return nil, &fantasy.Error{
			Title:   "invalid argument",
			Message: fmt.Sprintf("openai embeddings accept at most %d inputs per request, got %d", maxEmbeddingInputs, len(call.Inputs)),
		}
	}
	texts, err := embeddingTexts(call.Inputs)
	if err != nil {
		return nil, err
	}

	response, err := m.client.Embeddings.New(ctx, m.params(call, texts, options), embeddingCallRequestOptions(call)...)
	if err != nil {
		return nil, toProviderErr(err)
	}

	return m.toEmbeddingResponse(response, len(texts))
}

func embeddingProviderOptions(call fantasy.EmbeddingCall) (*EmbeddingProviderOptions, error) {
	v := call.ProviderOptions[Name]
	if v == nil {
		return &EmbeddingProviderOptions{}, nil
	}
	options, ok := v.(*EmbeddingProviderOptions)
	if !ok {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "openai provider options should be *openai.EmbeddingProviderOptions"}
	}
	return cmp.Or(options, &EmbeddingProviderOptions{}), nil
}

// NOTE: the endpoint takes plain strings, so there is no way to carry a file or
// to combine several parts into one vector.
func embeddingTexts(inputs []fantasy.EmbeddingInput) ([]string, error) {
	texts := make([]string, len(inputs))
	for i, input := range inputs {
		if len(input.Parts) != 1 {
			return nil, invalidEmbeddingInput(i)
		}
		text, ok := fantasy.AsMessagePart[fantasy.TextPart](input.Parts[0])
		if !ok {
			return nil, invalidEmbeddingInput(i)
		}
		texts[i] = text.Text
	}
	return texts, nil
}

func invalidEmbeddingInput(index int) error {
	return &fantasy.Error{
		Title:   "invalid argument",
		Message: fmt.Sprintf("input %d: openai embeddings accept exactly one text part per input; files are not supported", index),
	}
}

func (m embeddingModel) params(call fantasy.EmbeddingCall, texts []string, options *EmbeddingProviderOptions) openai.EmbeddingNewParams {
	params := openai.EmbeddingNewParams{
		Model: m.modelID,
		// NOTE: a single input is still sent as an array so the response has
		// one shape to handle.
		Input:          openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: texts},
		EncodingFormat: openai.EmbeddingNewParamsEncodingFormatFloat,
	}
	if call.Dimensions != nil {
		params.Dimensions = param.NewOpt(*call.Dimensions)
	}
	if options.User != nil {
		params.User = param.NewOpt(*options.User)
	}
	return params
}

func (m embeddingModel) toEmbeddingResponse(response *openai.CreateEmbeddingResponse, inputs int) (*fantasy.EmbeddingResponse, error) {
	if response == nil {
		return nil, invalidEmbeddingResponse("provider returned nil response")
	}
	if len(response.Data) != inputs {
		return nil, invalidEmbeddingResponse("expected %d embeddings, got %d", inputs, len(response.Data))
	}

	embeddings := make([][]float32, inputs)
	for pos, item := range response.Data {
		field := item.JSON.Index
		if field.Valid() != response.Data[0].JSON.Index.Valid() {
			return nil, invalidEmbeddingResponse("only some embeddings have an index")
		}
		index := item.Index
		if !field.Valid() {
			if raw := field.Raw(); raw != "" && raw != "null" {
				return nil, invalidEmbeddingResponse("embedding %d has an invalid index", pos)
			}
			// NOTE: some compatible backends omit every index and return vectors in input order.
			index = int64(pos)
		}
		if index < 0 || index >= int64(inputs) {
			return nil, invalidEmbeddingResponse("embedding index %d is out of range for %d inputs", index, inputs)
		}
		if embeddings[index] != nil {
			return nil, invalidEmbeddingResponse("embedding index %d appears more than once", index)
		}
		if len(item.Embedding) == 0 {
			return nil, invalidEmbeddingResponse("embedding %d is empty", index)
		}
		vector := make([]float32, len(item.Embedding))
		for i, value := range item.Embedding {
			vector[i] = float32(value)
			if v := float64(vector[i]); math.IsInf(v, 0) || math.IsNaN(v) {
				return nil, invalidEmbeddingResponse("embedding %d value %d is not finite as float32", index, i)
			}
		}
		embeddings[index] = vector
	}

	return &fantasy.EmbeddingResponse{
		Embeddings: embeddings,
		Model:      cmp.Or(response.Model, m.modelID),
		Usage: fantasy.Usage{
			InputTokens: response.Usage.PromptTokens,
			TotalTokens: response.Usage.TotalTokens,
		},
	}, nil
}

func invalidEmbeddingResponse(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid response", Message: fmt.Sprintf(format, args...)}
}
