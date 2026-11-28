package fantasy

import (
	"context"
	"errors"
	"fmt"
)

// EmbeddingInput is the content that is turned into one vector.
type EmbeddingInput struct {
	// Parts accepts TextPart and FilePart values. Providers whose models only
	// embed text reject files. Models that embed several parts together return
	// one vector for the whole input.
	Parts []MessagePart `json:"parts"`
}

// TextEmbeddingInput returns an input made of a single text part.
func TextEmbeddingInput(text string) EmbeddingInput {
	return EmbeddingInput{Parts: []MessagePart{TextPart{Text: text}}}
}

// TextEmbeddingInputs returns one text input per string.
func TextEmbeddingInputs(texts ...string) []EmbeddingInput {
	inputs := make([]EmbeddingInput, len(texts))
	for i, text := range texts {
		inputs[i] = TextEmbeddingInput(text)
	}
	return inputs
}

// EmbeddingCall represents a call to an embedding model.
type EmbeddingCall struct {
	// Inputs are embedded in one request. The response has one vector per input,
	// in the same order.
	Inputs []EmbeddingInput `json:"inputs"`
	// Dimensions shortens the vectors, where the model supports it. Shortened
	// vectors are not normalized by every model.
	Dimensions *int64 `json:"dimensions"`

	// UserAgent overrides the provider-level User-Agent header for this call.
	UserAgent string `json:"-"`

	// Headers overrides matching provider-level headers for this call.
	Headers map[string]string `json:"-"`

	// for provider specific options, the key is the provider id
	ProviderOptions ProviderOptions `json:"provider_options"`
}

// Validate checks the provider-independent structure of the call. Limits that
// differ between providers and models are left to the provider and the service.
func (c EmbeddingCall) Validate() error {
	if len(c.Inputs) == 0 {
		return invalidArgument("at least one input is required")
	}
	for i, input := range c.Inputs {
		if len(input.Parts) == 0 {
			return invalidArgument("input %d must have at least one part", i)
		}
		for j, part := range input.Parts {
			if err := validateEmbeddingPart(part); err != nil {
				return invalidArgument("input %d part %d: %v", i, j, err)
			}
		}
	}
	if c.Dimensions != nil && *c.Dimensions <= 0 {
		return invalidArgument("dimensions must be greater than zero, got %d", *c.Dimensions)
	}
	return nil
}

func validateEmbeddingPart(part MessagePart) error {
	if nilOrNilPointer(part) {
		return errors.New("part is nil")
	}
	if text, ok := AsMessagePart[TextPart](part); ok {
		if text.Text == "" {
			return errors.New("text is empty")
		}
		return nil
	}
	if file, ok := AsMessagePart[FilePart](part); ok {
		if len(file.Data) == 0 {
			return errors.New("file has no data")
		}
		if file.MediaType == "" {
			return errors.New("file has no media type")
		}
		return nil
	}
	return fmt.Errorf("unsupported part type %q, use text or files", part.GetType())
}

// EmbeddingResponse represents a response from an embedding model.
type EmbeddingResponse struct {
	// Embeddings[i] is the vector of Inputs[i].
	Embeddings [][]float32 `json:"embeddings"`
	// Model is the model that produced the vectors, or the requested model when
	// the provider does not report one.
	Model    string        `json:"model"`
	Warnings []CallWarning `json:"warnings"`
	Usage    Usage         `json:"usage"`

	// for provider specific response metadata, the key is the provider id
	ProviderMetadata ProviderMetadata `json:"provider_metadata"`
}

// EmbeddingModel represents a model that can turn content into vectors.
type EmbeddingModel interface {
	Embed(context.Context, EmbeddingCall) (*EmbeddingResponse, error)

	Provider() string
	Model() string
}

// EmbeddingModelProvider is implemented by providers that offer embedding models.
// Check for it with a type assertion on a Provider.
type EmbeddingModelProvider interface {
	EmbeddingModel(ctx context.Context, modelID string) (EmbeddingModel, error)
}
