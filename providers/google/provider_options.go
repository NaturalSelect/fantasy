// Package google provides an implementation of the fantasy AI SDK for Google's language models.
package google

import (
	"encoding/json"

	"charm.land/fantasy"
)

// Global type identifiers for Google-specific provider data.
const (
	TypeProviderOptions          = Name + ".options"
	TypeProviderImageOptions     = Name + ".image_options"
	TypeProviderEmbeddingOptions = Name + ".embedding_options"
	TypeReasoningMetadata        = Name + ".reasoning_metadata"
)

// Register Google provider-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeProviderOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeProviderImageOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ImageProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeProviderEmbeddingOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v EmbeddingProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeReasoningMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ReasoningMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// ThinkingLevel controls the amount of thinking a model does.
// Use this for Gemini 3+ models instead of ThinkingBudget.
// Mutually exclusive with ThinkingBudget.
type ThinkingLevel = string

// Predefined thinking levels for the Google provider.
const (
	ThinkingLevelLow     ThinkingLevel = "LOW"
	ThinkingLevelMedium  ThinkingLevel = "MEDIUM"
	ThinkingLevelHigh    ThinkingLevel = "HIGH"
	ThinkingLevelMinimal ThinkingLevel = "MINIMAL"
)

// ThinkingConfig represents thinking configuration for the Google provider.
type ThinkingConfig struct {
	ThinkingBudget  *int64  `json:"thinking_budget,omitempty"`
	IncludeThoughts *bool   `json:"include_thoughts,omitempty"`
	ThinkingLevel   *string `json:"thinking_level,omitempty"`
}

// ReasoningMetadata represents reasoning metadata for the Google provider.
type ReasoningMetadata struct {
	Signature string `json:"signature"`
	ToolID    string `json:"tool_id"`
}

// Options implements the ProviderOptionsData interface for ReasoningMetadata.
func (m *ReasoningMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ReasoningMetadata.
func (m ReasoningMetadata) MarshalJSON() ([]byte, error) {
	type plain ReasoningMetadata
	return fantasy.MarshalProviderType(TypeReasoningMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ReasoningMetadata.
func (m *ReasoningMetadata) UnmarshalJSON(data []byte) error {
	type plain ReasoningMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ReasoningMetadata(p)
	return nil
}

// SafetySetting represents safety settings for the Google provider.
type SafetySetting struct {
	// 'HARM_CATEGORY_UNSPECIFIED',
	// 'HARM_CATEGORY_HATE_SPEECH',
	// 'HARM_CATEGORY_DANGEROUS_CONTENT',
	// 'HARM_CATEGORY_HARASSMENT',
	// 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
	// 'HARM_CATEGORY_CIVIC_INTEGRITY',
	Category string `json:"category"`

	// 'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
	// 'BLOCK_LOW_AND_ABOVE',
	// 'BLOCK_MEDIUM_AND_ABOVE',
	// 'BLOCK_ONLY_HIGH',
	// 'BLOCK_NONE',
	// 'OFF',
	Threshold string `json:"threshold"`
}

// ProviderOptions represents additional options for the Google provider.
type ProviderOptions struct {
	ThinkingConfig *ThinkingConfig `json:"thinking_config"`

	// Optional.
	// The name of the cached content used as context to serve the prediction.
	// Format: cachedContents/{cachedContent}
	CachedContent string `json:"cached_content"`

	// Optional. A list of unique safety settings for blocking unsafe content.
	SafetySettings []SafetySetting `json:"safety_settings"`
	// 'HARM_BLOCK_THRESHOLD_UNSPECIFIED',
	// 'BLOCK_LOW_AND_ABOVE',
	// 'BLOCK_MEDIUM_AND_ABOVE',
	// 'BLOCK_ONLY_HIGH',
	// 'BLOCK_NONE',
	// 'OFF',
	Threshold string `json:"threshold"`
}

// Options implements the ProviderOptionsData interface for ProviderOptions.
func (o *ProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderOptions.
func (o ProviderOptions) MarshalJSON() ([]byte, error) {
	type plain ProviderOptions
	return fantasy.MarshalProviderType(TypeProviderOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderOptions.
func (o *ProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ProviderOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ProviderOptions(p)
	return nil
}

// ImageProviderOptions represents additional options for Google image generation.
// Fields marked Imagen only are ignored, with a warning, by Gemini image models.
type ImageProviderOptions struct {
	// ImageSize is the output resolution, e.g. "1K" or "2K".
	ImageSize *string `json:"image_size"`
	// NOTE: Person generation is configurable only on Vertex AI.
	PersonGeneration *string `json:"person_generation"`
	// NOTE: Only Vertex AI accepts an output MIME type.
	OutputMIMEType *string `json:"output_mime_type"`

	// Imagen only.
	NegativePrompt    *string  `json:"negative_prompt"`
	GuidanceScale     *float64 `json:"guidance_scale"`
	SafetyFilterLevel *string  `json:"safety_filter_level"`
}

// Options implements the ProviderOptionsData interface for ImageProviderOptions.
func (o *ImageProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ImageProviderOptions.
func (o ImageProviderOptions) MarshalJSON() ([]byte, error) {
	type plain ImageProviderOptions
	return fantasy.MarshalProviderType(TypeProviderImageOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ImageProviderOptions.
func (o *ImageProviderOptions) UnmarshalJSON(data []byte) error {
	type plain ImageProviderOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = ImageProviderOptions(p)
	return nil
}

// NOTE: A nil pointer must not become a non-nil options interface in the map.
func NewImageProviderOptions(opts *ImageProviderOptions) fantasy.ProviderOptions {
	if opts == nil {
		return fantasy.ProviderOptions{}
	}
	return fantasy.ProviderOptions{Name: opts}
}

// Task types for EmbeddingProviderOptions.TaskType. Only gemini-embedding-001
// takes them.
const (
	TaskTypeRetrievalQuery     = "RETRIEVAL_QUERY"
	TaskTypeRetrievalDocument  = "RETRIEVAL_DOCUMENT"
	TaskTypeSemanticSimilarity = "SEMANTIC_SIMILARITY"
	TaskTypeClassification     = "CLASSIFICATION"
	TaskTypeClustering         = "CLUSTERING"
	TaskTypeQuestionAnswering  = "QUESTION_ANSWERING"
	TaskTypeFactVerification   = "FACT_VERIFICATION"
	TaskTypeCodeRetrievalQuery = "CODE_RETRIEVAL_QUERY"
)

// EmbeddingProviderOptions represents additional options for Google embeddings.
// NOTE: The Vertex :predict route ignores DocumentOCR and AudioTrackExtraction;
// the response includes a warning for each of these options when set.
type EmbeddingProviderOptions struct {
	// TaskType is what the vectors are used for, see the TaskType constants.
	// It only applies to gemini-embedding-001. gemini-embedding-2 rejects it,
	// so put the task in the text instead, e.g. as a "task: search result | query: " prefix.
	TaskType *string `json:"task_type"`
	// Title describes the text. It only has an effect with the
	// RETRIEVAL_DOCUMENT task type.
	Title *string `json:"title"`

	// DocumentOCR reads the text of document inputs. Vertex only: the Gemini API
	// backend rejects it before the request is sent.
	DocumentOCR *bool `json:"document_ocr"`
	// AudioTrackExtraction embeds the audio track of video inputs. Vertex only:
	// the Gemini API backend rejects it before the request is sent.
	AudioTrackExtraction *bool `json:"audio_track_extraction"`
}

// Options implements the ProviderOptionsData interface for EmbeddingProviderOptions.
func (o *EmbeddingProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for EmbeddingProviderOptions.
func (o EmbeddingProviderOptions) MarshalJSON() ([]byte, error) {
	type plain EmbeddingProviderOptions
	return fantasy.MarshalProviderType(TypeProviderEmbeddingOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for EmbeddingProviderOptions.
func (o *EmbeddingProviderOptions) UnmarshalJSON(data []byte) error {
	type plain EmbeddingProviderOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = EmbeddingProviderOptions(p)
	return nil
}

// NewEmbeddingProviderOptions creates new embedding options for Google.
func NewEmbeddingProviderOptions(opts *EmbeddingProviderOptions) fantasy.ProviderOptions {
	if opts == nil {
		return fantasy.ProviderOptions{}
	}
	return fantasy.ProviderOptions{Name: opts}
}

// ParseOptions parses provider options from a map for the Google provider.
func ParseOptions(data map[string]any) (*ProviderOptions, error) {
	var options ProviderOptions
	if err := fantasy.ParseOptions(data, &options); err != nil {
		return nil, err
	}
	return &options, nil
}
