package fantasy

import (
	"encoding/json"
	"fmt"
)

type rawMessageParts []json.RawMessage

func (raw rawMessageParts) decode() ([]MessagePart, error) {
	if raw == nil {
		return nil, nil
	}
	parts := make([]MessagePart, len(raw))
	for i, data := range raw {
		part, err := UnmarshalMessagePart(data)
		if err != nil {
			return nil, fmt.Errorf("failed to unmarshal message part at index %d: %w", i, err)
		}
		parts[i] = part
	}
	return parts, nil
}

type rawProviderData map[string]json.RawMessage

func (raw rawProviderData) options() (ProviderOptions, error) {
	if raw == nil {
		return nil, nil
	}
	return UnmarshalProviderOptions(raw)
}

func (raw rawProviderData) metadata() (ProviderMetadata, error) {
	if raw == nil {
		return nil, nil
	}
	return UnmarshalProviderMetadata(raw)
}

type rawWarnings []struct {
	CallWarning
	Tool *json.RawMessage `json:"tool"`
}

func (raw rawWarnings) decode() ([]CallWarning, error) {
	if raw == nil {
		return nil, nil
	}
	warnings := make([]CallWarning, len(raw))
	for i, w := range raw {
		if w.Tool != nil {
			tool, err := UnmarshalTool(*w.Tool)
			if err != nil {
				return nil, fmt.Errorf("failed to unmarshal tool in warning %d: %w", i, err)
			}
			w.CallWarning.Tool = tool
		}
		warnings[i] = w.CallWarning
	}
	return warnings, nil
}

func (in *EmbeddingInput) UnmarshalJSON(data []byte) error {
	var wire struct {
		Parts rawMessageParts `json:"parts"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	parts, err := wire.Parts.decode()
	if err != nil {
		return err
	}
	in.Parts = parts
	return nil
}

func (c *EmbeddingCall) UnmarshalJSON(data []byte) error {
	var wire struct {
		Inputs          []EmbeddingInput `json:"inputs"`
		Dimensions      *int64           `json:"dimensions"`
		ProviderOptions rawProviderData  `json:"provider_options"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	options, err := wire.ProviderOptions.options()
	if err != nil {
		return err
	}
	c.Inputs = wire.Inputs
	c.Dimensions = wire.Dimensions
	c.ProviderOptions = options
	return nil
}

func (r *EmbeddingResponse) UnmarshalJSON(data []byte) error {
	var wire struct {
		Embeddings       [][]float32     `json:"embeddings"`
		Model            string          `json:"model"`
		Usage            Usage           `json:"usage"`
		Warnings         rawWarnings     `json:"warnings"`
		ProviderMetadata rawProviderData `json:"provider_metadata"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	metadata, err := wire.ProviderMetadata.metadata()
	if err != nil {
		return err
	}
	warnings, err := wire.Warnings.decode()
	if err != nil {
		return err
	}
	*r = EmbeddingResponse{
		Embeddings:       wire.Embeddings,
		Model:            wire.Model,
		Usage:            wire.Usage,
		Warnings:         warnings,
		ProviderMetadata: metadata,
	}
	return nil
}
