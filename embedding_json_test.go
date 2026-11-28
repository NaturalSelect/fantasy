package fantasy

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

type roundTripProviderData string

func (roundTripProviderData) Options() {}

func (v roundTripProviderData) MarshalJSON() ([]byte, error) {
	return MarshalProviderType("test.round_trip", string(v))
}

func (v *roundTripProviderData) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, (*string)(v))
}

func init() {
	RegisterProviderType("test.round_trip", func(data []byte) (ProviderOptionsData, error) {
		var v roundTripProviderData
		err := json.Unmarshal(data, &v)
		return &v, err
	})
}

func roundTripJSON[T any](t *testing.T, v T) T {
	t.Helper()
	data, err := json.Marshal(v)
	require.NoError(t, err)
	var got T
	require.NoError(t, json.Unmarshal(data, &got))
	return got
}

func TestEmbeddingCall_JSON(t *testing.T) {
	t.Parallel()

	opts := ProviderOptions{"test": new(roundTripProviderData("option"))}
	want := EmbeddingCall{
		Inputs: []EmbeddingInput{{Parts: []MessagePart{
			TextPart{Text: "cat", ProviderOptions: opts},
			FilePart{Filename: "cat.png", MediaType: "image/png", Data: []byte{1, 2}},
		}}, TextEmbeddingInput("dog")},
		Dimensions:      new(int64(12)),
		ProviderOptions: opts,
	}
	require.Equal(t, want, roundTripJSON(t, want))
}

func TestEmbeddingResponse_JSONMetadata(t *testing.T) {
	t.Parallel()

	for _, metadata := range []ProviderMetadata{nil, {}, {"test": new(roundTripProviderData("metadata"))}} {
		want := EmbeddingResponse{
			Embeddings:       [][]float32{{0.2, -0.4}},
			Model:            "test",
			Usage:            Usage{InputTokens: 2},
			Warnings:         []CallWarning{{Message: "warning"}},
			ProviderMetadata: metadata,
		}
		require.Equal(t, want, roundTripJSON(t, want))
	}
}
