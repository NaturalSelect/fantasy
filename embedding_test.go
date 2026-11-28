package fantasy

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func validEmbeddingCall() EmbeddingCall {
	return EmbeddingCall{Inputs: TextEmbeddingInputs("hello", "world")}
}

func TestTextEmbeddingInputs(t *testing.T) {
	t.Parallel()

	t.Run("single", func(t *testing.T) {
		t.Parallel()
		input := TextEmbeddingInput("hello")
		require.Equal(t, []MessagePart{TextPart{Text: "hello"}}, input.Parts)
	})

	t.Run("one input per text in order", func(t *testing.T) {
		t.Parallel()
		inputs := TextEmbeddingInputs("a", "b", "c")
		require.Len(t, inputs, 3)
		for i, want := range []string{"a", "b", "c"} {
			require.Equal(t, []MessagePart{TextPart{Text: want}}, inputs[i].Parts)
		}
	})

	t.Run("no texts", func(t *testing.T) {
		t.Parallel()
		require.Empty(t, TextEmbeddingInputs())
	})
}

func TestEmbeddingCall_Validate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		mutate  func(*EmbeddingCall)
		wantErr string
	}{
		{name: "valid text inputs", mutate: func(*EmbeddingCall) {}},
		{
			name: "valid mixed parts",
			mutate: func(c *EmbeddingCall) {
				c.Inputs = []EmbeddingInput{{Parts: []MessagePart{
					TextPart{Text: "a cat"},
					FilePart{Data: []byte("png"), MediaType: "image/png"},
					&FilePart{Data: []byte("pdf"), MediaType: "application/pdf"},
					&TextPart{Text: "pointer"},
				}}}
			},
		},
		{
			name: "valid dimensions",
			mutate: func(c *EmbeddingCall) {
				c.Dimensions = new(int64(256))
			},
		},
		{
			name:    "no inputs",
			mutate:  func(c *EmbeddingCall) { c.Inputs = nil },
			wantErr: "at least one input is required",
		},
		{
			name:    "input without parts",
			mutate:  func(c *EmbeddingCall) { c.Inputs = append(c.Inputs, EmbeddingInput{}) },
			wantErr: "input 2 must have at least one part",
		},
		{
			name: "empty text",
			mutate: func(c *EmbeddingCall) {
				c.Inputs[1] = TextEmbeddingInput("")
			},
			wantErr: "input 1 part 0: text is empty",
		},
		{
			name: "file without data",
			mutate: func(c *EmbeddingCall) {
				c.Inputs[0].Parts = append(c.Inputs[0].Parts, FilePart{MediaType: "image/png"})
			},
			wantErr: "input 0 part 1: file has no data",
		},
		{
			name: "file without media type",
			mutate: func(c *EmbeddingCall) {
				c.Inputs[0].Parts = []MessagePart{FilePart{Data: []byte("x")}}
			},
			wantErr: "input 0 part 0: file has no media type",
		},
		{
			name: "nil part",
			mutate: func(c *EmbeddingCall) {
				c.Inputs[0].Parts = []MessagePart{nil}
			},
			wantErr: "input 0 part 0: part is nil",
		},
		{
			name: "unsupported part type",
			mutate: func(c *EmbeddingCall) {
				c.Inputs[0].Parts = []MessagePart{ToolCallPart{ToolCallID: "1"}}
			},
			wantErr: `unsupported part type "tool-call"`,
		},
		{
			name:    "zero dimensions",
			mutate:  func(c *EmbeddingCall) { c.Dimensions = new(int64(0)) },
			wantErr: "dimensions must be greater than zero, got 0",
		},
		{
			name:    "negative dimensions",
			mutate:  func(c *EmbeddingCall) { c.Dimensions = new(int64(-3)) },
			wantErr: "dimensions must be greater than zero, got -3",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			call := validEmbeddingCall()
			tt.mutate(&call)
			err := call.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tt.wantErr)
			var fantasyErr *Error
			require.ErrorAs(t, err, &fantasyErr)
			require.Equal(t, "invalid argument", fantasyErr.Title)
		})
	}
}

func TestEmbeddingCall_ValidateTypedNilPart(t *testing.T) {
	t.Parallel()

	for _, part := range []MessagePart{(*TextPart)(nil), (*FilePart)(nil)} {
		call := validEmbeddingCall()
		call.Inputs[1].Parts = append(call.Inputs[1].Parts, part)
		var err error
		require.NotPanics(t, func() { err = call.Validate() })
		require.EqualError(t, err, "invalid argument: input 1 part 1: part is nil")
	}
}

func TestEmbeddingResponse_JSONRoundTrip(t *testing.T) {
	t.Parallel()

	want := EmbeddingResponse{
		Embeddings: [][]float32{{0.25, -0.5, 1}, {0, 0.125, 2}},
		Model:      "embed-1",
		Usage:      Usage{InputTokens: 7, TotalTokens: 7},
	}
	data, err := json.Marshal(want)
	require.NoError(t, err)

	var got EmbeddingResponse
	require.NoError(t, json.Unmarshal(data, &got))
	require.Equal(t, want.Embeddings, got.Embeddings)
	require.Equal(t, want.Model, got.Model)
	require.Equal(t, want.Usage, got.Usage)
}
