package openai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

type embeddingRequest struct {
	path   string
	header http.Header
	body   []byte
}

func (r embeddingRequest) jsonBody(t *testing.T) map[string]any {
	t.Helper()
	var body map[string]any
	require.NoError(t, json.Unmarshal(r.body, &body))
	return body
}

type embeddingServer struct {
	mu       sync.Mutex
	requests []embeddingRequest
}

func (s *embeddingServer) record(r embeddingRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, r)
}

func (s *embeddingServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

func (s *embeddingServer) last(t *testing.T) embeddingRequest {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	require.NotEmpty(t, s.requests, "the server received no request")
	return s.requests[len(s.requests)-1]
}

func newEmbeddingTestProvider(t *testing.T, status int, response any, opts ...Option) (fantasy.EmbeddingModelProvider, *embeddingServer) {
	t.Helper()

	recorder := &embeddingServer{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		recorder.record(embeddingRequest{path: r.URL.Path, header: r.Header.Clone(), body: body})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)

	opts = append([]Option{WithAPIKey("test-api-key"), WithBaseURL(server.URL)}, opts...)
	p, err := New(opts...)
	require.NoError(t, err)
	embeddingProvider, ok := p.(fantasy.EmbeddingModelProvider)
	require.True(t, ok)
	return embeddingProvider, recorder
}

func newEmbeddingTestModel(t *testing.T, response any, opts ...Option) (fantasy.EmbeddingModel, *embeddingServer) {
	t.Helper()

	embeddingProvider, server := newEmbeddingTestProvider(t, http.StatusOK, response, opts...)
	model, err := embeddingProvider.EmbeddingModel(context.Background(), "text-embedding-3-small")
	require.NoError(t, err)
	return model, server
}

type embeddingItem struct {
	index  int
	vector []float64
}

func embeddingResponse(model string, items ...embeddingItem) map[string]any {
	data := make([]map[string]any, 0, len(items))
	for _, item := range items {
		data = append(data, map[string]any{
			"object":    "embedding",
			"index":     item.index,
			"embedding": item.vector,
		})
	}
	response := map[string]any{
		"object": "list",
		"data":   data,
		"usage":  map[string]any{"prompt_tokens": 7, "total_tokens": 9},
	}
	if model != "" {
		response["model"] = model
	}
	return response
}

func TestEmbeddingModel_Embed(t *testing.T) {
	t.Parallel()

	t.Run("should send the request and map the response", func(t *testing.T) {
		t.Parallel()

		model, server := newEmbeddingTestModel(t, embeddingResponse(
			"text-embedding-3-small-2024",
			embeddingItem{0, []float64{0.25, -0.5}},
			embeddingItem{1, []float64{1.5, 2}},
		))
		require.Equal(t, "text-embedding-3-small", model.Model())
		require.Equal(t, "openai", model.Provider())

		response, err := model.Embed(context.Background(), fantasy.EmbeddingCall{
			Inputs:          fantasy.TextEmbeddingInputs("hello", "world"),
			Dimensions:      new(int64(2)),
			ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{User: new("user-1")}),
		})
		require.NoError(t, err)

		request := server.last(t)
		require.Equal(t, "/embeddings", request.path)
		body := request.jsonBody(t)
		require.Equal(t, "text-embedding-3-small", body["model"])
		require.Equal(t, []any{"hello", "world"}, body["input"])
		require.Equal(t, "float", body["encoding_format"])
		require.EqualValues(t, 2, body["dimensions"])
		require.Equal(t, "user-1", body["user"])

		require.Equal(t, [][]float32{{0.25, -0.5}, {1.5, 2}}, response.Embeddings)
		require.Equal(t, "text-embedding-3-small-2024", response.Model)
		require.Equal(t, fantasy.Usage{InputTokens: 7, TotalTokens: 9}, response.Usage)
		require.Empty(t, response.Warnings)
		require.Nil(t, response.ProviderMetadata)
	})

	t.Run("should omit dimensions and user when they are not set", func(t *testing.T) {
		t.Parallel()

		model, server := newEmbeddingTestModel(t, embeddingResponse("", embeddingItem{0, []float64{1}}))

		_, err := model.Embed(context.Background(), fantasy.EmbeddingCall{
			Inputs:          fantasy.TextEmbeddingInputs("hello"),
			ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{}),
		})
		require.NoError(t, err)

		body := server.last(t).jsonBody(t)
		require.NotContains(t, body, "dimensions")
		require.NotContains(t, body, "user")
		require.Equal(t, "float", body["encoding_format"])
	})

	t.Run("should send a single input as an array", func(t *testing.T) {
		t.Parallel()

		model, server := newEmbeddingTestModel(t, embeddingResponse("", embeddingItem{0, []float64{1, 2, 3}}))

		response, err := model.Embed(context.Background(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("hello")})
		require.NoError(t, err)

		require.Equal(t, []any{"hello"}, server.last(t).jsonBody(t)["input"])
		require.Equal(t, [][]float32{{1, 2, 3}}, response.Embeddings)
	})

	t.Run("should place embeddings by index when the response is out of order", func(t *testing.T) {
		t.Parallel()

		model, _ := newEmbeddingTestModel(t, embeddingResponse(
			"",
			embeddingItem{2, []float64{3, 3}},
			embeddingItem{0, []float64{1, 1}},
			embeddingItem{1, []float64{2, 2}},
		))

		response, err := model.Embed(context.Background(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a", "b", "c")})
		require.NoError(t, err)

		require.Equal(t, [][]float32{{1, 1}, {2, 2}, {3, 3}}, response.Embeddings)
	})

	t.Run("should fall back to the requested model when the response has none", func(t *testing.T) {
		t.Parallel()

		model, _ := newEmbeddingTestModel(t, embeddingResponse("", embeddingItem{0, []float64{1}}))

		response, err := model.Embed(context.Background(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("hello")})
		require.NoError(t, err)

		require.Equal(t, "text-embedding-3-small", response.Model)
	})

	t.Run("should accept text parts given as pointers", func(t *testing.T) {
		t.Parallel()

		model, server := newEmbeddingTestModel(t, embeddingResponse("", embeddingItem{0, []float64{1}}, embeddingItem{1, []float64{2}}))

		_, err := model.Embed(context.Background(), fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
			{Parts: []fantasy.MessagePart{&fantasy.TextPart{Text: "pointer"}}},
			{Parts: []fantasy.MessagePart{fantasy.TextPart{Text: "value"}}},
		}})
		require.NoError(t, err)

		require.Equal(t, []any{"pointer", "value"}, server.last(t).jsonBody(t)["input"])
	})

	t.Run("should accept the largest batch", func(t *testing.T) {
		t.Parallel()

		items := make([]embeddingItem, maxEmbeddingInputs)
		texts := make([]string, maxEmbeddingInputs)
		for i := range items {
			items[i] = embeddingItem{i, []float64{float64(i)}}
			texts[i] = "text"
		}
		model, server := newEmbeddingTestModel(t, embeddingResponse("", items...))

		response, err := model.Embed(context.Background(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs(texts...)})
		require.NoError(t, err)

		require.Len(t, response.Embeddings, maxEmbeddingInputs)
		require.Equal(t, []float32{2047}, response.Embeddings[2047])
		require.Equal(t, 1, server.count())
	})

	t.Run("should use the provider name as the model provider", func(t *testing.T) {
		t.Parallel()

		embeddingProvider, _ := newEmbeddingTestProvider(t, http.StatusOK, nil, WithName("custom"))
		model, err := embeddingProvider.EmbeddingModel(context.Background(), "my-deployment")
		require.NoError(t, err)

		require.Equal(t, "custom", model.Provider())
		require.Equal(t, "my-deployment", model.Model())
	})
}

func TestEmbeddingModel_Embed_InvalidCalls(t *testing.T) {
	t.Parallel()

	tooMany := make([]string, maxEmbeddingInputs+1)
	for i := range tooMany {
		tooMany[i] = "text"
	}
	file := fantasy.FilePart{Data: []byte("png"), MediaType: "image/png"}

	tests := []struct {
		name    string
		call    fantasy.EmbeddingCall
		message string
	}{
		{
			name:    "file part",
			call:    fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{{Parts: []fantasy.MessagePart{file}}}},
			message: "input 0: openai embeddings accept exactly one text part per input; files are not supported",
		},
		{
			name:    "pointer file part",
			call:    fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{{Parts: []fantasy.MessagePart{&file}}}},
			message: "input 0: openai embeddings accept exactly one text part per input; files are not supported",
		},
		{
			name: "several parts in one input",
			call: fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
				fantasy.TextEmbeddingInput("fine"),
				{Parts: []fantasy.MessagePart{fantasy.TextPart{Text: "a"}, fantasy.TextPart{Text: "b"}}},
			}},
			message: "input 1: openai embeddings accept exactly one text part per input; files are not supported",
		},
		{
			name: "text and file in one input",
			call: fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
				{Parts: []fantasy.MessagePart{fantasy.TextPart{Text: "a cat"}, file}},
			}},
			message: "input 0: openai embeddings accept exactly one text part per input; files are not supported",
		},
		{
			name:    "more than 2048 inputs",
			call:    fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs(tooMany...)},
			message: "at most 2048 inputs",
		},
		{
			name:    "no inputs",
			call:    fantasy.EmbeddingCall{},
			message: "at least one input",
		},
		{
			name: "dimensions of zero",
			call: fantasy.EmbeddingCall{
				Inputs:     fantasy.TextEmbeddingInputs("hello"),
				Dimensions: new(int64(0)),
			},
			message: "dimensions",
		},
		{
			name: "provider options of another type",
			call: fantasy.EmbeddingCall{
				Inputs:          fantasy.TextEmbeddingInputs("hello"),
				ProviderOptions: NewProviderOptions(&ProviderOptions{}),
			},
			message: "*openai.EmbeddingProviderOptions",
		},
		{
			name: "image provider options",
			call: fantasy.EmbeddingCall{
				Inputs:          fantasy.TextEmbeddingInputs("hello"),
				ProviderOptions: NewImageProviderOptions(&ImageProviderOptions{}),
			},
			message: "*openai.EmbeddingProviderOptions",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, server := newEmbeddingTestModel(t, embeddingResponse("", embeddingItem{0, []float64{1}}))

			response, err := model.Embed(context.Background(), tt.call)
			require.Nil(t, response)
			fantasyErr, ok := errors.AsType[*fantasy.Error](err)
			require.True(t, ok, "got %T: %v", err, err)
			require.Equal(t, "invalid argument", fantasyErr.Title)
			require.Contains(t, fantasyErr.Message, tt.message)
			require.Zero(t, server.count(), "the request must be rejected before it is sent")
		})
	}
}

func TestEmbeddingModel_Embed_InvalidResponses(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		response map[string]any
		message  string
	}{
		{
			name:     "fewer embeddings than inputs",
			response: embeddingResponse("", embeddingItem{0, []float64{1}}),
			message:  "expected 2 embeddings, got 1",
		},
		{
			name: "more embeddings than inputs",
			response: embeddingResponse(
				"",
				embeddingItem{0, []float64{1}},
				embeddingItem{1, []float64{2}},
				embeddingItem{2, []float64{3}},
			),
			message: "expected 2 embeddings, got 3",
		},
		{
			name: "duplicate index",
			response: embeddingResponse(
				"",
				embeddingItem{1, []float64{1}},
				embeddingItem{1, []float64{2}},
			),
			message: "index 1 appears more than once",
		},
		{
			name: "index past the last input",
			response: embeddingResponse(
				"",
				embeddingItem{0, []float64{1}},
				embeddingItem{2, []float64{2}},
			),
			message: "index 2 is out of range",
		},
		{
			name: "negative index",
			response: embeddingResponse(
				"",
				embeddingItem{0, []float64{1}},
				embeddingItem{-1, []float64{2}},
			),
			message: "index -1 is out of range",
		},
		{
			name: "empty vector",
			response: embeddingResponse(
				"",
				embeddingItem{0, []float64{1}},
				embeddingItem{1, []float64{}},
			),
			message: "embedding 1 is empty",
		},
		{
			name: "missing vector",
			response: embeddingResponse(
				"",
				embeddingItem{0, []float64{1}},
				embeddingItem{1, nil},
			),
			message: "embedding 1 is empty",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, server := newEmbeddingTestModel(t, tt.response)

			response, err := model.Embed(context.Background(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a", "b")})
			require.Nil(t, response)
			fantasyErr, ok := errors.AsType[*fantasy.Error](err)
			require.True(t, ok, "got %T: %v", err, err)
			require.Equal(t, "invalid response", fantasyErr.Title)
			require.Contains(t, fantasyErr.Message, tt.message)
			require.Equal(t, 1, server.count())
		})
	}
}

func TestEmbeddingModel_Embed_ProviderError(t *testing.T) {
	t.Parallel()

	embeddingProvider, server := newEmbeddingTestProvider(t, http.StatusTooManyRequests, map[string]any{
		"error": map[string]any{
			"message": "Rate limit reached",
			"type":    "requests",
			"code":    "rate_limit_exceeded",
		},
	})
	model, err := embeddingProvider.EmbeddingModel(context.Background(), "text-embedding-3-small")
	require.NoError(t, err)

	response, err := model.Embed(context.Background(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("hello")})
	require.Nil(t, response)

	providerErr, ok := errors.AsType[*fantasy.ProviderError](err)
	require.True(t, ok, "got %T: %v", err, err)
	require.Equal(t, http.StatusTooManyRequests, providerErr.StatusCode)
	require.Contains(t, providerErr.Message, "Rate limit reached")
	require.True(t, providerErr.IsRetryable())
	// NOTE: retrying is the agent's job. The SDK client must not retry on its own.
	require.Equal(t, 1, server.count())
}

func TestEmbeddingModel_Embed_RequestOverrides(t *testing.T) {
	t.Parallel()

	providerOptions := []Option{
		WithUserAgent("provider-agent/1.0"),
		WithHeaders(map[string]string{
			"X-Provider": "provider",
			"X-Shared":   "provider",
		}),
	}
	response := embeddingResponse("", embeddingItem{0, []float64{1}})

	t.Run("should keep the provider settings by default", func(t *testing.T) {
		t.Parallel()

		model, server := newEmbeddingTestModel(t, response, providerOptions...)

		_, err := model.Embed(context.Background(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("hello")})
		require.NoError(t, err)

		header := server.last(t).header
		require.Equal(t, []string{"provider-agent/1.0"}, header.Values("User-Agent"))
		require.Equal(t, []string{"provider"}, header.Values("X-Provider"))
		require.Equal(t, []string{"provider"}, header.Values("X-Shared"))
		require.Equal(t, []string{"Bearer test-api-key"}, header.Values("Authorization"))
	})

	t.Run("should let the call override the provider settings", func(t *testing.T) {
		t.Parallel()

		model, server := newEmbeddingTestModel(t, response, providerOptions...)

		_, err := model.Embed(context.Background(), fantasy.EmbeddingCall{
			Inputs:    fantasy.TextEmbeddingInputs("hello"),
			UserAgent: "call-agent/2.0",
			Headers:   map[string]string{"X-Shared": "call", "X-Call": "call"},
		})
		require.NoError(t, err)

		header := server.last(t).header
		require.Equal(t, []string{"call-agent/2.0"}, header.Values("User-Agent"))
		require.Equal(t, []string{"call"}, header.Values("X-Shared"))
		require.Equal(t, []string{"call"}, header.Values("X-Call"))
		require.Equal(t, []string{"provider"}, header.Values("X-Provider"))
	})
}

func TestEmbeddingProviderOptions_JSON(t *testing.T) {
	t.Parallel()

	t.Run("round trip through the registry", func(t *testing.T) {
		t.Parallel()

		options := NewEmbeddingProviderOptions(&EmbeddingProviderOptions{User: new("user-1")})
		data, err := json.Marshal(options)
		require.NoError(t, err)
		require.JSONEq(t, `{"openai":{"type":"openai.embedding_options","data":{"user":"user-1"}}}`, string(data))

		var raw map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(data, &raw))
		decoded, err := fantasy.UnmarshalProviderOptions(raw)
		require.NoError(t, err)
		require.Equal(t, options, decoded)
	})

	t.Run("unset user round trip", func(t *testing.T) {
		t.Parallel()

		options := NewEmbeddingProviderOptions(&EmbeddingProviderOptions{})
		data, err := json.Marshal(options)
		require.NoError(t, err)
		require.JSONEq(t, `{"openai":{"type":"openai.embedding_options","data":{"user":null}}}`, string(data))

		var raw map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(data, &raw))
		decoded, err := fantasy.UnmarshalProviderOptions(raw)
		require.NoError(t, err)
		require.Equal(t, options, decoded)
	})

	t.Run("type identifier", func(t *testing.T) {
		t.Parallel()

		require.Equal(t, "openai.embedding_options", TypeProviderEmbeddingOptions)
	})
}
