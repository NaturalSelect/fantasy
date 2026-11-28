package google

import (
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

type embeddingRequest struct {
	method string
	path   string
	header http.Header
	body   map[string]any
}

type embeddingRecorder struct {
	mu       sync.Mutex
	recorded []embeddingRequest
}

func (r *embeddingRecorder) record(request embeddingRequest) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.recorded = append(r.recorded, request)
}

func (r *embeddingRecorder) requests() []embeddingRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.recorded)
}

func (r *embeddingRecorder) only(t *testing.T) embeddingRequest {
	t.Helper()
	requests := r.requests()
	require.Len(t, requests, 1)
	return requests[0]
}

func newEmbeddingTestServer(t *testing.T, status int, response any) (*httptest.Server, *embeddingRecorder) {
	t.Helper()

	recorder := &embeddingRecorder{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		recorder.record(embeddingRequest{method: r.Method, path: r.URL.Path, header: r.Header.Clone(), body: body})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)
	return server, recorder
}

func newTestEmbeddingModel(t *testing.T, modelID string, opts ...Option) fantasy.EmbeddingModel {
	t.Helper()

	p, err := New(opts...)
	require.NoError(t, err)
	embeddingProvider, ok := p.(fantasy.EmbeddingModelProvider)
	require.True(t, ok)
	model, err := embeddingProvider.EmbeddingModel(t.Context(), modelID)
	require.NoError(t, err)
	return model
}

func newGeminiEmbeddingModel(t *testing.T, status int, response any, modelID string, opts ...Option) (fantasy.EmbeddingModel, *embeddingRecorder) {
	t.Helper()

	server, recorder := newEmbeddingTestServer(t, status, response)
	opts = append([]Option{WithGeminiAPIKey("test-api-key"), WithBaseURL(server.URL)}, opts...)
	return newTestEmbeddingModel(t, modelID, opts...), recorder
}

func newVertexEmbeddingModel(t *testing.T, status int, response any, modelID string, opts ...Option) (fantasy.EmbeddingModel, *embeddingRecorder) {
	t.Helper()

	server, recorder := newEmbeddingTestServer(t, status, response)
	opts = append([]Option{WithVertex("test-project", "us-central1"), WithBaseURL(server.URL), WithSkipAuth(true)}, opts...)
	return newTestEmbeddingModel(t, modelID, opts...), recorder
}

// geminiEmbeddingResponse is a batchEmbedContents response with one vector per
// values slice. A nil usage leaves the usage metadata out.
func geminiEmbeddingResponse(usage map[string]any, vectors ...[]float32) map[string]any {
	embeddings := make([]map[string]any, 0, len(vectors))
	for _, values := range vectors {
		embeddings = append(embeddings, map[string]any{"values": values})
	}
	response := map[string]any{"embeddings": embeddings}
	if usage != nil {
		response["usageMetadata"] = usage
	}
	return response
}

// vertexPredictResponse is a :predict response. Every vector gets the token
// count at the same index of tokenCounts.
func vertexPredictResponse(tokenCounts []int, vectors ...[]float32) map[string]any {
	predictions := make([]map[string]any, 0, len(vectors))
	for i, values := range vectors {
		embeddings := map[string]any{"values": values}
		if i < len(tokenCounts) {
			embeddings["statistics"] = map[string]any{"truncated": false, "token_count": tokenCounts[i]}
		}
		predictions = append(predictions, map[string]any{"embeddings": embeddings})
	}
	return map[string]any{"predictions": predictions}
}

func embeddingFile(mediaType, data string) fantasy.FilePart {
	return fantasy.FilePart{Data: []byte(data), MediaType: mediaType}
}

func embeddingProviderErrorBody(status int, message, errorStatus string) map[string]any {
	return map[string]any{"error": map[string]any{"code": status, "message": message, "status": errorStatus}}
}

// geminiRequests returns the entries of requests in the batchEmbedContents body.
func geminiRequests(t *testing.T, request embeddingRequest) []map[string]any {
	t.Helper()

	raw, ok := request.body["requests"].([]any)
	require.True(t, ok, "body has no requests: %v", request.body)
	requests := make([]map[string]any, len(raw))
	for i, item := range raw {
		requests[i], ok = item.(map[string]any)
		require.True(t, ok)
	}
	return requests
}

func requestParts(t *testing.T, request map[string]any) []map[string]any {
	t.Helper()

	content, ok := request["content"].(map[string]any)
	require.True(t, ok, "request has no content: %v", request)
	raw, ok := content["parts"].([]any)
	require.True(t, ok)
	parts := make([]map[string]any, len(raw))
	for i, item := range raw {
		parts[i], ok = item.(map[string]any)
		require.True(t, ok)
	}
	return parts
}

func requireInvalidEmbeddingArgument(t *testing.T, err error) *fantasy.Error {
	t.Helper()

	var fantasyErr *fantasy.Error
	require.ErrorAs(t, err, &fantasyErr)
	require.Equal(t, "invalid argument", fantasyErr.Title)
	return fantasyErr
}

func requireInvalidEmbeddingResponse(t *testing.T, err error) {
	t.Helper()

	var fantasyErr *fantasy.Error
	require.ErrorAs(t, err, &fantasyErr)
	require.Equal(t, "invalid response", fantasyErr.Title)
}

func TestEmbeddingModel_Gemini(t *testing.T) {
	t.Parallel()

	t.Run("should report the model and the provider", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiEmbeddingModel(t, http.StatusOK, nil, "gemini-embedding-2")
		require.Equal(t, "gemini-embedding-2", model.Model())
		require.Equal(t, "google", model.Provider())

		model, _ = newGeminiEmbeddingModel(t, http.StatusOK, nil, "gemini-embedding-2", WithName("my-google"))
		require.Equal(t, "my-google", model.Provider())
	})

	t.Run("should send one request per input in order to the v1beta batch endpoint", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK,
			geminiEmbeddingResponse(nil, []float32{1, 2}, []float32{3, 4}, []float32{5, 6}),
			"gemini-embedding-2")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("first", "second", "third")})
		require.NoError(t, err)

		request := recorder.only(t)
		require.Equal(t, http.MethodPost, request.method)
		// NOTE: The pinned v1beta matters, the SDK default is v1.
		require.Equal(t, "/v1beta/models/gemini-embedding-2:batchEmbedContents", request.path)
		require.Equal(t, "test-api-key", request.header.Get("X-Goog-Api-Key"))

		requests := geminiRequests(t, request)
		require.Len(t, requests, 3)
		for i, text := range []string{"first", "second", "third"} {
			require.Equal(t, "models/gemini-embedding-2", requests[i]["model"])
			parts := requestParts(t, requests[i])
			require.Len(t, parts, 1)
			require.Equal(t, text, parts[0]["text"])
			// NOTE: Embeddings have no turns, so the role stays out of the request.
			require.NotContains(t, requests[i]["content"], "role")
		}
	})

	t.Run("should send files as inline data", func(t *testing.T) {
		t.Parallel()

		for _, mediaType := range []string{"image/png", "image/jpeg", "audio/mpeg", "video/mp4", "application/pdf", "IMAGE/PNG"} {
			t.Run(mediaType, func(t *testing.T) {
				t.Parallel()

				model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, geminiEmbeddingResponse(nil, []float32{1}), "gemini-embedding-2")

				_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
					{Parts: []fantasy.MessagePart{embeddingFile(mediaType, "file-bytes")}},
				}})
				require.NoError(t, err)

				parts := requestParts(t, geminiRequests(t, recorder.only(t))[0])
				require.Len(t, parts, 1)
				inlineData, ok := parts[0]["inlineData"].(map[string]any)
				require.True(t, ok, "part is not inline data: %v", parts[0])
				require.Equal(t, mediaType, inlineData["mimeType"])
				require.Equal(t, encodeImage("file-bytes"), inlineData["data"])
				require.NotContains(t, parts[0], "text")
			})
		}
	})

	t.Run("should accept parts given by pointer", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, geminiEmbeddingResponse(nil, []float32{1}), "gemini-embedding-2")

		file := embeddingFile("image/png", "pointer-image")
		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
			{Parts: []fantasy.MessagePart{&fantasy.TextPart{Text: "pointer text"}, &file}},
		}})
		require.NoError(t, err)

		parts := requestParts(t, geminiRequests(t, recorder.only(t))[0])
		require.Len(t, parts, 2)
		require.Equal(t, "pointer text", parts[0]["text"])
		require.Equal(t, encodeImage("pointer-image"), parts[1]["inlineData"].(map[string]any)["data"])
	})

	t.Run("should send the parts of one input as one request", func(t *testing.T) {
		t.Parallel()

		// NOTE: gemini-embedding-2 turns several parts of one content into a
		// single vector, so the parts of an input must stay in one request.
		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK,
			geminiEmbeddingResponse(nil, []float32{1, 2}, []float32{3, 4}),
			"gemini-embedding-2")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
			{Parts: []fantasy.MessagePart{
				fantasy.TextPart{Text: "a red bicycle"},
				embeddingFile("image/png", "bicycle"),
			}},
			fantasy.TextEmbeddingInput("a blue car"),
		}})
		require.NoError(t, err)
		require.Len(t, response.Embeddings, 2)

		requests := geminiRequests(t, recorder.only(t))
		require.Len(t, requests, 2)
		parts := requestParts(t, requests[0])
		require.Len(t, parts, 2)
		require.Equal(t, "a red bicycle", parts[0]["text"])
		require.Equal(t, "image/png", parts[1]["inlineData"].(map[string]any)["mimeType"])
		require.Len(t, requestParts(t, requests[1]), 1)
	})

	t.Run("should send options only when they are set", func(t *testing.T) {
		t.Parallel()

		t.Run("unset", func(t *testing.T) {
			t.Parallel()

			model, recorder := newGeminiEmbeddingModel(t, http.StatusOK,
				geminiEmbeddingResponse(nil, []float32{1}, []float32{2}), "gemini-embedding-001")

			_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a", "b")})
			require.NoError(t, err)

			for _, request := range geminiRequests(t, recorder.only(t)) {
				require.NotContains(t, request, "taskType")
				require.NotContains(t, request, "title")
				require.NotContains(t, request, "outputDimensionality")
			}
		})

		t.Run("set", func(t *testing.T) {
			t.Parallel()

			model, recorder := newGeminiEmbeddingModel(t, http.StatusOK,
				geminiEmbeddingResponse(nil, []float32{1}, []float32{2}), "gemini-embedding-001")

			_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
				Inputs:     fantasy.TextEmbeddingInputs("a", "b"),
				Dimensions: new(int64(768)),
				ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{
					TaskType: new(TaskTypeRetrievalDocument),
					Title:    new("Notes"),
				}),
			})
			require.NoError(t, err)

			requests := geminiRequests(t, recorder.only(t))
			require.Len(t, requests, 2)
			for _, request := range requests {
				require.Equal(t, "RETRIEVAL_DOCUMENT", request["taskType"])
				require.Equal(t, "Notes", request["title"])
				require.EqualValues(t, 768, request["outputDimensionality"])
			}
		})

		t.Run("only the dimensions", func(t *testing.T) {
			t.Parallel()

			model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, geminiEmbeddingResponse(nil, []float32{1}), "gemini-embedding-2")

			_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
				Inputs:     fantasy.TextEmbeddingInputs("a"),
				Dimensions: new(int64(math.MaxInt32)),
			})
			require.NoError(t, err)

			request := geminiRequests(t, recorder.only(t))[0]
			require.EqualValues(t, math.MaxInt32, request["outputDimensionality"])
			require.NotContains(t, request, "taskType")
			require.NotContains(t, request, "title")
		})
	})

	t.Run("should map vectors, usage and the model", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiEmbeddingModel(t, http.StatusOK,
			geminiEmbeddingResponse(map[string]any{"promptTokenCount": 7, "totalTokenCount": 9}, []float32{0.25, -0.5, 1}, []float32{0.125, 2, 3}),
			"gemini-embedding-2")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a", "b")})
		require.NoError(t, err)

		require.Equal(t, [][]float32{{0.25, -0.5, 1}, {0.125, 2, 3}}, response.Embeddings)
		require.Equal(t, "gemini-embedding-2", response.Model)
		require.Equal(t, fantasy.Usage{InputTokens: 7, TotalTokens: 9}, response.Usage)
		require.Empty(t, response.Warnings)
		require.Nil(t, response.ProviderMetadata)
	})

	t.Run("should use the prompt tokens as the total when there is no total", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiEmbeddingModel(t, http.StatusOK,
			geminiEmbeddingResponse(map[string]any{"promptTokenCount": 7}, []float32{1}), "gemini-embedding-2")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a")})
		require.NoError(t, err)

		require.Equal(t, fantasy.Usage{InputTokens: 7, TotalTokens: 7}, response.Usage)
	})

	t.Run("should leave usage empty when the service reports none", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiEmbeddingModel(t, http.StatusOK, geminiEmbeddingResponse(nil, []float32{1}), "gemini-embedding-2")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a")})
		require.NoError(t, err)

		require.Equal(t, fantasy.Usage{}, response.Usage)
	})

	t.Run("should send images to a text-only model and leave the verdict to the service", func(t *testing.T) {
		t.Parallel()

		errorBody := embeddingProviderErrorBody(http.StatusBadRequest, "gemini-embedding-001 does not support image input", "INVALID_ARGUMENT")
		model, recorder := newGeminiEmbeddingModel(t, http.StatusBadRequest, errorBody, "gemini-embedding-001")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
			{Parts: []fantasy.MessagePart{embeddingFile("image/png", "cat")}},
		}})

		parts := requestParts(t, geminiRequests(t, recorder.only(t))[0])
		require.Equal(t, "image/png", parts[0]["inlineData"].(map[string]any)["mimeType"])
		var providerErr *fantasy.ProviderError
		require.ErrorAs(t, err, &providerErr)
		require.Equal(t, http.StatusBadRequest, providerErr.StatusCode)
		require.Equal(t, "gemini-embedding-001 does not support image input", providerErr.Message)
		require.False(t, providerErr.IsRetryable())
	})

	t.Run("should send a task type to a model that rejects it and leave the verdict to the service", func(t *testing.T) {
		t.Parallel()

		errorBody := embeddingProviderErrorBody(http.StatusBadRequest, "task_type is not supported by gemini-embedding-2", "INVALID_ARGUMENT")
		model, recorder := newGeminiEmbeddingModel(t, http.StatusBadRequest, errorBody, "gemini-embedding-2")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs:          fantasy.TextEmbeddingInputs("a"),
			ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{TaskType: new(TaskTypeSemanticSimilarity)}),
		})

		require.Equal(t, "SEMANTIC_SIMILARITY", geminiRequests(t, recorder.only(t))[0]["taskType"])
		var providerErr *fantasy.ProviderError
		require.ErrorAs(t, err, &providerErr)
		require.Equal(t, http.StatusBadRequest, providerErr.StatusCode)
		require.Equal(t, "task_type is not supported by gemini-embedding-2", providerErr.Message)
	})
}

func TestEmbeddingModel_InvalidArguments(t *testing.T) {
	t.Parallel()

	okResponse := geminiEmbeddingResponse(nil, []float32{1})
	textInput := fantasy.TextEmbeddingInputs("a")

	t.Run("should reject media types the service has no modality for", func(t *testing.T) {
		t.Parallel()

		for _, mediaType := range []string{"text/plain", "application/json", "application/zip", "imagepng"} {
			t.Run(mediaType, func(t *testing.T) {
				t.Parallel()

				model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2")

				_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
					fantasy.TextEmbeddingInput("fine"),
					{Parts: []fantasy.MessagePart{
						fantasy.TextPart{Text: "fine too"},
						embeddingFile(mediaType, "data"),
					}},
				}})

				fantasyErr := requireInvalidEmbeddingArgument(t, err)
				require.Contains(t, fantasyErr.Message, "input 1 part 1")
				require.Contains(t, fantasyErr.Message, mediaType)
				require.Empty(t, recorder.requests())
			})
		}
	})

	t.Run("should reject Vertex only options on the Gemini API backend", func(t *testing.T) {
		t.Parallel()

		tests := map[string]struct {
			options *EmbeddingProviderOptions
			message string
		}{
			"document ocr true":           {&EmbeddingProviderOptions{DocumentOCR: new(true)}, "document_ocr"},
			"document ocr false":          {&EmbeddingProviderOptions{DocumentOCR: new(false)}, "document_ocr"},
			"audio track extraction true": {&EmbeddingProviderOptions{AudioTrackExtraction: new(true)}, "audio_track_extraction"},
			"audio track extraction off":  {&EmbeddingProviderOptions{AudioTrackExtraction: new(false)}, "audio_track_extraction"},
		}
		for name, test := range tests {
			t.Run(name, func(t *testing.T) {
				t.Parallel()

				model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2")

				_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
					Inputs:          textInput,
					ProviderOptions: NewEmbeddingProviderOptions(test.options),
				})

				fantasyErr := requireInvalidEmbeddingArgument(t, err)
				require.Contains(t, fantasyErr.Message, test.message)
				require.Contains(t, fantasyErr.Message, "Vertex")
				require.Empty(t, recorder.requests())
			})
		}
	})

	t.Run("should reject dimensions that do not fit the API", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs:     textInput,
			Dimensions: new(int64(math.MaxInt32) + 1),
		})

		fantasyErr := requireInvalidEmbeddingArgument(t, err)
		require.Contains(t, fantasyErr.Message, "dimensions")
		require.Empty(t, recorder.requests())
	})

	t.Run("should reject provider options of another type", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs:          textInput,
			ProviderOptions: fantasy.ProviderOptions{Name: &ImageProviderOptions{}},
		})

		fantasyErr := requireInvalidEmbeddingArgument(t, err)
		require.Contains(t, fantasyErr.Message, "EmbeddingProviderOptions")
		require.Empty(t, recorder.requests())
	})

	t.Run("should reject calls the core package rejects", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2")

		for name, call := range map[string]fantasy.EmbeddingCall{
			"no inputs":         {},
			"empty input":       {Inputs: []fantasy.EmbeddingInput{{}}},
			"empty text":        {Inputs: fantasy.TextEmbeddingInputs("")},
			"zero dimensions":   {Inputs: textInput, Dimensions: new(int64(0))},
			"file without data": {Inputs: []fantasy.EmbeddingInput{{Parts: []fantasy.MessagePart{fantasy.FilePart{MediaType: "image/png"}}}}},
		} {
			_, err := model.Embed(t.Context(), call)
			requireInvalidEmbeddingArgument(t, err)
			require.Empty(t, recorder.requests(), name)
		}
	})
}

func TestEmbeddingModel_InvalidResponses(t *testing.T) {
	t.Parallel()

	tests := map[string]struct {
		response any
		inputs   int
	}{
		"fewer embeddings than inputs":  {geminiEmbeddingResponse(nil, []float32{1}), 2},
		"more embeddings than inputs":   {geminiEmbeddingResponse(nil, []float32{1}, []float32{2}), 1},
		"no embeddings":                 {map[string]any{}, 1},
		"empty values":                  {map[string]any{"embeddings": []any{map[string]any{"values": []float32{}}}}, 1},
		"missing values":                {map[string]any{"embeddings": []any{map[string]any{}}}, 1},
		"null embedding":                {map[string]any{"embeddings": []any{nil}}, 1},
		"null embedding after a vector": {map[string]any{"embeddings": []any{map[string]any{"values": []float32{1}}, nil}}, 2},
		"empty values after a vector":   {map[string]any{"embeddings": []any{map[string]any{"values": []float32{1}}, map[string]any{"values": []float32{}}}}, 2},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, test.response, "gemini-embedding-2")
			texts := make([]string, test.inputs)
			for i := range texts {
				texts[i] = "text"
			}

			response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs(texts...)})

			requireInvalidEmbeddingResponse(t, err)
			require.Nil(t, response)
			require.Len(t, recorder.requests(), 1)
		})
	}
}

func TestEmbeddingModel_Errors(t *testing.T) {
	t.Parallel()

	t.Run("should not retry retryable statuses and report them as retryable", func(t *testing.T) {
		t.Parallel()

		for _, status := range []int{http.StatusTooManyRequests, http.StatusServiceUnavailable} {
			t.Run(http.StatusText(status), func(t *testing.T) {
				t.Parallel()

				model, recorder := newGeminiEmbeddingModel(t, status, embeddingProviderErrorBody(status, "quota exceeded", "RESOURCE_EXHAUSTED"), "gemini-embedding-2")

				response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a")})

				// NOTE: The SDK has no retry options set, so one failed request is the end of the call.
				require.Len(t, recorder.requests(), 1)
				require.Nil(t, response)
				var providerErr *fantasy.ProviderError
				require.ErrorAs(t, err, &providerErr)
				require.True(t, providerErr.IsRetryable())
				require.Equal(t, status, providerErr.StatusCode)
				require.Equal(t, "quota exceeded", providerErr.Message)
				require.Equal(t, "RESOURCE_EXHAUSTED", providerErr.ErrorType)
			})
		}
	})

	t.Run("should not report a client error as retryable", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusBadRequest, embeddingProviderErrorBody(http.StatusBadRequest, "input is too long", "INVALID_ARGUMENT"), "gemini-embedding-2")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a")})

		require.Len(t, recorder.requests(), 1)
		var providerErr *fantasy.ProviderError
		require.ErrorAs(t, err, &providerErr)
		require.False(t, providerErr.IsRetryable())
		require.Equal(t, http.StatusBadRequest, providerErr.StatusCode)
	})
}

func TestEmbeddingModel_UserAgent(t *testing.T) {
	t.Parallel()

	call := fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a")}
	okResponse := geminiEmbeddingResponse(nil, []float32{1})

	t.Run("should send the provider user agent", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2", WithUserAgent("provider-agent/1.0"))

		_, err := model.Embed(t.Context(), call)
		require.NoError(t, err)

		userAgent := recorder.only(t).header.Get("User-Agent")
		// NOTE: The SDK adds its own labels, such as "google-genai-sdk/1.79.1", to the user agent.
		require.True(t, strings.HasPrefix(userAgent, "provider-agent/1.0"), userAgent)
	})

	t.Run("should send the default user agent without an explicit one", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2")

		_, err := model.Embed(t.Context(), call)
		require.NoError(t, err)

		userAgent := recorder.only(t).header.Get("User-Agent")
		require.True(t, strings.HasPrefix(userAgent, "Charm-Fantasy/"+fantasy.Version+" (https://charm.land/fantasy)"), userAgent)
	})

	t.Run("should override the user agent per call", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2", WithUserAgent("provider-agent/1.0"))
		call := call
		call.UserAgent = "my-agent/1.0"

		_, err := model.Embed(t.Context(), call)
		require.NoError(t, err)

		require.Equal(t, "my-agent/1.0", recorder.only(t).header.Get("User-Agent"))
	})

	t.Run("should override headers per call", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2", WithHeaders(map[string]string{"X-Source": "provider", "X-Other": "kept"}))
		call := call
		call.Headers = map[string]string{"X-Source": "call"}

		_, err := model.Embed(t.Context(), call)
		require.NoError(t, err)

		header := recorder.only(t).header
		require.Equal(t, "call", header.Get("X-Source"))
		require.Equal(t, "kept", header.Get("X-Other"))
	})

	t.Run("should not let one call leak its user agent into the next", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiEmbeddingModel(t, http.StatusOK, okResponse, "gemini-embedding-2", WithUserAgent("provider-agent/1.0"))
		first := call
		first.UserAgent = "my-agent/1.0"

		_, err := model.Embed(t.Context(), first)
		require.NoError(t, err)
		_, err = model.Embed(t.Context(), call)
		require.NoError(t, err)

		requests := recorder.requests()
		require.Len(t, requests, 2)
		require.Equal(t, "my-agent/1.0", requests[0].header.Get("User-Agent"))
		require.True(t, strings.HasPrefix(requests[1].header.Get("User-Agent"), "provider-agent/1.0"), requests[1].header.Get("User-Agent"))
	})
}

func TestEmbeddingModel_Vertex(t *testing.T) {
	t.Parallel()

	t.Run("should predict text embeddings and sum the token counts", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusOK,
			vertexPredictResponse([]int{3, 4}, []float32{0.25, 0.5}, []float32{-1, 2}),
			"text-embedding-005")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("first", "second")})
		require.NoError(t, err)

		request := recorder.only(t)
		require.True(t, strings.HasSuffix(request.path, "/publishers/google/models/text-embedding-005:predict"), request.path)
		require.Contains(t, request.path, "/projects/test-project/locations/us-central1/")
		instances := request.body["instances"].([]any)
		require.Len(t, instances, 2)
		require.Equal(t, "first", instances[0].(map[string]any)["content"])
		require.Equal(t, "second", instances[1].(map[string]any)["content"])

		require.Equal(t, [][]float32{{0.25, 0.5}, {-1, 2}}, response.Embeddings)
		require.Equal(t, "text-embedding-005", response.Model)
		// NOTE: The predict response has no usage metadata, only a token count per input.
		require.Equal(t, fantasy.Usage{InputTokens: 7, TotalTokens: 7}, response.Usage)
		require.Nil(t, response.ProviderMetadata)
	})

	t.Run("should leave usage empty when any input lacks statistics", func(t *testing.T) {
		t.Parallel()

		model, _ := newVertexEmbeddingModel(t, http.StatusOK,
			vertexPredictResponse([]int{5}, []float32{1}, []float32{2}),
			"text-embedding-005")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a", "b")})
		require.NoError(t, err)

		require.Equal(t, fantasy.Usage{}, response.Usage)
	})

	t.Run("should leave usage empty when the predict response has no statistics", func(t *testing.T) {
		t.Parallel()

		model, _ := newVertexEmbeddingModel(t, http.StatusOK, vertexPredictResponse(nil, []float32{1}), "text-embedding-005")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a")})
		require.NoError(t, err)

		require.Equal(t, fantasy.Usage{}, response.Usage)
	})

	t.Run("should map the options of the predict endpoint", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusOK, vertexPredictResponse([]int{1}, []float32{1}), "text-embedding-005")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs:     fantasy.TextEmbeddingInputs("a"),
			Dimensions: new(int64(256)),
			ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{
				TaskType: new(TaskTypeRetrievalDocument),
				Title:    new("Notes"),
			}),
		})
		require.NoError(t, err)

		request := recorder.only(t)
		instance := request.body["instances"].([]any)[0].(map[string]any)
		require.Equal(t, "RETRIEVAL_DOCUMENT", instance["task_type"])
		require.Equal(t, "Notes", instance["title"])
		require.EqualValues(t, 256, request.body["parameters"].(map[string]any)["outputDimensionality"])
	})

	t.Run("should warn about each Vertex only option ignored by predict", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusOK, vertexPredictResponse([]int{1}, []float32{1}), "text-embedding-005")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs: fantasy.TextEmbeddingInputs("a"),
			ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{
				DocumentOCR:          new(true),
				AudioTrackExtraction: new(false),
			}),
		})

		require.NoError(t, err)
		require.Equal(t, []string{"document_ocr", "audio_track_extraction"}, warningSettings(response.Warnings))
		for _, warning := range response.Warnings {
			require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, warning.Type)
			require.Contains(t, warning.Message, ":predict")
		}
		require.Equal(t, map[string]any{"instances": []any{map[string]any{"content": "a"}}}, recorder.only(t).body)
	})

	t.Run("should send Vertex only options to gemini models", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusOK, map[string]any{
			"embedding":     map[string]any{"values": []float32{0.5, 0.25}},
			"usageMetadata": map[string]any{"promptTokenCount": 11},
		}, "gemini-embedding-2")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs: []fantasy.EmbeddingInput{{Parts: []fantasy.MessagePart{
				fantasy.TextPart{Text: "a scanned invoice"},
				embeddingFile("application/pdf", "pdf-bytes"),
			}}},
			ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{
				DocumentOCR:          new(true),
				AudioTrackExtraction: new(false),
			}),
		})
		require.NoError(t, err)

		request := recorder.only(t)
		require.True(t, strings.HasSuffix(request.path, "/publishers/google/models/gemini-embedding-2:embedContent"), request.path)
		parts := requestParts(t, request.body)
		require.Len(t, parts, 2)
		require.Equal(t, "a scanned invoice", parts[0]["text"])
		require.Equal(t, "application/pdf", parts[1]["inlineData"].(map[string]any)["mimeType"])
		config := request.body["embedContentConfig"].(map[string]any)
		require.Equal(t, true, config["documentOcr"])
		require.Equal(t, false, config["audioTrackExtraction"])

		require.Equal(t, [][]float32{{0.5, 0.25}}, response.Embeddings)
		require.Equal(t, fantasy.Usage{InputTokens: 11, TotalTokens: 11}, response.Usage)
		require.Empty(t, response.Warnings)
	})

	t.Run("should batch several inputs for gemini models", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusOK, map[string]any{
			"embeddings": []map[string]any{{"values": []float32{1, 2}}, {"values": []float32{3, 4}}},
		}, "gemini-embedding-2")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
			Inputs:          fantasy.TextEmbeddingInputs("first", "second"),
			ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{DocumentOCR: new(true)}),
		})
		require.NoError(t, err)

		request := recorder.only(t)
		require.True(t, strings.HasSuffix(request.path, "/publishers/google/models/gemini-embedding-2:batchEmbedContents"), request.path)
		requests := geminiRequests(t, request)
		require.Len(t, requests, 2)
		require.Equal(t, "first", requestParts(t, requests[0])[0]["text"])
		require.Equal(t, "second", requestParts(t, requests[1])[0]["text"])
		require.Equal(t, true, requests[0]["embedContentConfig"].(map[string]any)["documentOcr"])

		require.Equal(t, [][]float32{{1, 2}, {3, 4}}, response.Embeddings)
		require.Equal(t, fantasy.Usage{}, response.Usage)
	})

	t.Run("should surface the SDK error when a maas model gets several inputs", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusOK, map[string]any{}, "text-embedding-maas")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("first", "second")})

		// NOTE: The SDK refuses this before it sends anything, with a plain error
		// that is neither a fantasy.Error nor a fantasy.ProviderError.
		require.Error(t, err)
		require.Nil(t, response)
		require.Empty(t, recorder.requests())
		var fantasyErr *fantasy.Error
		require.NotErrorAs(t, err, &fantasyErr)
		var providerErr *fantasy.ProviderError
		require.NotErrorAs(t, err, &providerErr)
	})

	t.Run("should surface the SDK error when a predict model gets a file first", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusOK, vertexPredictResponse(nil, []float32{1}), "text-embedding-005")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
			{Parts: []fantasy.MessagePart{embeddingFile("image/png", "cat")}},
		}})

		// NOTE: The SDK builds :predict requests from the text of the first part,
		// so files never reach the service on this route.
		require.Error(t, err)
		require.Empty(t, recorder.requests())
	})

	t.Run("should only send the text of the first part to predict models", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusOK, vertexPredictResponse(nil, []float32{1}, []float32{2}), "text-embedding-005")

		response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: []fantasy.EmbeddingInput{
			fantasy.TextEmbeddingInput("single"),
			{Parts: []fantasy.MessagePart{
				fantasy.TextPart{Text: "first part"},
				fantasy.TextPart{Text: "second part"},
				embeddingFile("image/png", "cat"),
			}},
		}})
		require.NoError(t, err)

		instances := recorder.only(t).body["instances"].([]any)
		require.Equal(t, []any{map[string]any{"content": "single"}, map[string]any{"content": "first part"}}, instances)
		require.Len(t, response.Warnings, 1)
		require.Contains(t, response.Warnings[0].Message, "input 1")
		require.Contains(t, response.Warnings[0].Message, "after the first")
	})

	t.Run("should report service errors as provider errors", func(t *testing.T) {
		t.Parallel()

		model, recorder := newVertexEmbeddingModel(t, http.StatusTooManyRequests, embeddingProviderErrorBody(http.StatusTooManyRequests, "quota exceeded", "RESOURCE_EXHAUSTED"), "text-embedding-005")

		_, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a")})

		require.Len(t, recorder.requests(), 1)
		var providerErr *fantasy.ProviderError
		require.ErrorAs(t, err, &providerErr)
		require.True(t, providerErr.IsRetryable())
	})
}

func TestEmbeddingModel_NilOptions(t *testing.T) {
	t.Parallel()
	require.Equal(t, fantasy.ProviderOptions{}, NewEmbeddingProviderOptions(nil))
	model, _ := newGeminiEmbeddingModel(t, http.StatusOK, geminiEmbeddingResponse(nil, []float32{1}), "gemini-embedding-2")
	response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{
		Inputs: fantasy.TextEmbeddingInputs("a"), ProviderOptions: fantasy.ProviderOptions{Name: (*EmbeddingProviderOptions)(nil)},
	})
	require.NoError(t, err)
	require.Len(t, response.Embeddings, 1)
}

func TestEmbeddingModel_PredictRoute(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		backend genai.Backend
		id      string
		predict bool
	}{
		{genai.BackendGeminiAPI, "text-embedding-005", false},
		{genai.BackendVertexAI, "projects/gemini-maas/locations/x/publishers/google/models/text-embedding-005", true},
		{genai.BackendVertexAI, "models/pre-gemini-post", false},
		{genai.BackendVertexAI, "models/pre-maas-post", false},
		{genai.BackendVertexAI, "models/Gemini-MAAS", true},
	} {
		t.Run(tc.id, func(t *testing.T) { require.Equal(t, tc.predict, usesVertexPredict(tc.backend, tc.id)) })
	}
}

func TestEmbeddingModel_TruncatedInputs(t *testing.T) {
	t.Parallel()
	body := vertexPredictResponse([]int{5, 7, 9}, []float32{1}, []float32{2}, []float32{3})
	predictions := body["predictions"].([]map[string]any)
	for _, idx := range []int{0, 2} {
		stats := predictions[idx]["embeddings"].(map[string]any)["statistics"].(map[string]any)
		stats["truncated"] = true
	}
	model, _ := newVertexEmbeddingModel(t, http.StatusOK, body, "text-embedding-005")
	response, err := model.Embed(t.Context(), fantasy.EmbeddingCall{Inputs: fantasy.TextEmbeddingInputs("a", "b", "c")})
	require.NoError(t, err)
	require.Equal(t, fantasy.Usage{InputTokens: 21, TotalTokens: 21}, response.Usage)
	require.Len(t, response.Warnings, 2)
	for i, label := range []string{"input 0", "input 2"} {
		require.Equal(t, fantasy.CallWarningTypeOther, response.Warnings[i].Type)
		require.Contains(t, response.Warnings[i].Message, label)
		require.Contains(t, response.Warnings[i].Message, "truncated")
	}
}

func TestEmbeddingProviderOptions_JSON(t *testing.T) {
	t.Parallel()

	t.Run("should round trip through the registry", func(t *testing.T) {
		t.Parallel()

		options := NewEmbeddingProviderOptions(&EmbeddingProviderOptions{
			TaskType:             new(TaskTypeRetrievalDocument),
			Title:                new("Notes"),
			DocumentOCR:          new(true),
			AudioTrackExtraction: new(false),
		})
		data, err := json.Marshal(options)
		require.NoError(t, err)
		require.JSONEq(t, `{"google":{"type":"google.embedding_options","data":{"task_type":"RETRIEVAL_DOCUMENT","title":"Notes","document_ocr":true,"audio_track_extraction":false}}}`, string(data))

		var raw map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(data, &raw))
		decoded, err := fantasy.UnmarshalProviderOptions(raw)
		require.NoError(t, err)
		require.Equal(t, options, decoded)
	})

	t.Run("should round trip unset options", func(t *testing.T) {
		t.Parallel()

		options := NewEmbeddingProviderOptions(&EmbeddingProviderOptions{})
		data, err := json.Marshal(options)
		require.NoError(t, err)
		require.JSONEq(t, `{"google":{"type":"google.embedding_options","data":{"task_type":null,"title":null,"document_ocr":null,"audio_track_extraction":null}}}`, string(data))

		var raw map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(data, &raw))
		decoded, err := fantasy.UnmarshalProviderOptions(raw)
		require.NoError(t, err)
		require.Equal(t, options, decoded)
	})

	t.Run("should use the registered type id", func(t *testing.T) {
		t.Parallel()

		require.Equal(t, "google.embedding_options", TypeProviderEmbeddingOptions)
	})

	t.Run("should define the task types", func(t *testing.T) {
		t.Parallel()

		require.Equal(t, []string{
			"RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT", "SEMANTIC_SIMILARITY", "CLASSIFICATION",
			"CLUSTERING", "QUESTION_ANSWERING", "FACT_VERIFICATION", "CODE_RETRIEVAL_QUERY",
		}, []string{
			TaskTypeRetrievalQuery, TaskTypeRetrievalDocument, TaskTypeSemanticSimilarity, TaskTypeClassification,
			TaskTypeClustering, TaskTypeQuestionAnswering, TaskTypeFactVerification, TaskTypeCodeRetrievalQuery,
		})
	})
}
