package openaicompat

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestProviderEmbedsThroughOpenAI(t *testing.T) {
	t.Parallel()

	var path string
	var body map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"object":"list","model":"embed-1","data":[{"object":"embedding","index":0,"embedding":[0.5,-0.25]}],"usage":{"prompt_tokens":3,"total_tokens":3}}`))
	}))
	t.Cleanup(srv.Close)

	p, err := New(WithBaseURL(srv.URL), WithAPIKey("key"), WithName("compat"))
	require.NoError(t, err)

	embeddingProvider, ok := p.(fantasy.EmbeddingModelProvider)
	require.True(t, ok, "derived providers inherit embeddings from the openai provider")

	model, err := embeddingProvider.EmbeddingModel(context.Background(), "embed-1")
	require.NoError(t, err)
	require.Equal(t, "compat", model.Provider())

	response, err := model.Embed(context.Background(), fantasy.EmbeddingCall{
		Inputs: fantasy.TextEmbeddingInputs("hello"),
	})
	require.NoError(t, err)
	require.Equal(t, [][]float32{{0.5, -0.25}}, response.Embeddings)
	require.Equal(t, int64(3), response.Usage.InputTokens)
	require.Equal(t, "/embeddings", path)
	require.Equal(t, "embed-1", body["model"])
}
