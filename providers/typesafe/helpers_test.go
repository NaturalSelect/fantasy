package typesafe

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

type recordedRequest struct {
	method string
	path   string
	query  string
	header http.Header
	body   []byte
}

func (r recordedRequest) fields(t *testing.T) map[string]json.RawMessage {
	t.Helper()
	var fields map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(r.body, &fields))
	return fields
}

func (r recordedRequest) question(t *testing.T, name string) string {
	t.Helper()
	var questions map[string]json.RawMessage
	require.NoError(t, json.Unmarshal(r.fields(t)["questions"], &questions))
	return string(questions[name])
}

type testServer struct {
	url string

	mu       sync.Mutex
	requests []recordedRequest
}

func (s *testServer) add(r recordedRequest) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.requests = append(s.requests, r)
}

func (s *testServer) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.requests)
}

func (s *testServer) last(t *testing.T) recordedRequest {
	t.Helper()
	s.mu.Lock()
	defer s.mu.Unlock()
	require.NotEmpty(t, s.requests, "the server saw no request")
	return s.requests[len(s.requests)-1]
}

func newTestProvider(t *testing.T, handler http.HandlerFunc, opts ...Option) (fantasy.DecisionModelProvider, *testServer) {
	t.Helper()

	recorded := &testServer{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		recorded.add(recordedRequest{method: r.Method, path: r.URL.Path, query: r.URL.RawQuery, header: r.Header.Clone(), body: body})
		r.Body = io.NopCloser(bytes.NewReader(body))
		handler(w, r)
	}))
	t.Cleanup(server.Close)
	recorded.url = server.URL

	p, err := New(append([]Option{WithAPIKey("test-key"), WithBaseURL(server.URL)}, opts...)...)
	require.NoError(t, err)
	decisionProvider, ok := p.(fantasy.DecisionModelProvider)
	require.True(t, ok)
	return decisionProvider, recorded
}

func newTestModel(t *testing.T, handler http.HandlerFunc, opts ...Option) (fantasy.DecisionModel, *testServer) {
	t.Helper()

	provider, server := newTestProvider(t, handler, opts...)
	model, err := provider.DecisionModel(t.Context(), "")
	require.NoError(t, err)
	return model, server
}

// respond answers every request with the given status, headers (as key and
// value pairs) and body.
func respond(status int, body string, headers ...string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		for i := 0; i+1 < len(headers); i += 2 {
			w.Header().Set(headers[i], headers[i+1])
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// answerAll answers every question of the request with a valid answer of its type.
func answerAll(w http.ResponseWriter, r *http.Request) {
	var request struct {
		Questions map[string]struct {
			Type     string          `json:"type"`
			Criteria json.RawMessage `json:"criteria"`
		} `json:"questions"`
	}
	_ = json.NewDecoder(r.Body).Decode(&request)

	answers := map[string]any{}
	for name, question := range request.Questions {
		switch question.Type {
		case "noul":
			answers[name] = map[string]any{"type": "noul", "noul": 0.5}
		case "choice":
			var options map[string]json.RawMessage
			if err := json.Unmarshal(question.Criteria, &options); err != nil || len(options) == 0 {
				http.Error(w, "invalid choice criteria", http.StatusBadRequest)
				return
			}
			for option := range options {
				answers[name] = map[string]any{"type": "choice", "choice": option, "probabilities": map[string]int{option: 1}}
				break
			}
		case "score":
			answers[name] = map[string]any{"type": "score", "score": 0, "probabilities": map[string]int{"0": 1}}
		case "box":
			answers[name] = map[string]any{"type": "box", "box": map[string]any{"top": 0, "bottom": 1, "left": 0, "right": 1}}
		}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"model": "jev-1.14.0", "answers": answers})
}

func textCall(questions map[string]fantasy.Question) fantasy.DecisionCall {
	return fantasy.DecisionCall{
		Input:     []fantasy.MessagePart{fantasy.TextPart{Text: "hello"}},
		Questions: questions,
	}
}

func noulCall() fantasy.DecisionCall {
	return textCall(map[string]fantasy.Question{
		"q": fantasy.NoulQuestion{Instructions: fantasy.String("Is it spam?")},
	})
}

func pngPart(data string) fantasy.FilePart {
	return fantasy.FilePart{Data: []byte(data), MediaType: "image/png"}
}

func imageRef(data string) string {
	sum := sha256.Sum256([]byte(data))
	return "image_" + hex.EncodeToString(sum[:])[:16]
}
