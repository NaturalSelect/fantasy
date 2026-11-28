package typesafe

import (
	"context"
	"io"
	"net/http"
	"strconv"
	"strings"
	"testing"
	"testing/iotest"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func decideProviderErr(t *testing.T, model fantasy.DecisionModel, call fantasy.DecisionCall) *fantasy.ProviderError {
	t.Helper()

	response, err := model.Decide(t.Context(), call)
	require.Nil(t, response)
	var providerErr *fantasy.ProviderError
	require.ErrorAs(t, err, &providerErr)
	return providerErr
}

func TestDecide_Errors(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name          string
		status        int
		headers       []string
		body          string
		wantTitle     string
		wantMessage   string
		wantErrorType string
		wantRetryable bool
	}{
		{
			name:        "400 string detail",
			status:      http.StatusBadRequest,
			body:        `{"detail":"questions must not be empty"}`,
			wantTitle:   "bad request",
			wantMessage: "questions must not be empty",
		},
		{
			name:          "400 object detail with a code",
			status:        http.StatusBadRequest,
			body:          `{"detail": {"code": "image_unreferenced", "message": "image_abc is never used"}}`,
			wantTitle:     "bad request",
			wantMessage:   `{"code":"image_unreferenced","message":"image_abc is never used"}`,
			wantErrorType: "image_unreferenced",
		},
		{
			name:          "400 object detail with a type",
			status:        http.StatusBadRequest,
			body:          `{"detail":{"type":"invalid_state","message":"bad"}}`,
			wantTitle:     "bad request",
			wantMessage:   `{"type":"invalid_state","message":"bad"}`,
			wantErrorType: "invalid_state",
		},
		{
			name:        "400 object detail without a code",
			status:      http.StatusBadRequest,
			body:        `{"detail":{"message":"bad"}}`,
			wantTitle:   "bad request",
			wantMessage: `{"message":"bad"}`,
		},
		{
			name:        "400 detail null falls back to the body",
			status:      http.StatusBadRequest,
			body:        `{"detail":null}`,
			wantTitle:   "bad request",
			wantMessage: `{"detail":null}`,
		},
		{
			name:   "422 validation list",
			status: http.StatusUnprocessableEntity,
			body: `{"detail":[
				{"loc":["body","questions"],"msg":"Field required","type":"missing","input":null},
				{"loc":["body","questions","tone","criteria",0],"msg":"Too short","type":"too_short","ctx":{"min_length":1}},
				{"loc":[],"msg":"No location","type":"other"}
			]}`,
			wantTitle:     strings.ToLower(http.StatusText(http.StatusUnprocessableEntity)),
			wantMessage:   "body.questions: Field required; body.questions.tone.criteria.0: Too short; No location",
			wantErrorType: "missing",
		},
		{
			name:        "401 plain body",
			status:      http.StatusUnauthorized,
			body:        "invalid api key",
			wantTitle:   "unauthorized",
			wantMessage: "invalid api key",
		},
		{
			name:        "413 empty body",
			status:      http.StatusRequestEntityTooLarge,
			wantTitle:   strings.ToLower(http.StatusText(http.StatusRequestEntityTooLarge)),
			wantMessage: http.StatusText(http.StatusRequestEntityTooLarge),
		},
		{
			name:          "429 with retry-after",
			status:        http.StatusTooManyRequests,
			headers:       []string{"Retry-After", "2"},
			wantTitle:     "too many requests",
			wantMessage:   "Too Many Requests",
			wantRetryable: true,
		},
		{
			name:          "529 overloaded",
			status:        529,
			body:          "overloaded",
			wantTitle:     "provider request failed",
			wantMessage:   "overloaded",
			wantRetryable: true,
		},
		{
			name:          "502 html body",
			status:        http.StatusBadGateway,
			body:          "<html>bad gateway</html>\n",
			wantTitle:     "bad gateway",
			wantMessage:   "<html>bad gateway</html>",
			wantRetryable: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, server := newTestModel(t, respond(tt.status, tt.body, tt.headers...))
			providerErr := decideProviderErr(t, model, noulCall())

			require.Equal(t, tt.status, providerErr.StatusCode)
			require.Equal(t, tt.wantTitle, providerErr.Title)
			require.Equal(t, tt.wantMessage, providerErr.Message)
			require.Equal(t, tt.wantErrorType, providerErr.ErrorType)
			require.Equal(t, tt.wantRetryable, providerErr.IsRetryable())
			require.False(t, providerErr.IsContextTooLarge())
			require.Equal(t, server.url+"/v1/systemone", providerErr.URL)
			require.Equal(t, tt.body, string(providerErr.ResponseBody))
			require.Equal(t, string(server.last(t).body), string(providerErr.RequestBody))
		})
	}
}

func modelWithClient(t *testing.T, do doFunc) fantasy.DecisionModel {
	t.Helper()
	p, err := New(WithHTTPClient(do))
	require.NoError(t, err)
	m, err := p.(fantasy.DecisionModelProvider).DecisionModel(t.Context(), "")
	require.NoError(t, err)
	return m
}

func TestDecide_ResponseBodyLimits(t *testing.T) {
	t.Parallel()
	const body = `{"answers":{"q":{"type":"noul","noul":0}}}`
	for _, tt := range []struct {
		name               string
		status, size, read int
	}{
		{"error body", http.StatusTooManyRequests, (1 << 20) + 10, 1 << 20},
		{"success at limit", http.StatusOK, 8 << 20, 8 << 20},
		{"success above limit", http.StatusOK, (8 << 20) + 10, (8 << 20) + 1},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r := strings.NewReader(body + strings.Repeat(" ", tt.size-len(body)))
			model := modelWithClient(t, func(*http.Request) (*http.Response, error) {
				return &http.Response{StatusCode: tt.status, Body: io.NopCloser(r)}, nil
			})
			response, err := model.Decide(t.Context(), noulCall())
			require.Equal(t, tt.read, tt.size-r.Len())
			switch {
			case tt.status != http.StatusOK:
				var pe *fantasy.ProviderError
				require.ErrorAs(t, err, &pe)
				require.Equal(t, tt.status, pe.StatusCode)
				require.Len(t, pe.ResponseBody, tt.read)
			case tt.size > tt.read:
				var invalid *fantasy.Error
				require.ErrorAs(t, err, &invalid)
				require.Equal(t, "invalid response", invalid.Title)
				require.Contains(t, invalid.Message, "exceeds")
			default:
				require.NoError(t, err)
				require.NotNil(t, response)
			}
		})
	}
}

func TestDecide_PartialErrorBody(t *testing.T) {
	t.Parallel()
	for _, status := range []int{http.StatusTooManyRequests, http.StatusUnauthorized} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			model, _ := newTestModel(t, respond(status, "partial", "Content-Length", "1000", "Retry-After", "3"))
			err := decideProviderErr(t, model, noulCall())
			require.Equal(t, status, err.StatusCode)
			require.Equal(t, "partial", string(err.ResponseBody))
			require.Equal(t, "3", err.ResponseHeaders["retry-after"])
			require.Equal(t, "3", err.ResponseHeaders["Retry-After"])
			require.Equal(t, status == http.StatusTooManyRequests, err.IsRetryable())
		})
	}
}

func TestDecide_CanceledResponseBody(t *testing.T) {
	t.Parallel()
	for _, status := range []int{http.StatusOK, http.StatusUnauthorized} {
		t.Run(strconv.Itoa(status), func(t *testing.T) {
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			model := modelWithClient(t, func(*http.Request) (*http.Response, error) {
				cancel()
				return &http.Response{StatusCode: status, Body: io.NopCloser(iotest.ErrReader(io.ErrUnexpectedEOF))}, nil
			})
			_, err := model.Decide(ctx, noulCall())
			require.Equal(t, context.Canceled, err)
		})
	}
}

func TestDecide_ContextErrorMarker(t *testing.T) {
	t.Parallel()
	for _, tt := range []struct {
		name   string
		status int
		body   string
		want   bool
	}{
		{"422 question name", 422, `{"detail":[{"loc":["questions","max_tokens_exceeded"],"msg":"missing","type":"validation"}]}`, false},
		{"wrong status", 429, `{"error":"max_tokens_exceeded"}`, false},
		{"echoed input", 400, `{"detail":{"code":"invalid_state","input":"max_tokens_exceeded"}}`, false},
		{"echoed input in error text", 400, `{"detail":"invalid question: max_tokens_exceeded"}`, false},
		{"unrelated error", 400, `{"error":"not_max_tokens_exceeded"}`, false},
		{"invalid json", 400, `{"error":"max_tokens_exceeded"`, false},
		{"escaped error code", 400, `{"error":"max_tokens_\u0065xceeded"}`, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			err := toProviderErr(&http.Response{StatusCode: tt.status}, []byte(tt.body), "")
			require.Equal(t, tt.want, err.IsContextTooLarge())
		})
	}
}

func TestDecide_ErrorDetails(t *testing.T) {
	t.Parallel()

	t.Run("429 exposes retry-after and is not retried by the provider", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, respond(http.StatusTooManyRequests, `{"detail":"slow down"}`,
			"Retry-After", "3", "Retry-After-Ms", "2500"))
		providerErr := decideProviderErr(t, model, noulCall())

		require.True(t, providerErr.IsRetryable())
		require.Equal(t, "3", providerErr.ResponseHeaders["retry-after"])
		require.Equal(t, "3", providerErr.ResponseHeaders["Retry-After"])
		require.Equal(t, "2500", providerErr.ResponseHeaders["retry-after-ms"])
		require.Equal(t, 1, server.count(), "the provider must not retry")
	})

	t.Run("max_tokens_exceeded marks the context as too large", func(t *testing.T) {
		t.Parallel()

		bodies := map[string]string{
			"string detail": `{"detail":"max_tokens_exceeded: the input is too long"}`,
			"object detail": `{"detail":{"code":"max_tokens_exceeded","limit":1000}}`,
			"other field":   `{"error":"max_tokens_exceeded"}`,
		}
		for name, body := range bodies {
			t.Run(name, func(t *testing.T) {
				t.Parallel()

				model, _ := newTestModel(t, respond(http.StatusBadRequest, body))
				providerErr := decideProviderErr(t, model, noulCall())
				require.True(t, providerErr.ContextTooLargeErr)
				require.True(t, providerErr.IsContextTooLarge())
				require.False(t, providerErr.IsRetryable())
			})
		}
	})

	t.Run("the request body is left out when it has images", func(t *testing.T) {
		t.Parallel()

		model, _ := newTestModel(t, respond(http.StatusBadRequest, `{"detail":"bad"}`))
		providerErr := decideProviderErr(t, model, imageCall(fantasy.NoulQuestion{Instructions: fantasy.String("Is it a logo?")}))
		require.Empty(t, providerErr.RequestBody)
		require.Equal(t, "bad", providerErr.Message)
	})

	t.Run("a truncated detail is passed through", func(t *testing.T) {
		t.Parallel()

		model, _ := newTestModel(t, respond(http.StatusBadRequest, `{"detail":"the state is too lo…"}`))
		require.Equal(t, "the state is too lo…", decideProviderErr(t, model, noulCall()).Message)
	})
}
