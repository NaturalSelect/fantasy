package typesafe

import (
	"net/http"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func imageCall(question fantasy.Question) fantasy.DecisionCall {
	return fantasy.DecisionCall{
		Input:     []fantasy.MessagePart{pngPart("image")},
		Questions: map[string]fantasy.Question{"q": question},
	}
}

func TestDecide_Headers(t *testing.T) {
	t.Parallel()

	t.Run("defaults", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		_, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)

		header := server.last(t).header
		require.Equal(t, "Bearer test-key", header.Get("Authorization"))
		require.Equal(t, "application/json", header.Get("Content-Type"))
		require.True(t, strings.HasPrefix(header.Get("User-Agent"), "Charm-Fantasy/"), header.Get("User-Agent"))
		require.NotContains(t, header, http.CanonicalHeaderKey(imagesHeader))
		require.NotContains(t, header, http.CanonicalHeaderKey(longContextHeader))
	})

	t.Run("provider level user agent and headers", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(
			t, answerAll,
			WithUserAgent("provider-agent"),
			WithHeaders(map[string]string{"X-Custom": "provider", "X-Other": "other"}),
		)
		_, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)

		header := server.last(t).header
		require.Equal(t, "provider-agent", header.Get("User-Agent"))
		require.Equal(t, "provider", header.Get("X-Custom"))
		require.Equal(t, "other", header.Get("X-Other"))
	})

	t.Run("user agent from the provider headers", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll, WithHeaders(map[string]string{"user-agent": "from-headers"}))
		_, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.Equal(t, "from-headers", server.last(t).header.Get("User-Agent"))
	})

	t.Run("per call overrides", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(
			t, answerAll,
			WithUserAgent("provider-agent"),
			WithHeaders(map[string]string{"X-Custom": "provider", "X-Other": "other"}),
		)
		call := noulCall()
		call.UserAgent = "call-agent"
		call.Headers = map[string]string{"X-Custom": "call", "X-Extra": "extra"}
		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)

		header := server.last(t).header
		require.Equal(t, "call-agent", header.Get("User-Agent"))
		require.Equal(t, "call", header.Get("X-Custom"))
		require.Equal(t, "other", header.Get("X-Other"))
		require.Equal(t, "extra", header.Get("X-Extra"))
		require.Equal(t, "Bearer test-key", header.Get("Authorization"))
	})

	t.Run("no authorization without an API key", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll, WithAPIKey(""))
		_, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.NotContains(t, server.last(t).header, "Authorization")
	})

	t.Run("a custom authorization header is kept without an API key", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll, WithAPIKey(""), WithHeaders(map[string]string{"Authorization": "Custom abc"}))
		_, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.Equal(t, "Custom abc", server.last(t).header.Get("Authorization"))
	})

	t.Run("images header", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name string
			call fantasy.DecisionCall
			want bool
		}{
			{name: "text only", call: noulCall(), want: false},
			{name: "image", call: imageCall(fantasy.NoulQuestion{Instructions: fantasy.String("Is it a logo?")}), want: true},
			{name: "box question", call: imageCall(BoxQuestion{Instructions: fantasy.String("Find it")}), want: true},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()

				model, server := newTestModel(t, answerAll)
				_, err := model.Decide(t.Context(), tt.call)
				require.NoError(t, err)

				header := server.last(t).header
				if tt.want {
					require.Equal(t, "true", header.Get(imagesHeader))
				} else {
					require.NotContains(t, header, http.CanonicalHeaderKey(imagesHeader))
				}
			})
		}
	})

	t.Run("long context header", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name    string
			options fantasy.ProviderOptions
			want    bool
		}{
			{name: "no options", options: nil, want: false},
			{name: "unset", options: NewDecisionProviderOptions(&DecisionProviderOptions{}), want: false},
			{name: "nil options", options: NewDecisionProviderOptions(nil), want: false},
			{name: "false", options: NewDecisionProviderOptions(&DecisionProviderOptions{LongContext: new(false)}), want: false},
			{name: "true", options: NewDecisionProviderOptions(&DecisionProviderOptions{LongContext: new(true)}), want: true},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()

				model, server := newTestModel(t, answerAll)
				call := noulCall()
				call.ProviderOptions = tt.options
				_, err := model.Decide(t.Context(), call)
				require.NoError(t, err)

				header := server.last(t).header
				if tt.want {
					require.Equal(t, "true", header.Get(longContextHeader))
				} else {
					require.NotContains(t, header, http.CanonicalHeaderKey(longContextHeader))
				}
			})
		}
	})

	t.Run("options of another type are rejected before sending", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		call := noulCall()
		call.ProviderOptions = fantasy.ProviderOptions{Name: &ProviderMetadata{}}
		_, err := model.Decide(t.Context(), call)

		var fantasyErr *fantasy.Error
		require.ErrorAs(t, err, &fantasyErr)
		require.Equal(t, "invalid argument", fantasyErr.Title)
		require.Contains(t, fantasyErr.Message, "DecisionProviderOptions")
		require.Zero(t, server.count())
	})
}
