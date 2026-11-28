package typesafe

import (
	"encoding/json"
	"net/http"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestProvider(t *testing.T) {
	t.Parallel()

	t.Run("language models are not supported", func(t *testing.T) {
		t.Parallel()

		p, err := New()
		require.NoError(t, err)

		model, err := p.LanguageModel(t.Context(), "jev-latest")
		require.Nil(t, model)
		var fantasyErr *fantasy.Error
		require.ErrorAs(t, err, &fantasyErr)
		require.Equal(t, "unsupported", fantasyErr.Title)
		require.Contains(t, fantasyErr.Message, "use DecisionModel")
	})

	t.Run("name defaults and honors WithName", func(t *testing.T) {
		t.Parallel()

		p, err := New()
		require.NoError(t, err)
		require.Equal(t, "typesafe", p.Name())

		p, err = New(WithName("custom"))
		require.NoError(t, err)
		require.Equal(t, "custom", p.Name())

		model, err := p.(fantasy.DecisionModelProvider).DecisionModel(t.Context(), "")
		require.NoError(t, err)
		require.Equal(t, "custom", model.Provider())
	})

	t.Run("empty model id selects the default model", func(t *testing.T) {
		t.Parallel()

		provider, server := newTestProvider(t, answerAll)
		model, err := provider.DecisionModel(t.Context(), "")
		require.NoError(t, err)
		require.Equal(t, "jev-latest", model.Model())

		_, err = model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.JSONEq(t, `"jev-latest"`, string(server.last(t).fields(t)["model"]))
	})

	t.Run("explicit model id is sent", func(t *testing.T) {
		t.Parallel()

		provider, server := newTestProvider(t, answerAll)
		model, err := provider.DecisionModel(t.Context(), "jev-1.13.0")
		require.NoError(t, err)
		require.Equal(t, "jev-1.13.0", model.Model())

		_, err = model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.JSONEq(t, `"jev-1.13.0"`, string(server.last(t).fields(t)["model"]))
	})

	t.Run("trailing slash of the base URL is ignored", func(t *testing.T) {
		t.Parallel()

		_, server := newTestProvider(t, answerAll)
		p, err := New(WithAPIKey("test-key"), WithBaseURL(server.url+"/"))
		require.NoError(t, err)

		model, err := p.(fantasy.DecisionModelProvider).DecisionModel(t.Context(), "")
		require.NoError(t, err)
		_, err = model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.Equal(t, "/v1/systemone", server.last(t).path)
	})

	t.Run("uses the configured HTTP client", func(t *testing.T) {
		t.Parallel()

		client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
			r.Header.Set("X-Via-Client", "yes")
			return http.DefaultTransport.RoundTrip(r)
		})}
		model, server := newTestModel(t, answerAll, WithHTTPClient(client))

		_, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.Equal(t, "yes", server.last(t).header.Get("X-Via-Client"))
	})
}

func TestProviderOptionsJSON(t *testing.T) {
	t.Parallel()

	t.Run("decision options round trip", func(t *testing.T) {
		t.Parallel()

		options := NewDecisionProviderOptions(&DecisionProviderOptions{LongContext: new(true)})
		data, err := json.Marshal(options)
		require.NoError(t, err)
		require.JSONEq(t, `{"typesafe":{"type":"typesafe.decision_options","data":{"long_context":true}}}`, string(data))

		var raw map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(data, &raw))
		decoded, err := fantasy.UnmarshalProviderOptions(raw)
		require.NoError(t, err)
		require.Equal(t, options, decoded)
	})

	t.Run("unset long context is omitted", func(t *testing.T) {
		t.Parallel()

		data, err := json.Marshal(NewDecisionProviderOptions(&DecisionProviderOptions{}))
		require.NoError(t, err)
		require.JSONEq(t, `{"typesafe":{"type":"typesafe.decision_options","data":{}}}`, string(data))
	})

	t.Run("metadata round trip", func(t *testing.T) {
		t.Parallel()

		metadata := fantasy.ProviderMetadata{Name: &ProviderMetadata{RequestID: "req_123"}}
		data, err := json.Marshal(metadata)
		require.NoError(t, err)
		require.JSONEq(t, `{"typesafe":{"type":"typesafe.metadata","data":{"request_id":"req_123"}}}`, string(data))

		var raw map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(data, &raw))
		decoded, err := fantasy.UnmarshalProviderMetadata(raw)
		require.NoError(t, err)
		require.Equal(t, metadata, decoded)
	})
}
