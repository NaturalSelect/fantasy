// Package typesafe provides an implementation of the fantasy decision API for
// TypeSafe's System One models (Jev).
//
// TypeSafe does not generate text. It answers typed questions about a shared
// input with probabilities and confidence, so the provider implements
// [fantasy.DecisionModelProvider] and not language models. Unlike the TypeSafe
// SDKs, it never reads TYPESAFE_API_KEY; pass the key with [WithAPIKey].
package typesafe

import (
	"cmp"
	"context"
	"maps"
	"net/http"

	"charm.land/fantasy"
)

const (
	// Name is the name of the TypeSafe provider.
	Name = "typesafe"
	// DefaultURL is the default URL for the TypeSafe API.
	DefaultURL = "https://api.typesafe.ai"
	// DefaultModel is the model used when none is given. It is an alias that
	// follows the latest Jev release.
	DefaultModel = "jev-latest"
)

type provider struct {
	options options
}

type options struct {
	apiKey    string
	baseURL   string
	name      string
	headers   map[string]string
	userAgent string
	client    *http.Client
}

// Option defines a function that configures TypeSafe provider options.
type Option = func(*options)

var _ fantasy.DecisionModelProvider = (*provider)(nil)

// New creates a new TypeSafe provider with the given options.
func New(opts ...Option) (fantasy.Provider, error) {
	providerOptions := options{
		headers: map[string]string{},
	}
	for _, o := range opts {
		o(&providerOptions)
	}

	providerOptions.baseURL = cmp.Or(providerOptions.baseURL, DefaultURL)
	providerOptions.name = cmp.Or(providerOptions.name, Name)
	providerOptions.client = cmp.Or(providerOptions.client, http.DefaultClient)

	return &provider{options: providerOptions}, nil
}

// WithAPIKey sets the API key for the TypeSafe provider.
func WithAPIKey(apiKey string) Option {
	return func(o *options) {
		o.apiKey = apiKey
	}
}

// WithBaseURL sets the base URL for the TypeSafe provider.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

// WithName sets the name for the TypeSafe provider.
func WithName(name string) Option {
	return func(o *options) {
		o.name = name
	}
}

// WithHeaders sets the headers for the TypeSafe provider.
func WithHeaders(headers map[string]string) Option {
	return func(o *options) {
		maps.Copy(o.headers, headers)
	}
}

// WithHTTPClient sets the HTTP client for the TypeSafe provider.
func WithHTTPClient(client *http.Client) Option {
	return func(o *options) {
		o.client = client
	}
}

// WithUserAgent sets an explicit User-Agent header, overriding the default and any
// value set via WithHeaders.
func WithUserAgent(ua string) Option {
	return func(o *options) {
		o.userAgent = ua
	}
}

// Name implements fantasy.Provider.
func (p *provider) Name() string {
	return p.options.name
}

// LanguageModel implements fantasy.Provider. TypeSafe has no language models,
// so it always returns an error.
func (p *provider) LanguageModel(context.Context, string) (fantasy.LanguageModel, error) {
	return nil, &fantasy.Error{Title: "unsupported", Message: "typesafe offers decision models only; use DecisionModel"}
}

// DecisionModel implements fantasy.DecisionModelProvider. An empty modelID
// selects DefaultModel.
func (p *provider) DecisionModel(_ context.Context, modelID string) (fantasy.DecisionModel, error) {
	return &decisionModel{
		provider: p.options.name,
		modelID:  cmp.Or(modelID, DefaultModel),
		options:  p.options,
	}, nil
}
