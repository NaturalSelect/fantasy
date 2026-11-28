package typesafe

import (
	"encoding/json"

	"charm.land/fantasy"
)

// Global type identifiers for TypeSafe-specific provider data.
const (
	TypeProviderDecisionOptions = Name + ".decision_options"
	TypeProviderMetadata        = Name + ".metadata"
)

// Register TypeSafe provider-specific types with the global registry.
func init() {
	fantasy.RegisterProviderType(TypeProviderDecisionOptions, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v DecisionProviderOptions
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
	fantasy.RegisterProviderType(TypeProviderMetadata, func(data []byte) (fantasy.ProviderOptionsData, error) {
		var v ProviderMetadata
		if err := json.Unmarshal(data, &v); err != nil {
			return nil, err
		}
		return &v, nil
	})
}

// DecisionProviderOptions represents additional options for TypeSafe decision calls.
type DecisionProviderOptions struct {
	// LongContext opts in to long inputs by sending the
	// TypeSafe-Danger-Zone-Long-Context header.
	LongContext *bool `json:"long_context,omitempty"`
}

// Options implements the ProviderOptions interface.
func (*DecisionProviderOptions) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for DecisionProviderOptions.
func (o DecisionProviderOptions) MarshalJSON() ([]byte, error) {
	type plain DecisionProviderOptions
	return fantasy.MarshalProviderType(TypeProviderDecisionOptions, plain(o))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for DecisionProviderOptions.
func (o *DecisionProviderOptions) UnmarshalJSON(data []byte) error {
	type plain DecisionProviderOptions
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*o = DecisionProviderOptions(p)
	return nil
}

// NewDecisionProviderOptions creates new decision options for TypeSafe.
func NewDecisionProviderOptions(opts *DecisionProviderOptions) fantasy.ProviderOptions {
	return fantasy.ProviderOptions{
		Name: opts,
	}
}

// ProviderMetadata represents additional metadata from the TypeSafe provider.
// It is found in the response under Name.
type ProviderMetadata struct {
	// RequestID identifies the request for TypeSafe support. It is empty
	// when the response had no request ID.
	RequestID string `json:"request_id"`
}

// Options implements the ProviderOptions interface.
func (*ProviderMetadata) Options() {}

// MarshalJSON implements custom JSON marshaling with type info for ProviderMetadata.
func (m ProviderMetadata) MarshalJSON() ([]byte, error) {
	type plain ProviderMetadata
	return fantasy.MarshalProviderType(TypeProviderMetadata, plain(m))
}

// UnmarshalJSON implements custom JSON unmarshaling with type info for ProviderMetadata.
func (m *ProviderMetadata) UnmarshalJSON(data []byte) error {
	type plain ProviderMetadata
	var p plain
	if err := fantasy.UnmarshalProviderType(data, &p); err != nil {
		return err
	}
	*m = ProviderMetadata(p)
	return nil
}
