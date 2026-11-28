package fantasy

import (
	"context"
	"fmt"
)

// ImageCall represents a call to an image generation model.
type ImageCall struct {
	Prompt string `json:"prompt"`
	// Files are reference images, or the images to edit.
	Files []FilePart `json:"files"`
	// Mask marks the region of the first file to edit, where supported.
	Mask *FilePart `json:"mask"`
	// N is the number of images to generate. Providers generate one image when it is nil.
	N *int64 `json:"n"`
	// Size is the output size as "WIDTHxHEIGHT", e.g. "1024x1024".
	Size string `json:"size"`
	// AspectRatio is the output aspect ratio, e.g. "16:9".
	AspectRatio string `json:"aspect_ratio"`
	Seed        *int64 `json:"seed"`

	// UserAgent overrides the provider-level User-Agent header for this call.
	UserAgent string `json:"-"`

	// Headers overrides matching provider-level headers for this call.
	Headers map[string]string `json:"-"`

	// for provider specific options, the key is the provider id
	ProviderOptions ProviderOptions `json:"provider_options"`
}

// Validate checks the provider-independent structure of the call. Limits that
// differ between providers and models are left to the provider.
func (c ImageCall) Validate() error {
	if c.N != nil && *c.N <= 0 {
		return invalidArgument("n must be greater than zero, got %d", *c.N)
	}
	for i, file := range c.Files {
		if len(file.Data) == 0 {
			return invalidArgument("file %d has no data", i)
		}
	}
	if c.Mask != nil {
		if len(c.Files) == 0 {
			return invalidArgument("a mask needs a file to edit")
		}
		if len(c.Mask.Data) == 0 {
			return invalidArgument("mask has no data")
		}
	}
	return nil
}

func invalidArgument(format string, args ...any) error {
	return &Error{Title: "invalid argument", Message: fmt.Sprintf(format, args...)}
}

// ImageResponse represents a response from an image generation model.
type ImageResponse struct {
	Images   []FileContent `json:"images"`
	Warnings []CallWarning `json:"warnings"`
	Usage    Usage         `json:"usage"`

	// for provider specific response metadata, the key is the provider id
	ProviderMetadata ProviderMetadata `json:"provider_metadata"`
}

// ImageModel represents a model that can generate images.
type ImageModel interface {
	Generate(context.Context, ImageCall) (*ImageResponse, error)

	Provider() string
	Model() string
}

// ImageModelProvider is implemented by providers that offer image generation models.
// Check for it with a type assertion on a Provider.
type ImageModelProvider interface {
	ImageModel(ctx context.Context, modelID string) (ImageModel, error)
}
