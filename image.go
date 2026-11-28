package fantasy

import "context"

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
