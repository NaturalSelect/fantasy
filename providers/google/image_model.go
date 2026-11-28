package google

import (
	"cmp"
	"context"
	"fmt"
	"math"
	"net/http"
	"strings"

	"charm.land/fantasy"
	"google.golang.org/genai"
)

type imageModel struct {
	modelID  string
	provider string
	client   *genai.Client
}

var _ fantasy.ImageModelProvider = (*provider)(nil)

// ImageModel implements fantasy.ImageModelProvider. Imagen models and Gemini
// image models are both supported, and told apart by their model ID.
func (a *provider) ImageModel(ctx context.Context, modelID string) (fantasy.ImageModel, error) {
	if modelID == "" {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "image model ID is empty"}
	}
	client, err := a.newClient(ctx)
	if err != nil {
		return nil, err
	}
	return &imageModel{
		modelID:  modelID,
		provider: a.options.name,
		client:   client,
	}, nil
}

// Model implements fantasy.ImageModel.
func (m *imageModel) Model() string {
	return m.modelID
}

// Provider implements fantasy.ImageModel.
func (m *imageModel) Provider() string {
	return m.provider
}

func modelName(id string) string {
	return id[strings.LastIndexByte(id, '/')+1:]
}

func (m *imageModel) isImagen() bool {
	return strings.HasPrefix(modelName(m.modelID), "imagen-")
}

// Generate implements fantasy.ImageModel.
func (m *imageModel) Generate(ctx context.Context, call fantasy.ImageCall) (*fantasy.ImageResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, err
	}
	if call.Seed != nil && (*call.Seed < math.MinInt32 || *call.Seed > math.MaxInt32) {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "seed must fit in a signed 32-bit integer"}
	}
	if call.Prompt == "" {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "image prompt is empty"}
	}
	options, err := imageProviderOptions(call)
	if err != nil {
		return nil, err
	}
	if m.client.ClientConfig().Backend != genai.BackendVertexAI && (options.PersonGeneration != nil || options.OutputMIMEType != nil) {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "person_generation and output_mime_type are only supported on Vertex AI"}
	}
	ctx = withImageCallUA(ctx, call)

	if m.isImagen() {
		return m.generateWithImagen(ctx, call, options)
	}
	return m.generateWithGemini(ctx, call, options)
}

func imageProviderOptions(call fantasy.ImageCall) (*ImageProviderOptions, error) {
	v, ok := call.ProviderOptions[Name]
	if !ok {
		return &ImageProviderOptions{}, nil
	}
	options, ok := v.(*ImageProviderOptions)
	if !ok {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "google provider options should be *google.ImageProviderOptions"}
	}
	return cmp.Or(options, &ImageProviderOptions{}), nil
}

func unsupportedSettingWarning(setting, message string) fantasy.CallWarning {
	return fantasy.CallWarning{
		Type:    fantasy.CallWarningTypeUnsupportedSetting,
		Setting: setting,
		Message: message,
	}
}

func (m *imageModel) generateWithImagen(ctx context.Context, call fantasy.ImageCall, opts *ImageProviderOptions) (*fantasy.ImageResponse, error) {
	// NOTE: The genai SDK only implements GenerateImages for the Vertex backend.
	if m.client.ClientConfig().Backend != genai.BackendVertexAI {
		return nil, &fantasy.Error{Title: "unsupported model", Message: "imagen models need the Vertex AI backend (google.WithVertex); use a gemini image model with an API key"}
	}
	if len(call.Files) > 0 {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "imagen models do not accept reference images; use a gemini image model"}
	}

	var warnings []fantasy.CallWarning
	if call.Size != "" {
		warnings = append(warnings, unsupportedSettingWarning("size", "use aspect_ratio and the image_size option instead"))
	}

	// NOTE: The SDK defaults to four images when NumberOfImages is unset.
	config := &genai.GenerateImagesConfig{
		NumberOfImages:   1,
		AspectRatio:      call.AspectRatio,
		IncludeRAIReason: true,
	}
	if call.N != nil {
		if *call.N > math.MaxInt32 {
			return nil, &fantasy.Error{Title: "invalid argument", Message: "n must fit in a signed 32-bit integer"}
		}
		config.NumberOfImages = int32(*call.N)
	}
	if call.Seed != nil {
		config.Seed = new(int32(*call.Seed))
		// NOTE: Imagen requires watermarking off for a seed. The SDK omits a false AddWatermark.
		config.HTTPOptions = &genai.HTTPOptions{
			ExtraBody: map[string]any{"parameters": map[string]any{"addWatermark": false}},
		}
	}
	if opts.ImageSize != nil {
		config.ImageSize = *opts.ImageSize
	}
	if opts.PersonGeneration != nil {
		config.PersonGeneration = genai.PersonGeneration(*opts.PersonGeneration)
	}
	if opts.OutputMIMEType != nil {
		config.OutputMIMEType = *opts.OutputMIMEType
	}
	if opts.NegativePrompt != nil {
		config.NegativePrompt = *opts.NegativePrompt
	}
	if opts.GuidanceScale != nil {
		config.GuidanceScale = new(float32(*opts.GuidanceScale))
	}
	if opts.SafetyFilterLevel != nil {
		config.SafetyFilterLevel = genai.SafetyFilterLevel(*opts.SafetyFilterLevel)
	}

	response, err := m.client.Models.GenerateImages(ctx, m.modelID, call.Prompt, config)
	if err != nil {
		return nil, toProviderErr(err)
	}

	return imagenImageResponse(response, config.NumberOfImages, warnings)
}

func imagenImageResponse(response *genai.GenerateImagesResponse, requested int32, warnings []fantasy.CallWarning) (*fantasy.ImageResponse, error) {
	if response == nil {
		return nil, &fantasy.Error{Title: "invalid response", Message: "missing Imagen response"}
	}
	var reasons []string
	images := make([]fantasy.FileContent, 0, len(response.GeneratedImages))
	for _, generated := range response.GeneratedImages {
		if generated == nil {
			return nil, &fantasy.Error{Title: "invalid response", Message: "null generated image"}
		}
		if generated.Image == nil || len(generated.Image.ImageBytes) == 0 {
			message := "an image contained no data"
			if generated.RAIFilteredReason != "" {
				reasons = append(reasons, generated.RAIFilteredReason)
				message = "an image was filtered: " + generated.RAIFilteredReason
			}
			warnings = append(warnings, fantasy.CallWarning{Type: fantasy.CallWarningTypeOther, Message: message})
			continue
		}
		images = append(images, generatedImageFile(generated.Image.ImageBytes, generated.Image.MIMEType))
	}
	if len(images) == 0 {
		message := "the response contained no image data"
		if len(reasons) > 0 {
			message += ": " + strings.Join(reasons, "; ")
		}
		return nil, &fantasy.Error{Title: "no image generated", Message: message}
	}
	if len(response.GeneratedImages) < int(requested) {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeOther,
			Message: fmt.Sprintf("requested %d images but received only %d results", requested, len(response.GeneratedImages)),
		})
	}

	return &fantasy.ImageResponse{Images: images, Warnings: warnings}, nil
}

func generatedImageFile(data []byte, mediaType string) fantasy.FileContent {
	if mediaType == "" {
		mediaType = http.DetectContentType(data)
	}
	return fantasy.FileContent{MediaType: mediaType, Data: data}
}

func (m *imageModel) generateWithGemini(ctx context.Context, call fantasy.ImageCall, opts *ImageProviderOptions) (*fantasy.ImageResponse, error) {
	var warnings []fantasy.CallWarning
	if call.N != nil && *call.N > 1 {
		warnings = append(warnings, unsupportedSettingWarning("n", "gemini image models generate one image per call"))
	}
	if call.Size != "" {
		warnings = append(warnings, unsupportedSettingWarning("size", "use aspect_ratio and the image_size option instead"))
	}
	if call.Mask != nil {
		warnings = append(warnings, unsupportedSettingWarning("mask", ""))
	}
	for _, option := range []struct {
		setting string
		isSet   bool
	}{
		{"negative_prompt", opts.NegativePrompt != nil},
		{"guidance_scale", opts.GuidanceScale != nil},
		{"safety_filter_level", opts.SafetyFilterLevel != nil},
	} {
		if option.isSet {
			warnings = append(warnings, unsupportedSettingWarning(option.setting, "only supported by imagen models"))
		}
	}

	parts := make([]*genai.Part, 0, len(call.Files)+1)
	for _, file := range call.Files {
		parts = append(parts, &genai.Part{InlineData: &genai.Blob{Data: file.Data, MIMEType: file.MediaType}})
	}
	parts = append(parts, &genai.Part{Text: call.Prompt})
	contents := []*genai.Content{{Role: genai.RoleUser, Parts: parts}}

	imageConfig := &genai.ImageConfig{AspectRatio: call.AspectRatio}
	if opts.ImageSize != nil {
		imageConfig.ImageSize = *opts.ImageSize
	}
	if opts.PersonGeneration != nil {
		imageConfig.PersonGeneration = *opts.PersonGeneration
	}
	if opts.OutputMIMEType != nil {
		imageConfig.OutputMIMEType = *opts.OutputMIMEType
	}
	config := &genai.GenerateContentConfig{
		ResponseModalities: []string{string(genai.ModalityText), string(genai.ModalityImage)},
		ImageConfig:        imageConfig,
	}
	if call.Seed != nil {
		config.Seed = new(int32(*call.Seed))
	}

	response, err := m.client.Models.GenerateContent(ctx, m.modelID, contents, config)
	if err != nil {
		return nil, toProviderErr(err)
	}

	if response == nil {
		return nil, &fantasy.Error{Title: "invalid response", Message: "missing Gemini image response"}
	}
	var images []fantasy.FileContent
	var text strings.Builder
	for _, candidate := range response.Candidates {
		// NOTE: The Developer API converter turns null into an empty candidate. A filtered candidate can lack content but has a finish reason.
		if candidate == nil || (candidate.Content == nil && candidate.FinishReason == "") {
			return nil, &fantasy.Error{Title: "invalid response", Message: "missing image candidate"}
		}
		if candidate.Content == nil {
			continue
		}
		for _, part := range candidate.Content.Parts {
			if part == nil {
				return nil, &fantasy.Error{Title: "invalid response", Message: "null image part"}
			}
			switch {
			case part.Thought:
				// NOTE: Thought parts can carry draft images that are not the result.
			case part.InlineData != nil && len(part.InlineData.Data) > 0:
				images = append(images, generatedImageFile(part.InlineData.Data, part.InlineData.MIMEType))
			case part.Text != "":
				text.WriteString(part.Text)
			}
		}
	}
	if len(images) == 0 {
		message := "the response contained no image data"
		if text.Len() > 0 {
			message += ": " + text.String()
		}
		return nil, &fantasy.Error{Title: "no image generated", Message: message}
	}

	imageResponse := &fantasy.ImageResponse{Images: images, Warnings: warnings}
	if response.UsageMetadata != nil {
		imageResponse.Usage = languageModel{providerOptions: options{backend: m.client.ClientConfig().Backend}}.mapUsage(response.UsageMetadata)
	}
	return imageResponse, nil
}
