package google

import (
	"context"
	"strings"

	"charm.land/fantasy"
	"google.golang.org/genai"
)

type imageModel struct {
	modelID         string
	provider        string
	client          *genai.Client
	providerOptions options
}

var _ fantasy.ImageModelProvider = (*provider)(nil)

// ImageModel implements fantasy.ImageModelProvider. Imagen models and Gemini
// image models are both supported, and told apart by their model ID.
func (a *provider) ImageModel(ctx context.Context, modelID string) (fantasy.ImageModel, error) {
	client, err := a.newClient(ctx)
	if err != nil {
		return nil, err
	}
	return &imageModel{
		modelID:         modelID,
		provider:        a.options.name,
		client:          client,
		providerOptions: a.options,
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

func (m *imageModel) isImagen() bool {
	return strings.Contains(m.modelID, "imagen")
}

// Generate implements fantasy.ImageModel.
func (m *imageModel) Generate(ctx context.Context, call fantasy.ImageCall) (*fantasy.ImageResponse, error) {
	if call.Prompt == "" {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "image prompt is empty"}
	}
	options, err := imageProviderOptions(call)
	if err != nil {
		return nil, err
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
	return options, nil
}

func unsupportedSettingWarning(setting, message string) fantasy.CallWarning {
	return fantasy.CallWarning{
		Type:    fantasy.CallWarningTypeUnsupportedSetting,
		Setting: setting,
		Message: message,
	}
}

func (m *imageModel) generateWithImagen(ctx context.Context, call fantasy.ImageCall, options *ImageProviderOptions) (*fantasy.ImageResponse, error) {
	// NOTE: The genai SDK only implements GenerateImages for the Vertex backend.
	if m.providerOptions.backend != genai.BackendVertexAI {
		return nil, &fantasy.Error{Title: "unsupported model", Message: "imagen models need the Vertex AI backend (google.WithVertex); use a gemini image model with an API key"}
	}
	if len(call.Files) > 0 {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "imagen models do not accept reference images; use a gemini image model"}
	}

	var warnings []fantasy.CallWarning
	if call.Size != "" {
		warnings = append(warnings, unsupportedSettingWarning("size", "use aspect_ratio and the image_size option instead"))
	}
	if call.Mask != nil {
		warnings = append(warnings, unsupportedSettingWarning("mask", ""))
	}

	// NOTE: The SDK defaults to four images when NumberOfImages is unset.
	config := &genai.GenerateImagesConfig{
		NumberOfImages:   1,
		AspectRatio:      call.AspectRatio,
		IncludeRAIReason: true,
	}
	if call.N != nil {
		config.NumberOfImages = int32(*call.N)
	}
	if call.Seed != nil {
		config.Seed = new(int32(*call.Seed))
	}
	if options.ImageSize != nil {
		config.ImageSize = *options.ImageSize
	}
	if options.PersonGeneration != nil {
		config.PersonGeneration = genai.PersonGeneration(*options.PersonGeneration)
	}
	if options.OutputMIMEType != nil {
		config.OutputMIMEType = *options.OutputMIMEType
	}
	if options.NegativePrompt != nil {
		config.NegativePrompt = *options.NegativePrompt
	}
	if options.GuidanceScale != nil {
		config.GuidanceScale = new(float32(*options.GuidanceScale))
	}
	if options.SafetyFilterLevel != nil {
		config.SafetyFilterLevel = genai.SafetyFilterLevel(*options.SafetyFilterLevel)
	}

	response, err := m.client.Models.GenerateImages(ctx, m.modelID, call.Prompt, config)
	if err != nil {
		return nil, toProviderErr(err)
	}

	images := make([]fantasy.FileContent, 0, len(response.GeneratedImages))
	for _, generated := range response.GeneratedImages {
		if generated.Image == nil || len(generated.Image.ImageBytes) == 0 {
			if generated.RAIFilteredReason != "" {
				warnings = append(warnings, fantasy.CallWarning{
					Type:    fantasy.CallWarningTypeOther,
					Message: "an image was filtered: " + generated.RAIFilteredReason,
				})
			}
			continue
		}
		images = append(images, fantasy.FileContent{
			MediaType: generated.Image.MIMEType,
			Data:      generated.Image.ImageBytes,
		})
	}
	if len(images) == 0 {
		return nil, &fantasy.Error{Title: "no image generated", Message: "the response contained no image data"}
	}

	return &fantasy.ImageResponse{Images: images, Warnings: warnings}, nil
}

func (m *imageModel) generateWithGemini(ctx context.Context, call fantasy.ImageCall, options *ImageProviderOptions) (*fantasy.ImageResponse, error) {
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
		{"negative_prompt", options.NegativePrompt != nil},
		{"guidance_scale", options.GuidanceScale != nil},
		{"safety_filter_level", options.SafetyFilterLevel != nil},
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
	if options.ImageSize != nil {
		imageConfig.ImageSize = *options.ImageSize
	}
	if options.PersonGeneration != nil {
		imageConfig.PersonGeneration = *options.PersonGeneration
	}
	if options.OutputMIMEType != nil {
		imageConfig.OutputMIMEType = *options.OutputMIMEType
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

	var images []fantasy.FileContent
	var text strings.Builder
	for _, candidate := range response.Candidates {
		if candidate.Content == nil {
			continue
		}
		for _, part := range candidate.Content.Parts {
			switch {
			case part.Thought:
				// NOTE: Thought parts can carry draft images that are not the result.
			case part.InlineData != nil && len(part.InlineData.Data) > 0:
				images = append(images, fantasy.FileContent{MediaType: part.InlineData.MIMEType, Data: part.InlineData.Data})
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
		imageResponse.Usage = languageModel{providerOptions: m.providerOptions}.mapUsage(response.UsageMetadata)
	}
	return imageResponse, nil
}
