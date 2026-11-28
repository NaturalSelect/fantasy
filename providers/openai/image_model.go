package openai

import (
	"bytes"
	"cmp"
	"context"
	"encoding/base64"
	"io"
	"strings"

	"charm.land/fantasy"
	"github.com/charmbracelet/openai-go"
	"github.com/charmbracelet/openai-go/packages/param"
)

type imageModel struct {
	modelID  string
	provider string
	client   openai.Client
}

var _ fantasy.ImageModelProvider = (*provider)(nil)

func newImageModel(modelID, provider string, client openai.Client) imageModel {
	return imageModel{modelID: modelID, provider: provider, client: client}
}

// Model implements fantasy.ImageModel.
func (m imageModel) Model() string {
	return m.modelID
}

// Provider implements fantasy.ImageModel.
func (m imageModel) Provider() string {
	return m.provider
}

// Generate implements fantasy.ImageModel. It edits the reference images when
// the call has any, and generates from the prompt alone otherwise.
func (m imageModel) Generate(ctx context.Context, call fantasy.ImageCall) (*fantasy.ImageResponse, error) {
	if call.Prompt == "" {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "image prompt is empty"}
	}
	options, err := imageProviderOptions(call)
	if err != nil {
		return nil, err
	}
	warnings := imageCallWarnings(call, options)
	requestOptions := imageCallRequestOptions(call)

	var response *openai.ImagesResponse
	if len(call.Files) == 0 {
		response, err = m.client.Images.Generate(ctx, m.generateParams(call, options), requestOptions...)
	} else {
		response, err = m.client.Images.Edit(ctx, m.editParams(call, options), requestOptions...)
	}
	if err != nil {
		return nil, toProviderErr(err)
	}

	return toImageResponse(response, warnings)
}

func imageProviderOptions(call fantasy.ImageCall) (*ImageProviderOptions, error) {
	v, ok := call.ProviderOptions[Name]
	if !ok {
		return &ImageProviderOptions{}, nil
	}
	options, ok := v.(*ImageProviderOptions)
	if !ok {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "openai provider options should be *openai.ImageProviderOptions"}
	}
	return options, nil
}

func imageCallWarnings(call fantasy.ImageCall, options *ImageProviderOptions) []fantasy.CallWarning {
	var warnings []fantasy.CallWarning
	if call.AspectRatio != "" {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "aspect_ratio",
			Message: "use size instead",
		})
	}
	if call.Seed != nil {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "seed",
		})
	}
	if call.Mask != nil && len(call.Files) == 0 {
		warnings = append(warnings, fantasy.CallWarning{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "mask",
			Message: "a mask needs a reference image to edit",
		})
	}
	if len(call.Files) > 0 {
		if options.Moderation != nil {
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "moderation",
				Message: "not supported when editing images",
			})
		}
		if options.Style != nil {
			warnings = append(warnings, fantasy.CallWarning{
				Type:    fantasy.CallWarningTypeUnsupportedSetting,
				Setting: "style",
				Message: "not supported when editing images",
			})
		}
	}
	return warnings
}

// isDallE reports whether the model returns image URLs unless asked for base64.
func (m imageModel) isDallE() bool {
	return strings.HasPrefix(m.modelID, "dall-e")
}

func (m imageModel) generateParams(call fantasy.ImageCall, options *ImageProviderOptions) openai.ImageGenerateParams {
	params := openai.ImageGenerateParams{
		Model:  m.modelID,
		Prompt: call.Prompt,
		Size:   openai.ImageGenerateParamsSize(call.Size),
	}
	if call.N != nil {
		params.N = param.NewOpt(*call.N)
	}
	if m.isDallE() {
		params.ResponseFormat = openai.ImageGenerateParamsResponseFormatB64JSON
	}
	if options.Quality != nil {
		params.Quality = openai.ImageGenerateParamsQuality(*options.Quality)
	}
	if options.Background != nil {
		params.Background = openai.ImageGenerateParamsBackground(*options.Background)
	}
	if options.OutputFormat != nil {
		params.OutputFormat = openai.ImageGenerateParamsOutputFormat(*options.OutputFormat)
	}
	if options.OutputCompression != nil {
		params.OutputCompression = param.NewOpt(*options.OutputCompression)
	}
	if options.Moderation != nil {
		params.Moderation = openai.ImageGenerateParamsModeration(*options.Moderation)
	}
	if options.Style != nil {
		params.Style = openai.ImageGenerateParamsStyle(*options.Style)
	}
	if options.User != nil {
		params.User = param.NewOpt(*options.User)
	}
	return params
}

func (m imageModel) editParams(call fantasy.ImageCall, options *ImageProviderOptions) openai.ImageEditParams {
	params := openai.ImageEditParams{
		Model:  m.modelID,
		Prompt: call.Prompt,
		Size:   openai.ImageEditParamsSize(call.Size),
	}
	if len(call.Files) == 1 {
		params.Image.OfFile = toImageUpload(call.Files[0])
	} else {
		readers := make([]io.Reader, 0, len(call.Files))
		for _, file := range call.Files {
			readers = append(readers, toImageUpload(file))
		}
		params.Image.OfFileArray = readers
	}
	if call.Mask != nil {
		params.Mask = toImageUpload(*call.Mask)
	}
	if call.N != nil {
		params.N = param.NewOpt(*call.N)
	}
	if m.isDallE() {
		params.ResponseFormat = openai.ImageEditParamsResponseFormatB64JSON
	}
	if options.Quality != nil {
		params.Quality = openai.ImageEditParamsQuality(*options.Quality)
	}
	if options.Background != nil {
		params.Background = openai.ImageEditParamsBackground(*options.Background)
	}
	if options.OutputFormat != nil {
		params.OutputFormat = openai.ImageEditParamsOutputFormat(*options.OutputFormat)
	}
	if options.OutputCompression != nil {
		params.OutputCompression = param.NewOpt(*options.OutputCompression)
	}
	if options.User != nil {
		params.User = param.NewOpt(*options.User)
	}
	return params
}

// toImageUpload turns a file into a multipart upload. The API reads the image
// type from the file name, so a missing name is derived from the media type.
func toImageUpload(file fantasy.FilePart) io.Reader {
	filename := file.Filename
	if filename == "" {
		filename = "image"
		if subtype, ok := strings.CutPrefix(file.MediaType, "image/"); ok {
			filename += "." + subtype
		}
	}
	return openai.File(bytes.NewReader(file.Data), filename, file.MediaType)
}

func toImageResponse(response *openai.ImagesResponse, warnings []fantasy.CallWarning) (*fantasy.ImageResponse, error) {
	mediaType := "image/" + cmp.Or(string(response.OutputFormat), "png")

	images := make([]fantasy.FileContent, 0, len(response.Data))
	for _, image := range response.Data {
		if image.B64JSON == "" {
			continue
		}
		data, err := base64.StdEncoding.DecodeString(image.B64JSON)
		if err != nil {
			return nil, &fantasy.Error{Title: "invalid response", Message: "generated image is not valid base64", Cause: err}
		}
		images = append(images, fantasy.FileContent{MediaType: mediaType, Data: data})
	}
	if len(images) == 0 {
		return nil, &fantasy.Error{Title: "no image generated", Message: "the response contained no image data"}
	}

	return &fantasy.ImageResponse{
		Images:   images,
		Warnings: warnings,
		Usage: fantasy.Usage{
			InputTokens:  response.Usage.InputTokens,
			OutputTokens: response.Usage.OutputTokens,
			TotalTokens:  response.Usage.TotalTokens,
		},
	}, nil
}
