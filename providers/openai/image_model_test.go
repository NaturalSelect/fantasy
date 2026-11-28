package openai

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"io"
	"mime"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

type imageRequest struct {
	path        string
	contentType string
	userAgent   string
	body        []byte
}

func (r imageRequest) jsonBody(t *testing.T) map[string]any {
	t.Helper()
	var body map[string]any
	require.NoError(t, json.Unmarshal(r.body, &body))
	return body
}

func (r imageRequest) multipartForm(t *testing.T) *multipart.Form {
	t.Helper()
	mediaType, params, err := mime.ParseMediaType(r.contentType)
	require.NoError(t, err)
	require.Equal(t, "multipart/form-data", mediaType)
	form, err := multipart.NewReader(bytes.NewReader(r.body), params["boundary"]).ReadForm(1 << 20)
	require.NoError(t, err)
	return form
}

func newImageTestProvider(t *testing.T, response map[string]any) (fantasy.ImageModelProvider, *imageRequest) {
	t.Helper()

	request := &imageRequest{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		*request = imageRequest{
			path:        r.URL.Path,
			contentType: r.Header.Get("Content-Type"),
			userAgent:   r.Header.Get("User-Agent"),
			body:        body,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)

	p, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL))
	require.NoError(t, err)
	imageProvider, ok := p.(fantasy.ImageModelProvider)
	require.True(t, ok)
	return imageProvider, request
}

func imageResponse(data ...[]byte) map[string]any {
	images := make([]map[string]any, 0, len(data))
	for _, d := range data {
		images = append(images, map[string]any{"b64_json": base64.StdEncoding.EncodeToString(d)})
	}
	return map[string]any{
		"created":       1700000000,
		"data":          images,
		"output_format": "webp",
		"usage": map[string]any{
			"input_tokens":  11,
			"output_tokens": 22,
			"total_tokens":  33,
		},
	}
}

func TestImageModel_Generate(t *testing.T) {
	t.Parallel()

	t.Run("should generate from the prompt and map options and output", func(t *testing.T) {
		t.Parallel()

		imageProvider, request := newImageTestProvider(t, imageResponse([]byte("image-1"), []byte("image-2")))
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)
		require.Equal(t, "gpt-image-1", model.Model())
		require.Equal(t, "openai", model.Provider())

		n := int64(2)
		quality := "low"
		response, err := model.Generate(context.Background(), fantasy.ImageCall{
			Prompt:          "a cat",
			N:               &n,
			Size:            "1024x1024",
			ProviderOptions: NewImageProviderOptions(&ImageProviderOptions{Quality: &quality}),
		})
		require.NoError(t, err)

		require.Equal(t, "/images/generations", request.path)
		body := request.jsonBody(t)
		require.Equal(t, "gpt-image-1", body["model"])
		require.Equal(t, "a cat", body["prompt"])
		require.Equal(t, "1024x1024", body["size"])
		require.EqualValues(t, 2, body["n"])
		require.Equal(t, "low", body["quality"])
		require.NotContains(t, body, "response_format")

		require.Len(t, response.Images, 2)
		require.Equal(t, []byte("image-1"), response.Images[0].Data)
		require.Equal(t, []byte("image-2"), response.Images[1].Data)
		require.Equal(t, "image/webp", response.Images[0].MediaType)
		require.Equal(t, fantasy.Usage{InputTokens: 11, OutputTokens: 22, TotalTokens: 33}, response.Usage)
		require.Empty(t, response.Warnings)
	})

	t.Run("should request base64 output from dall-e models", func(t *testing.T) {
		t.Parallel()

		imageProvider, request := newImageTestProvider(t, imageResponse([]byte("image")))
		model, err := imageProvider.ImageModel(context.Background(), "dall-e-3")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.ImageCall{Prompt: "a cat"})
		require.NoError(t, err)

		require.Equal(t, "b64_json", request.jsonBody(t)["response_format"])
	})

	t.Run("should warn about unsupported settings", func(t *testing.T) {
		t.Parallel()

		imageProvider, _ := newImageTestProvider(t, imageResponse([]byte("image")))
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)

		seed := int64(7)
		response, err := model.Generate(context.Background(), fantasy.ImageCall{
			Prompt:      "a cat",
			AspectRatio: "16:9",
			Seed:        &seed,
			Mask:        &fantasy.FilePart{Data: []byte("mask"), MediaType: "image/png"},
		})
		require.NoError(t, err)

		settings := make([]string, 0, len(response.Warnings))
		for _, warning := range response.Warnings {
			require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, warning.Type)
			settings = append(settings, warning.Setting)
		}
		require.ElementsMatch(t, []string{"aspect_ratio", "seed", "mask"}, settings)
	})

	t.Run("should override the user agent per call", func(t *testing.T) {
		t.Parallel()

		imageProvider, request := newImageTestProvider(t, imageResponse([]byte("image")))
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.ImageCall{Prompt: "a cat", UserAgent: "my-agent/1.0"})
		require.NoError(t, err)

		require.Equal(t, "my-agent/1.0", request.userAgent)
	})

	t.Run("should fail when the prompt is empty", func(t *testing.T) {
		t.Parallel()

		imageProvider, request := newImageTestProvider(t, imageResponse([]byte("image")))
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.ImageCall{})
		require.Error(t, err)
		require.Empty(t, request.path)
	})

	t.Run("should fail when the provider options have the wrong type", func(t *testing.T) {
		t.Parallel()

		imageProvider, _ := newImageTestProvider(t, imageResponse([]byte("image")))
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.ImageCall{
			Prompt:          "a cat",
			ProviderOptions: NewProviderOptions(&ProviderOptions{}),
		})
		require.ErrorContains(t, err, "ImageProviderOptions")
	})

	t.Run("should fail when the response has no image", func(t *testing.T) {
		t.Parallel()

		imageProvider, _ := newImageTestProvider(t, map[string]any{"created": 1700000000, "data": []any{}})
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.ImageCall{Prompt: "a cat"})
		require.ErrorContains(t, err, "no image generated")
	})
}

func TestImageModel_Generate_Edit(t *testing.T) {
	t.Parallel()

	t.Run("should edit with a single reference image and a mask", func(t *testing.T) {
		t.Parallel()

		imageProvider, request := newImageTestProvider(t, imageResponse([]byte("edited")))
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)

		response, err := model.Generate(context.Background(), fantasy.ImageCall{
			Prompt: "add a hat",
			Files:  []fantasy.FilePart{{Data: []byte("cat-bytes"), MediaType: "image/png"}},
			Mask:   &fantasy.FilePart{Data: []byte("mask-bytes"), MediaType: "image/png"},
			Size:   "1024x1024",
		})
		require.NoError(t, err)
		require.Equal(t, []byte("edited"), response.Images[0].Data)

		require.Equal(t, "/images/edits", request.path)
		form := request.multipartForm(t)
		require.Equal(t, []string{"add a hat"}, form.Value["prompt"])
		require.Equal(t, []string{"gpt-image-1"}, form.Value["model"])
		require.Equal(t, []string{"1024x1024"}, form.Value["size"])

		require.Len(t, form.File["image"], 1)
		require.Equal(t, "image.png", form.File["image"][0].Filename)
		require.Equal(t, "image/png", form.File["image"][0].Header.Get("Content-Type"))
		require.Equal(t, []byte("cat-bytes"), readFormFile(t, form.File["image"][0]))

		require.Len(t, form.File["mask"], 1)
		require.Equal(t, []byte("mask-bytes"), readFormFile(t, form.File["mask"][0]))
	})

	t.Run("should edit with several reference images", func(t *testing.T) {
		t.Parallel()

		imageProvider, request := newImageTestProvider(t, imageResponse([]byte("edited")))
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)

		_, err = model.Generate(context.Background(), fantasy.ImageCall{
			Prompt: "combine them",
			Files: []fantasy.FilePart{
				{Filename: "first.jpg", Data: []byte("first"), MediaType: "image/jpeg"},
				{Data: []byte("second"), MediaType: "image/webp"},
			},
		})
		require.NoError(t, err)

		require.Equal(t, "/images/edits", request.path)
		files := request.multipartForm(t).File["image[]"]
		require.Len(t, files, 2)
		require.Equal(t, "first.jpg", files[0].Filename)
		require.Equal(t, "image.webp", files[1].Filename)
		require.Equal(t, []byte("second"), readFormFile(t, files[1]))
	})

	t.Run("should warn about options that only apply to generation", func(t *testing.T) {
		t.Parallel()

		imageProvider, _ := newImageTestProvider(t, imageResponse([]byte("edited")))
		model, err := imageProvider.ImageModel(context.Background(), "gpt-image-1")
		require.NoError(t, err)

		moderation := "low"
		style := "vivid"
		response, err := model.Generate(context.Background(), fantasy.ImageCall{
			Prompt:          "add a hat",
			Files:           []fantasy.FilePart{{Data: []byte("cat"), MediaType: "image/png"}},
			ProviderOptions: NewImageProviderOptions(&ImageProviderOptions{Moderation: &moderation, Style: &style}),
		})
		require.NoError(t, err)

		settings := make([]string, 0, len(response.Warnings))
		for _, warning := range response.Warnings {
			settings = append(settings, warning.Setting)
		}
		require.ElementsMatch(t, []string{"moderation", "style"}, settings)
	})
}

func readFormFile(t *testing.T, header *multipart.FileHeader) []byte {
	t.Helper()
	file, err := header.Open()
	require.NoError(t, err)
	defer file.Close()
	data, err := io.ReadAll(file)
	require.NoError(t, err)
	return data
}
