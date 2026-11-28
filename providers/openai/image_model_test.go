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

func newImageTestProvider(t *testing.T, response any) (fantasy.ImageModelProvider, *imageRequest) {
	t.Helper()

	data, err := json.Marshal(response)
	require.NoError(t, err)
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
		_, _ = w.Write(data)
	}))
	t.Cleanup(server.Close)

	p, err := New(WithAPIKey("test-api-key"), WithBaseURL(server.URL))
	require.NoError(t, err)
	imageProvider, ok := p.(fantasy.ImageModelProvider)
	require.True(t, ok)
	return imageProvider, request
}

func newImageTestModel(t *testing.T, response any) (fantasy.ImageModel, *imageRequest) {
	t.Helper()
	p, request := newImageTestProvider(t, response)
	model, err := p.ImageModel(t.Context(), "gpt-image-1")
	require.NoError(t, err)
	return model, request
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
		})
		require.NoError(t, err)

		settings := make([]string, 0, len(response.Warnings))
		for _, warning := range response.Warnings {
			require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, warning.Type)
			settings = append(settings, warning.Setting)
		}
		require.ElementsMatch(t, []string{"aspect_ratio", "seed"}, settings)
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

func TestImageModel_EmptyModelID(t *testing.T) {
	t.Parallel()
	p, _ := newImageTestProvider(t, nil)
	model, err := p.ImageModel(t.Context(), "")
	var apiErr *fantasy.Error
	require.ErrorAs(t, err, &apiErr)
	require.Equal(t, "invalid argument", apiErr.Title)
	require.Nil(t, model)
}

func TestImageModel_NilOptions(t *testing.T) {
	t.Parallel()
	t.Run("constructor", func(t *testing.T) {
		require.Equal(t, fantasy.ProviderOptions{}, NewImageProviderOptions(nil))
	})
	t.Run("typed nil", func(t *testing.T) {
		model, request := newImageTestModel(t, imageResponse([]byte("image")))
		_, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt:          "a cat",
			ProviderOptions: fantasy.ProviderOptions{Name: (*ImageProviderOptions)(nil)},
		})
		require.NoError(t, err)
		require.NotContains(t, request.jsonBody(t), "quality")
	})
}

func TestImageModel_Generate_InvalidCalls(t *testing.T) {
	t.Parallel()
	file := fantasy.FilePart{Data: []byte("image"), MediaType: "image/png"}
	for _, tt := range []struct {
		name string
		call fantasy.ImageCall
	}{
		{"zero n", fantasy.ImageCall{N: new(int64(0))}},
		{"negative n", fantasy.ImageCall{N: new(int64(-1))}},
		{"empty reference file", fantasy.ImageCall{Files: []fantasy.FilePart{file, {}}}},
		{"mask without file", fantasy.ImageCall{Mask: &file}},
		{"empty mask", fantasy.ImageCall{Files: []fantasy.FilePart{file}, Mask: &fantasy.FilePart{}}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			model, request := newImageTestModel(t, imageResponse([]byte("image")))
			tt.call.Prompt = "a cat"
			_, err := model.Generate(t.Context(), tt.call)
			var apiErr *fantasy.Error
			require.ErrorAs(t, err, &apiErr)
			require.Equal(t, "invalid argument", apiErr.Title)
			require.Empty(t, request.path)
		})
	}
}

func TestImageModel_Generate_ResponseData(t *testing.T) {
	t.Parallel()
	call := fantasy.ImageCall{Prompt: "a cat"}
	t.Run("JSON null", func(t *testing.T) {
		model, _ := newImageTestModel(t, json.RawMessage(`null`))
		response, err := model.Generate(t.Context(), call)
		var apiErr *fantasy.Error
		require.ErrorAs(t, err, &apiErr)
		require.Equal(t, "invalid response", apiErr.Title)
		require.Nil(t, response)
	})
	t.Run("URL only", func(t *testing.T) {
		model, _ := newImageTestModel(t, json.RawMessage(`{"data":[{"url":"https://example.invalid/image"}]}`))
		response, err := model.Generate(t.Context(), call)
		require.ErrorContains(t, err, "URL")
		require.ErrorContains(t, err, "does not download")
		require.Nil(t, response)
	})
	t.Run("mixed URLs and image data", func(t *testing.T) {
		model, _ := newImageTestModel(t, json.RawMessage(`{"data":[
			{"url":"https://example.invalid/1"}, {}, {"url":"https://example.invalid/2"},
			{"url":"https://example.invalid/3","b64_json":"Y2F0"}]}`))
		response, err := model.Generate(t.Context(), call)
		require.NoError(t, err)
		require.Len(t, response.Images, 1)
		require.Equal(t, []byte("cat"), response.Images[0].Data)
		require.Len(t, response.Warnings, 1)
		require.Equal(t, fantasy.CallWarningTypeOther, response.Warnings[0].Type)
		require.Contains(t, response.Warnings[0].Message, "URL-only images skipped: 2")
	})
	t.Run("detect each format when omitted", func(t *testing.T) {
		data := imageResponse([]byte("\xff\xd8\xff"), []byte("\x89PNG\r\n\x1a\n"))
		delete(data, "output_format")
		model, _ := newImageTestModel(t, data)
		response, err := model.Generate(t.Context(), call)
		require.NoError(t, err)
		require.Len(t, response.Images, 2)
		require.Equal(t, "image/jpeg", response.Images[0].MediaType)
		require.Equal(t, "image/png", response.Images[1].MediaType)
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

		imageProvider, request := newImageTestProvider(t, imageResponse([]byte("edited")))
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
		form := request.multipartForm(t)
		for _, setting := range settings {
			require.NotContains(t, form.Value, setting)
			require.NotContains(t, form.File, setting)
		}
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
