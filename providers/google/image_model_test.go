package google

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

type imageRequest struct {
	path      string
	userAgent string
	body      map[string]any
}

func newImageTestServer(t *testing.T, response map[string]any) (*httptest.Server, *imageRequest) {
	t.Helper()

	request := &imageRequest{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		*request = imageRequest{path: r.URL.Path, userAgent: r.Header.Get("User-Agent"), body: body}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)
	return server, request
}

func newGeminiImageModel(t *testing.T, response map[string]any, modelID string) (fantasy.ImageModel, *imageRequest) {
	t.Helper()

	server, request := newImageTestServer(t, response)
	p, err := New(WithGeminiAPIKey("test-api-key"), WithBaseURL(server.URL))
	require.NoError(t, err)
	imageProvider, ok := p.(fantasy.ImageModelProvider)
	require.True(t, ok)
	model, err := imageProvider.ImageModel(t.Context(), modelID)
	require.NoError(t, err)
	return model, request
}

func newVertexImageModel(t *testing.T, response map[string]any, modelID string) (fantasy.ImageModel, *imageRequest) {
	t.Helper()

	server, request := newImageTestServer(t, response)
	p, err := New(WithVertex("test-project", "us-central1"), WithBaseURL(server.URL), WithSkipAuth(true))
	require.NoError(t, err)
	imageProvider, ok := p.(fantasy.ImageModelProvider)
	require.True(t, ok)
	model, err := imageProvider.ImageModel(t.Context(), modelID)
	require.NoError(t, err)
	return model, request
}

func encodeImage(data string) string {
	return base64.StdEncoding.EncodeToString([]byte(data))
}

func geminiImageResponse(parts ...map[string]any) map[string]any {
	return map[string]any{
		"candidates": []map[string]any{{
			"content":      map[string]any{"role": "model", "parts": parts},
			"finishReason": "STOP",
		}},
		"usageMetadata": map[string]any{
			"promptTokenCount":     5,
			"candidatesTokenCount": 1290,
			"totalTokenCount":      1295,
		},
	}
}

func inlineImagePart(data string) map[string]any {
	return map[string]any{"inlineData": map[string]any{"mimeType": "image/png", "data": encodeImage(data)}}
}

func warningSettings(warnings []fantasy.CallWarning) []string {
	settings := make([]string, 0, len(warnings))
	for _, warning := range warnings {
		settings = append(settings, warning.Setting)
	}
	return settings
}

func TestImageModel_Gemini(t *testing.T) {
	t.Parallel()

	t.Run("should generate an image and map the request", func(t *testing.T) {
		t.Parallel()

		model, request := newGeminiImageModel(t, geminiImageResponse(
			map[string]any{"text": "Here is your cat."},
			map[string]any{"thought": true, "inlineData": map[string]any{"mimeType": "image/png", "data": encodeImage("draft")}},
			inlineImagePart("final-image"),
		), "gemini-2.5-flash-image")
		require.Equal(t, "gemini-2.5-flash-image", model.Model())
		require.Equal(t, "google", model.Provider())

		imageSize := "2K"
		response, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt:          "a cat",
			AspectRatio:     "16:9",
			ProviderOptions: fantasy.ProviderOptions{Name: &ImageProviderOptions{ImageSize: &imageSize}},
		})
		require.NoError(t, err)

		require.True(t, strings.HasPrefix(request.path, "/v1beta/"), request.path)
		require.True(t, strings.HasSuffix(request.path, "gemini-2.5-flash-image:generateContent"), request.path)
		contents := request.body["contents"].([]any)
		require.Len(t, contents, 1)
		parts := contents[0].(map[string]any)["parts"].([]any)
		require.Len(t, parts, 1)
		require.Equal(t, "a cat", parts[0].(map[string]any)["text"])

		generationConfig := request.body["generationConfig"].(map[string]any)
		require.Equal(t, []any{"TEXT", "IMAGE"}, generationConfig["responseModalities"])
		imageConfig := generationConfig["imageConfig"].(map[string]any)
		require.Equal(t, "16:9", imageConfig["aspectRatio"])
		require.Equal(t, "2K", imageConfig["imageSize"])

		require.Len(t, response.Images, 1)
		require.Equal(t, []byte("final-image"), response.Images[0].Data)
		require.Equal(t, "image/png", response.Images[0].MediaType)
		require.EqualValues(t, 5, response.Usage.InputTokens)
		require.EqualValues(t, 1295, response.Usage.TotalTokens)
		require.Empty(t, response.Warnings)
	})

	t.Run("should send reference images before the prompt", func(t *testing.T) {
		t.Parallel()

		model, request := newGeminiImageModel(t, geminiImageResponse(inlineImagePart("edited")), "gemini-2.5-flash-image")

		_, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt: "add a hat",
			Files:  []fantasy.FilePart{{Data: []byte("cat-bytes"), MediaType: "image/jpeg"}},
		})
		require.NoError(t, err)

		parts := request.body["contents"].([]any)[0].(map[string]any)["parts"].([]any)
		require.Len(t, parts, 2)
		inlineData := parts[0].(map[string]any)["inlineData"].(map[string]any)
		require.Equal(t, "image/jpeg", inlineData["mimeType"])
		require.Equal(t, encodeImage("cat-bytes"), inlineData["data"])
		require.Equal(t, "add a hat", parts[1].(map[string]any)["text"])
	})

	t.Run("should warn about unsupported settings", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiImageModel(t, geminiImageResponse(inlineImagePart("image")), "gemini-2.5-flash-image")

		n := int64(2)
		negativePrompt := "dogs"
		response, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt:          "a cat",
			N:               &n,
			Size:            "1024x1024",
			Mask:            &fantasy.FilePart{Data: []byte("mask"), MediaType: "image/png"},
			ProviderOptions: fantasy.ProviderOptions{Name: &ImageProviderOptions{NegativePrompt: &negativePrompt}},
		})
		require.NoError(t, err)

		require.Equal(t, []string{"n", "size", "mask", "negative_prompt"}, warningSettings(response.Warnings))
	})

	t.Run("should override the user agent per call", func(t *testing.T) {
		t.Parallel()

		model, request := newGeminiImageModel(t, geminiImageResponse(inlineImagePart("image")), "gemini-2.5-flash-image")

		_, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat", UserAgent: "my-agent/1.0"})
		require.NoError(t, err)

		require.Equal(t, "my-agent/1.0", request.userAgent)
	})

	t.Run("should fail with the model text when no image is returned", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiImageModel(t, geminiImageResponse(map[string]any{"text": "I cannot draw that."}), "gemini-2.5-flash-image")

		_, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat"})
		require.ErrorContains(t, err, "no image generated")
		require.ErrorContains(t, err, "I cannot draw that.")
	})

	t.Run("should fail when the prompt is empty", func(t *testing.T) {
		t.Parallel()

		model, request := newGeminiImageModel(t, geminiImageResponse(inlineImagePart("image")), "gemini-2.5-flash-image")

		_, err := model.Generate(t.Context(), fantasy.ImageCall{})
		require.Error(t, err)
		require.Empty(t, request.path)
	})

	t.Run("should fail when the provider options have the wrong type", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiImageModel(t, geminiImageResponse(inlineImagePart("image")), "gemini-2.5-flash-image")

		_, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt:          "a cat",
			ProviderOptions: fantasy.ProviderOptions{Name: &ProviderOptions{}},
		})
		require.ErrorContains(t, err, "ImageProviderOptions")
	})
}

func imagenResponse(predictions ...map[string]any) map[string]any {
	return map[string]any{"predictions": predictions}
}

func imagenPrediction(data string) map[string]any {
	return map[string]any{"mimeType": "image/png", "bytesBase64Encoded": encodeImage(data)}
}

func TestImageModel_Imagen(t *testing.T) {
	t.Parallel()

	t.Run("should generate one image by default and map the request", func(t *testing.T) {
		t.Parallel()

		model, request := newVertexImageModel(t, imagenResponse(imagenPrediction("imagen-image")), "imagen-4.0-generate-001")

		seed := int64(42)
		negativePrompt := "dogs"
		imageSize := "2K"
		response, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt:      "a cat",
			AspectRatio: "16:9",
			Seed:        &seed,
			ProviderOptions: fantasy.ProviderOptions{Name: &ImageProviderOptions{
				NegativePrompt: &negativePrompt,
				ImageSize:      &imageSize,
			}},
		})
		require.NoError(t, err)

		require.True(t, strings.HasSuffix(request.path, "imagen-4.0-generate-001:predict"), request.path)
		instance := request.body["instances"].([]any)[0].(map[string]any)
		require.Equal(t, "a cat", instance["prompt"])
		parameters := request.body["parameters"].(map[string]any)
		require.EqualValues(t, 1, parameters["sampleCount"])
		require.Equal(t, "16:9", parameters["aspectRatio"])
		require.EqualValues(t, 42, parameters["seed"])
		require.Equal(t, "dogs", parameters["negativePrompt"])
		require.Equal(t, "2K", parameters["sampleImageSize"])

		require.Len(t, response.Images, 1)
		require.Equal(t, []byte("imagen-image"), response.Images[0].Data)
		require.Equal(t, "image/png", response.Images[0].MediaType)
		require.Empty(t, response.Warnings)
	})

	t.Run("should request the number of images asked for", func(t *testing.T) {
		t.Parallel()

		model, request := newVertexImageModel(t, imagenResponse(imagenPrediction("one"), imagenPrediction("two")), "imagen-4.0-generate-001")

		n := int64(2)
		response, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat", N: &n})
		require.NoError(t, err)

		require.EqualValues(t, 2, request.body["parameters"].(map[string]any)["sampleCount"])
		require.Len(t, response.Images, 2)
	})

	t.Run("should warn about unsupported settings and filtered images", func(t *testing.T) {
		t.Parallel()

		model, _ := newVertexImageModel(t, imagenResponse(
			imagenPrediction("kept"),
			map[string]any{"raiFilteredReason": "blocked by policy"},
		), "imagen-4.0-generate-001")

		response, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt: "a cat",
			Size:   "1024x1024",
			Mask:   &fantasy.FilePart{Data: []byte("mask"), MediaType: "image/png"},
		})
		require.NoError(t, err)

		require.Len(t, response.Images, 1)
		require.Len(t, response.Warnings, 3)
		require.Equal(t, []string{"size", "mask", ""}, warningSettings(response.Warnings))
		require.Equal(t, fantasy.CallWarningTypeOther, response.Warnings[2].Type)
		require.Contains(t, response.Warnings[2].Message, "blocked by policy")
	})

	t.Run("should reject reference images", func(t *testing.T) {
		t.Parallel()

		model, request := newVertexImageModel(t, imagenResponse(imagenPrediction("image")), "imagen-4.0-generate-001")

		_, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt: "add a hat",
			Files:  []fantasy.FilePart{{Data: []byte("cat"), MediaType: "image/png"}},
		})
		require.ErrorContains(t, err, "reference images")
		require.Empty(t, request.path)
	})

	t.Run("should reject the Gemini API backend", func(t *testing.T) {
		t.Parallel()

		model, request := newGeminiImageModel(t, imagenResponse(imagenPrediction("image")), "imagen-4.0-generate-001")

		_, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat"})
		require.ErrorContains(t, err, "Vertex")
		require.Empty(t, request.path)
	})

	t.Run("should fail when every image is filtered", func(t *testing.T) {
		t.Parallel()

		model, _ := newVertexImageModel(t, imagenResponse(map[string]any{"raiFilteredReason": "blocked by policy"}), "imagen-4.0-generate-001")

		_, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat"})
		require.ErrorContains(t, err, "no image generated")
	})
}
