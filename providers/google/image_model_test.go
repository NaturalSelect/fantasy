package google

import (
	"encoding/base64"
	"encoding/json"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
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
			Files:           []fantasy.FilePart{{Data: []byte("cat"), MediaType: "image/png"}},
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

func requireImageError(t *testing.T, err error, title string) *fantasy.Error {
	t.Helper()
	var imageErr *fantasy.Error
	require.ErrorAs(t, err, &imageErr)
	require.Equal(t, title, imageErr.Title)
	return imageErr
}

func TestImageModel_InvalidArguments(t *testing.T) {
	t.Parallel()

	for name, tc := range map[string]struct {
		call  fantasy.ImageCall
		field string
	}{
		"zero count":     {fantasy.ImageCall{N: new(int64(0))}, "n"},
		"negative count": {fantasy.ImageCall{N: new(int64(-1))}, "n"},
		"orphan mask":    {fantasy.ImageCall{Mask: &fantasy.FilePart{Data: []byte("mask")}}, "mask"},
		"empty file":     {fantasy.ImageCall{Files: []fantasy.FilePart{{}}}, "file"},
		"empty mask":     {fantasy.ImageCall{Files: []fantasy.FilePart{{Data: []byte("cat")}}, Mask: &fantasy.FilePart{}}, "mask"},
		"seed overflow":  {fantasy.ImageCall{Seed: new(int64(1<<32 + 42))}, "seed"},
		"seed underflow": {fantasy.ImageCall{Seed: new(int64(math.MinInt32 - 1))}, "seed"},
	} {
		t.Run(name, func(t *testing.T) {
			for _, id := range []string{"imagen-4.0-generate-001", "gemini-2.5-flash-image"} {
				model, request := newVertexImageModel(t, nil, id)
				call := tc.call
				call.Prompt = "a cat"
				_, err := model.Generate(t.Context(), call)
				require.Contains(t, requireImageError(t, err, "invalid argument").Message, tc.field)
				require.Empty(t, request.path)
			}
		})
	}
	t.Run("Imagen count overflow", func(t *testing.T) {
		model, request := newVertexImageModel(t, nil, "imagen-4.0-generate-001")
		_, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat", N: new(int64(1 << 32))})
		requireImageError(t, err, "invalid argument")
		require.Empty(t, request.path)
	})
	for field, opts := range map[string]*ImageProviderOptions{
		"person_generation": {PersonGeneration: new("allow_adult")},
		"output_mime_type":  {OutputMIMEType: new("image/jpeg")},
	} {
		t.Run(field, func(t *testing.T) {
			model, request := newGeminiImageModel(t, nil, "gemini-2.5-flash-image")
			_, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat", ProviderOptions: NewImageProviderOptions(opts)})
			imageErr := requireImageError(t, err, "invalid argument")
			require.Contains(t, imageErr.Message, field)
			require.Contains(t, imageErr.Message, "Vertex")
			require.Empty(t, request.path)
		})
	}
}

func TestImageModel_NilOptions(t *testing.T) {
	t.Parallel()
	require.Equal(t, fantasy.ProviderOptions{}, NewImageProviderOptions(nil))
	model, _ := newGeminiImageModel(t, geminiImageResponse(inlineImagePart("cat")), "gemini-2.5-flash-image")
	response, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat", ProviderOptions: fantasy.ProviderOptions{Name: (*ImageProviderOptions)(nil)}})
	require.NoError(t, err)
	require.Len(t, response.Images, 1)
}

func TestImageModel_InvalidResponses(t *testing.T) {
	t.Parallel()
	for name, body := range map[string]map[string]any{
		"null part":      geminiImageResponse(inlineImagePart("kept"), nil),
		"null candidate": {"candidates": []any{nil}},
	} {
		t.Run(name, func(t *testing.T) {
			for _, factory := range []func(*testing.T, map[string]any, string) (fantasy.ImageModel, *imageRequest){newGeminiImageModel, newVertexImageModel} {
				model, _ := factory(t, body, "gemini-2.5-flash-image")
				response, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat"})
				requireImageError(t, err, "invalid response")
				require.Nil(t, response)
			}
		})
	}
	t.Run("filtered candidate without content", func(t *testing.T) {
		model, _ := newGeminiImageModel(t, map[string]any{"candidates": []any{map[string]any{"finishReason": "SAFETY"}}}, "gemini-2.5-flash-image")
		_, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat"})
		requireImageError(t, err, "no image generated")
	})
	t.Run("null generated image", func(t *testing.T) {
		_, err := imagenImageResponse(&genai.GenerateImagesResponse{GeneratedImages: []*genai.GeneratedImage{nil}}, 1, nil)
		requireImageError(t, err, "invalid response")
	})
}

func TestImageModel_ResourceNamesAndMIME(t *testing.T) {
	t.Parallel()
	const jpeg = "\xff\xd8\xff\xe0"
	for _, tc := range []struct {
		id, method string
		response   map[string]any
	}{
		{"projects/imagen-project/locations/x/publishers/google/models/gemini-image", ":generateContent", geminiImageResponse(map[string]any{"inlineData": map[string]any{"data": encodeImage(jpeg)}})},
		{"projects/p/locations/x/publishers/google/models/imagen-4.0-generate-001", ":predict", imagenResponse(map[string]any{"bytesBase64Encoded": encodeImage(jpeg)})},
		{"not-imagen-image", ":generateContent", geminiImageResponse(map[string]any{"inlineData": map[string]any{"data": encodeImage(jpeg)}})},
	} {
		t.Run(tc.id, func(t *testing.T) {
			model, request := newVertexImageModel(t, tc.response, tc.id)
			response, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat"})
			require.NoError(t, err)
			require.True(t, strings.HasSuffix(request.path, tc.id+tc.method), request.path)
			require.Len(t, response.Images, 1)
			require.Equal(t, "image/jpeg", response.Images[0].MediaType)
		})
	}
}

func TestImageModel_Seed(t *testing.T) {
	t.Parallel()
	model, request := newVertexImageModel(t, imagenResponse(imagenPrediction("cat")), "imagen-4.0-generate-001")
	_, err := model.Generate(t.Context(), fantasy.ImageCall{
		Prompt:          "a cat",
		Seed:            new(int64(42)),
		ProviderOptions: NewImageProviderOptions(&ImageProviderOptions{ImageSize: new("2K")}),
	})
	require.NoError(t, err)
	parameters := request.body["parameters"].(map[string]any)
	require.Equal(t, false, parameters["addWatermark"])
	require.EqualValues(t, 42, parameters["seed"])
	require.EqualValues(t, 1, parameters["sampleCount"])
	require.Equal(t, "2K", parameters["sampleImageSize"])

	model, request = newGeminiImageModel(t, geminiImageResponse(inlineImagePart("cat")), "gemini-2.5-flash-image")
	_, err = model.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat", Seed: new(int64(math.MaxInt32))})
	require.NoError(t, err)
	require.EqualValues(t, math.MaxInt32, request.body["generationConfig"].(map[string]any)["seed"])
}

func TestImageModel_PartialResults(t *testing.T) {
	t.Parallel()
	for name, body := range map[string]map[string]any{
		"empty prediction":  imagenResponse(imagenPrediction("cat"), map[string]any{}),
		"absent prediction": imagenResponse(imagenPrediction("cat")),
	} {
		t.Run(name, func(t *testing.T) {
			model, _ := newVertexImageModel(t, body, "imagen-4.0-generate-001")
			response, err := model.Generate(t.Context(), fantasy.ImageCall{Prompt: "cats", N: new(int64(2))})
			require.NoError(t, err)
			require.Len(t, response.Images, 1)
			require.Equal(t, []byte("cat"), response.Images[0].Data)
			require.Len(t, response.Warnings, 1)
			require.Equal(t, fantasy.CallWarningTypeOther, response.Warnings[0].Type)
		})
	}
}

func TestImageAndEmbeddingModel_EmptyModelID(t *testing.T) {
	t.Parallel()
	p, err := New(WithGeminiAPIKey("test-key"))
	require.NoError(t, err)
	t.Run("image", func(t *testing.T) {
		_, err := p.(fantasy.ImageModelProvider).ImageModel(t.Context(), "")
		requireImageError(t, err, "invalid argument")
	})
	t.Run("embedding", func(t *testing.T) {
		_, err := p.(fantasy.EmbeddingModelProvider).EmbeddingModel(t.Context(), "")
		requireInvalidEmbeddingArgument(t, err)
	})
}

func TestImageAndEmbeddingModel_EnvironmentBackends(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "test-key")
	t.Setenv("GEMINI_API_KEY", "")
	t.Setenv("GOOGLE_GENAI_USE_ENTERPRISE", "")
	require.NoError(t, os.Unsetenv("GOOGLE_GENAI_USE_ENTERPRISE"))
	for _, tc := range []struct {
		name, vertex, enterprise string
		wantVertex               bool
	}{
		{"Vertex", "true", "", true},
		{"Developer", "false", "", false},
		{"Enterprise overrides Vertex", "true", "false", false},
		{"Enterprise enables Vertex", "false", "1", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", tc.vertex)
			if tc.enterprise != "" {
				t.Setenv("GOOGLE_GENAI_USE_ENTERPRISE", tc.enterprise)
			}
			t.Run("Imagen", func(t *testing.T) {
				server, requests := newEmbeddingTestServer(t, http.StatusOK, imagenResponse(imagenPrediction("cat")))
				p, err := New(WithBaseURL(server.URL), WithSkipAuth(true))
				require.NoError(t, err)
				image, err := p.(fantasy.ImageModelProvider).ImageModel(t.Context(), "imagen-4.0-generate-001")
				require.NoError(t, err)
				_, err = image.Generate(t.Context(), fantasy.ImageCall{Prompt: "a cat"})
				if !tc.wantVertex {
					requireImageError(t, err, "unsupported model")
					require.Empty(t, requests.requests())
					return
				}
				require.NoError(t, err)
				require.True(t, strings.HasSuffix(requests.only(t).path, ":predict"))
			})
			t.Run("Gemini image", func(t *testing.T) {
				body := geminiImageResponse(inlineImagePart("cat"))
				body["usageMetadata"] = map[string]any{"candidatesTokenCount": 3, "thoughtsTokenCount": 2}
				server, requests := newEmbeddingTestServer(t, http.StatusOK, body)
				p, err := New(WithBaseURL(server.URL), WithSkipAuth(true))
				require.NoError(t, err)
				image, err := p.(fantasy.ImageModelProvider).ImageModel(t.Context(), "gemini-2.5-flash-image")
				require.NoError(t, err)
				response, err := image.Generate(t.Context(), fantasy.ImageCall{
					Prompt: "a cat",
					ProviderOptions: NewImageProviderOptions(&ImageProviderOptions{
						PersonGeneration: new("allow_adult"), OutputMIMEType: new("image/jpeg"),
					}),
				})
				if !tc.wantVertex {
					requireImageError(t, err, "invalid argument")
					require.Empty(t, requests.requests())
					return
				}
				require.NoError(t, err)
				imageConfig := requests.only(t).body["generationConfig"].(map[string]any)["imageConfig"].(map[string]any)
				require.Equal(t, "allow_adult", imageConfig["personGeneration"])
				require.Equal(t, map[string]any{"mimeType": "image/jpeg"}, imageConfig["imageOutputOptions"])
				require.EqualValues(t, 5, response.Usage.OutputTokens)
			})
			t.Run("embedding", func(t *testing.T) {
				server, requests := newEmbeddingTestServer(t, http.StatusOK, map[string]any{"embedding": map[string]any{"values": []float32{1}}})
				embedding := newTestEmbeddingModel(t, "gemini-embedding-2", WithBaseURL(server.URL), WithSkipAuth(true))
				response, err := embedding.Embed(t.Context(), fantasy.EmbeddingCall{
					Inputs: fantasy.TextEmbeddingInputs("a"),
					ProviderOptions: NewEmbeddingProviderOptions(&EmbeddingProviderOptions{
						DocumentOCR: new(true), AudioTrackExtraction: new(false),
					}),
				})
				if !tc.wantVertex {
					requireInvalidEmbeddingArgument(t, err)
					require.Empty(t, requests.requests())
					return
				}
				require.NoError(t, err)
				req := requests.only(t)
				require.True(t, strings.HasSuffix(req.path, ":embedContent"))
				require.Equal(t, map[string]any{"documentOcr": true, "audioTrackExtraction": false}, req.body["embedContentConfig"])
				require.Empty(t, response.Warnings)
			})
		})
	}
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

		model, request := newVertexImageModel(t, imagenResponse(imagenPrediction("imagen-image")), "imagen-3.0-generate-001")

		response, err := model.Generate(t.Context(), fantasy.ImageCall{
			Prompt:          "a cat",
			AspectRatio:     "16:9",
			ProviderOptions: NewImageProviderOptions(&ImageProviderOptions{NegativePrompt: new("dogs")}),
		})
		require.NoError(t, err)

		require.True(t, strings.HasSuffix(request.path, "imagen-3.0-generate-001:predict"), request.path)
		instance := request.body["instances"].([]any)[0].(map[string]any)
		require.Equal(t, "a cat", instance["prompt"])
		parameters := request.body["parameters"].(map[string]any)
		require.EqualValues(t, 1, parameters["sampleCount"])
		require.Equal(t, "16:9", parameters["aspectRatio"])
		require.NotContains(t, parameters, "seed")
		require.NotContains(t, parameters, "addWatermark")
		require.Equal(t, "dogs", parameters["negativePrompt"])

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
			N:      new(int64(2)),
		})
		require.NoError(t, err)

		require.Len(t, response.Images, 1)
		require.Len(t, response.Warnings, 2)
		require.Equal(t, []string{"size", ""}, warningSettings(response.Warnings))
		require.Equal(t, fantasy.CallWarningTypeOther, response.Warnings[1].Type)
		require.Contains(t, response.Warnings[1].Message, "blocked by policy")
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
		require.ErrorContains(t, err, "blocked by policy")
	})
}
