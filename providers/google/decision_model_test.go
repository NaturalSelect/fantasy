package google

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
	"google.golang.org/genai/interactions/models/apierrors"
)

type decisionRequest struct {
	method string
	path   string
	header http.Header
	body   map[string]any
}

type decisionRecorder struct {
	mu       sync.Mutex
	recorded []decisionRequest
}

func (r *decisionRecorder) record(request decisionRequest) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.recorded = append(r.recorded, request)
}

func (r *decisionRecorder) requests() []decisionRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.recorded)
}

func (r *decisionRecorder) only(t *testing.T) decisionRequest {
	t.Helper()
	requests := r.requests()
	require.Len(t, requests, 1)
	return requests[0]
}

func newDecisionTestServer(t *testing.T, status int, response any) (*httptest.Server, *decisionRecorder) {
	t.Helper()

	recorder := &decisionRecorder{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		recorder.record(decisionRequest{method: r.Method, path: r.URL.Path, header: r.Header.Clone(), body: body})
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_ = json.NewEncoder(w).Encode(response)
	}))
	t.Cleanup(server.Close)
	return server, recorder
}

func newGeminiDecisionProvider(t *testing.T, server *httptest.Server, opts ...Option) fantasy.DecisionModelProvider {
	t.Helper()

	p, err := New(append([]Option{WithGeminiAPIKey("test-api-key"), WithBaseURL(server.URL)}, opts...)...)
	require.NoError(t, err)
	decisionProvider, ok := p.(fantasy.DecisionModelProvider)
	require.True(t, ok)
	return decisionProvider
}

func newGeminiDecisionModel(t *testing.T, status int, response any, opts ...Option) (fantasy.DecisionModel, *decisionRecorder) {
	t.Helper()

	server, recorder := newDecisionTestServer(t, status, response)
	model, err := newGeminiDecisionProvider(t, server, opts...).DecisionModel(t.Context(), "")
	require.NoError(t, err)
	return model, recorder
}

func noulAnswerBody(noul, confidence float64) map[string]any {
	return map[string]any{"type": "noul", "noul": noul, "confidence": confidence}
}

func choiceAnswerBody(choice string, probabilities map[string]float64, confidence float64) map[string]any {
	return map[string]any{"type": "choice", "choice": choice, "probabilities": probabilities, "confidence": confidence}
}

func scoreAnswerBody(score float64, probabilities map[string]float64, legend map[string]string, confidence float64) map[string]any {
	return map[string]any{"type": "score", "score": score, "probabilities": probabilities, "legend": legend, "confidence": confidence}
}

func decisionResponseBody(answers map[string]any, usage map[string]any) map[string]any {
	body := map[string]any{"answers": answers}
	if usage != nil {
		body["usage"] = usage
	}
	return body
}

// blueResponseBody answers the question of blueCall.
func blueResponseBody() map[string]any {
	return decisionResponseBody(map[string]any{"blue": noulAnswerBody(0.9, 0.9)}, nil)
}

func decisionText(text string) []fantasy.MessagePart {
	return []fantasy.MessagePart{fantasy.TextPart{Text: text}}
}

func questionCall(questions map[string]fantasy.Question) fantasy.DecisionCall {
	return fantasy.DecisionCall{Input: decisionText("The sky is blue."), Questions: questions}
}

func blueCall() fantasy.DecisionCall {
	return questionCall(map[string]fantasy.Question{
		"blue": fantasy.NoulQuestion{Instructions: fantasy.String("Is the sky blue?")},
	})
}

func choiceOptions(n int) []fantasy.ChoiceOption {
	options := make([]fantasy.ChoiceOption, 0, n)
	for i := range n {
		options = append(options, fantasy.ChoiceOption{Name: "option" + strconv.Itoa(i)})
	}
	return options
}

func scoreLevels(n int) []fantasy.ScoreLevel {
	levels := make([]fantasy.ScoreLevel, 0, n)
	for i := range n {
		levels = append(levels, fantasy.ScoreLevel{Label: "level" + strconv.Itoa(i)})
	}
	return levels
}

func requireFantasyError(t *testing.T, err error, title, message string) {
	t.Helper()

	var fantasyErr *fantasy.Error
	require.ErrorAs(t, err, &fantasyErr)
	require.Equal(t, title, fantasyErr.Title)
	require.Contains(t, fantasyErr.Message, message)
}

func TestDecisionModel_Request(t *testing.T) {
	t.Parallel()

	t.Run("should post to the v1beta decisions endpoint with the api key", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())
		require.Equal(t, "google", model.Provider())
		require.Equal(t, "gemini-decision-1", model.Model())

		_, err := model.Decide(t.Context(), blueCall())
		require.NoError(t, err)

		request := recorder.only(t)
		require.Equal(t, http.MethodPost, request.method)
		require.Equal(t, "/v1beta/decisions", request.path)
		require.Equal(t, "test-api-key", request.header.Get("x-goog-api-key"))
		require.Equal(t, "gemini-decision-1", request.body["model"])
	})

	t.Run("should use the requested model id", func(t *testing.T) {
		t.Parallel()

		server, recorder := newDecisionTestServer(t, http.StatusOK, blueResponseBody())
		model, err := newGeminiDecisionProvider(t, server).DecisionModel(t.Context(), "gemini-decision-2")
		require.NoError(t, err)
		require.Equal(t, "gemini-decision-2", model.Model())

		response, err := model.Decide(t.Context(), blueCall())
		require.NoError(t, err)

		require.Equal(t, "gemini-decision-2", recorder.only(t).body["model"])
		require.Equal(t, "gemini-decision-2", response.Model)
	})

	t.Run("should map a choice question with and without option descriptions", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(map[string]any{
			"animal": choiceAnswerBody("cat", map[string]float64{"cat": 0.8, "dog": 0.2}, 0.8),
		}, nil))

		_, err := model.Decide(t.Context(), questionCall(map[string]fantasy.Question{
			"animal": fantasy.ChoiceQuestion{
				Instructions: fantasy.String("Which animal is shown?"),
				Options: []fantasy.ChoiceOption{
					{Name: "cat", Description: fantasy.String("A small feline")},
					{Name: "dog"},
				},
			},
		}))
		require.NoError(t, err)

		require.Equal(t, map[string]any{
			"animal": map[string]any{
				"type":         "choice",
				"instructions": "Which animal is shown?",
				"criteria":     map[string]any{"cat": "A small feline", "dog": ""},
			},
		}, recorder.only(t).body["questions"])
	})

	t.Run("should map score levels as label and description text", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(map[string]any{
			"quality": scoreAnswerBody(1, map[string]float64{"0": 0, "1": 1}, map[string]string{}, 1),
		}, nil))

		_, err := model.Decide(t.Context(), questionCall(map[string]fantasy.Question{
			"quality": fantasy.ScoreQuestion{
				Instructions: fantasy.String("Rate the quality."),
				Levels: []fantasy.ScoreLevel{
					{Label: "Bad", Description: fantasy.String("Unusable")},
					{Label: "Good"},
				},
			},
		}))
		require.NoError(t, err)

		require.Equal(t, map[string]any{
			"quality": map[string]any{
				"type":         "score",
				"instructions": "Rate the quality.",
				"criteria":     []any{"Bad: Unusable", "Good"},
			},
		}, recorder.only(t).body["questions"])
	})

	t.Run("should only send noul criteria that have a description", func(t *testing.T) {
		t.Parallel()

		for _, tc := range []struct {
			name             string
			trueDescription  fantasy.Text
			falseDescription fantasy.Text
			wantQuestion     map[string]any
		}{
			{
				name:         "no descriptions",
				wantQuestion: map[string]any{"type": "noul", "instructions": "Is the sky blue?"},
			},
			{
				name:            "both descriptions",
				trueDescription: fantasy.String("It is blue"), falseDescription: fantasy.String("It is not blue"),
				wantQuestion: map[string]any{
					"type":         "noul",
					"instructions": "Is the sky blue?",
					"criteria":     map[string]any{"true_description": "It is blue", "false_description": "It is not blue"},
				},
			},
			{
				name:            "only the true description",
				trueDescription: fantasy.String("It is blue"),
				wantQuestion: map[string]any{
					"type":         "noul",
					"instructions": "Is the sky blue?",
					"criteria":     map[string]any{"true_description": "It is blue"},
				},
			},
			{
				name:             "only the false description",
				falseDescription: fantasy.String("It is not blue"),
				wantQuestion: map[string]any{
					"type":         "noul",
					"instructions": "Is the sky blue?",
					"criteria":     map[string]any{"false_description": "It is not blue"},
				},
			},
			{
				name:            "empty descriptions",
				trueDescription: fantasy.String(""), falseDescription: fantasy.String(""),
				wantQuestion: map[string]any{"type": "noul", "instructions": "Is the sky blue?"},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())

				_, err := model.Decide(t.Context(), questionCall(map[string]fantasy.Question{
					"blue": fantasy.NoulQuestion{
						Instructions:     fantasy.String("Is the sky blue?"),
						TrueDescription:  tc.trueDescription,
						FalseDescription: tc.falseDescription,
					},
				}))
				require.NoError(t, err)

				require.Equal(t, map[string]any{"blue": tc.wantQuestion}, recorder.only(t).body["questions"])
			})
		}
	})

	t.Run("should render json text as instructions", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())

		_, err := model.Decide(t.Context(), questionCall(map[string]fantasy.Question{
			"blue": fantasy.NoulQuestion{Instructions: fantasy.JSONText(map[string]string{"topic": "sky"})},
		}))
		require.NoError(t, err)

		question := recorder.only(t).body["questions"].(map[string]any)["blue"].(map[string]any)
		require.Equal(t, `{"topic":"sky"}`, question["instructions"])
	})
}

func TestDecisionModel_Input(t *testing.T) {
	t.Parallel()

	t.Run("should send a single text part as a plain string", func(t *testing.T) {
		t.Parallel()

		for name, part := range map[string]fantasy.MessagePart{
			"by value":   fantasy.TextPart{Text: "The sky is blue."},
			"by pointer": &fantasy.TextPart{Text: "The sky is blue."},
		} {
			t.Run(name, func(t *testing.T) {
				t.Parallel()

				model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())
				call := blueCall()
				call.Input = []fantasy.MessagePart{part}

				_, err := model.Decide(t.Context(), call)
				require.NoError(t, err)

				require.Equal(t, "The sky is blue.", recorder.only(t).body["input"])
			})
		}
	})

	t.Run("should send several parts as a content array in order", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())
		call := blueCall()
		call.Input = []fantasy.MessagePart{
			fantasy.TextPart{Text: "Look at this picture:"},
			fantasy.FilePart{Data: []byte("picture-bytes"), MediaType: "image/png"},
			&fantasy.FilePart{Data: []byte("other-bytes"), MediaType: "image/jpeg"},
			fantasy.TextPart{Text: "What do you see?"},
		}

		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)

		require.Equal(t, []any{
			map[string]any{"type": "text", "text": "Look at this picture:"},
			map[string]any{"type": "image", "data": encodeImage("picture-bytes"), "mime_type": "image/png"},
			map[string]any{"type": "image", "data": encodeImage("other-bytes"), "mime_type": "image/jpeg"},
			map[string]any{"type": "text", "text": "What do you see?"},
		}, recorder.only(t).body["input"])
	})

	t.Run("should send a single image as a content array", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())
		call := blueCall()
		call.Input = []fantasy.MessagePart{fantasy.FilePart{Data: []byte("picture-bytes"), MediaType: "image/webp"}}

		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)

		require.Equal(t, []any{
			map[string]any{"type": "image", "data": encodeImage("picture-bytes"), "mime_type": "image/webp"},
		}, recorder.only(t).body["input"])
	})

	t.Run("should accept every supported image type", func(t *testing.T) {
		t.Parallel()

		mediaTypes := []string{"image/png", "image/jpeg", "image/webp", "image/heic", "image/heif", "image/gif", "image/bmp", "image/tiff"}
		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())
		for _, mediaType := range mediaTypes {
			call := blueCall()
			call.Input = []fantasy.MessagePart{fantasy.FilePart{Data: []byte("picture-bytes"), MediaType: mediaType}}

			_, err := model.Decide(t.Context(), call)
			require.NoError(t, err, mediaType)
		}

		requests := recorder.requests()
		require.Len(t, requests, len(mediaTypes))
		for i, mediaType := range mediaTypes {
			image := requests[i].body["input"].([]any)[0].(map[string]any)
			require.Equal(t, mediaType, image["mime_type"])
		}
	})
}

func TestDecisionModel_UserAgent(t *testing.T) {
	t.Parallel()

	t.Run("should send the provider user agent", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody(), WithUserAgent("provider-agent/1.0"))

		_, err := model.Decide(t.Context(), blueCall())
		require.NoError(t, err)

		userAgent := recorder.only(t).header.Get("User-Agent")
		// NOTE: The SDK appends its own labels, such as "google-genai-sdk/1.79.1", to the Decisions user agent.
		require.True(t, strings.HasPrefix(userAgent, "provider-agent/1.0"), userAgent)
	})

	t.Run("should send the default user agent without an explicit one", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())

		_, err := model.Decide(t.Context(), blueCall())
		require.NoError(t, err)

		userAgent := recorder.only(t).header.Get("User-Agent")
		require.True(t, strings.HasPrefix(userAgent, "Charm-Fantasy/"+fantasy.Version+" (https://charm.land/fantasy)"), userAgent)
	})

	t.Run("should override the user agent per call", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody(), WithUserAgent("provider-agent/1.0"))
		call := blueCall()
		call.UserAgent = "my-agent/1.0"

		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)

		require.Equal(t, "my-agent/1.0", recorder.only(t).header.Get("User-Agent"))
	})

	t.Run("should override headers per call", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody(), WithHeaders(map[string]string{"X-Source": "provider"}))
		call := blueCall()
		call.Headers = map[string]string{"X-Source": "call"}

		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)

		require.Equal(t, "call", recorder.only(t).header.Get("X-Source"))
	})
}

func TestDecisionModel_Errors(t *testing.T) {
	t.Parallel()

	t.Run("should not retry retryable statuses and report them as retryable", func(t *testing.T) {
		t.Parallel()

		for _, status := range []int{http.StatusTooManyRequests, http.StatusServiceUnavailable} {
			t.Run(strconv.Itoa(status), func(t *testing.T) {
				t.Parallel()

				errorBody := map[string]any{"error": map[string]any{"code": status, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED"}}
				model, recorder := newGeminiDecisionModel(t, status, errorBody)

				_, err := model.Decide(t.Context(), blueCall())

				require.Len(t, recorder.requests(), 1)
				var providerErr *fantasy.ProviderError
				require.ErrorAs(t, err, &providerErr)
				require.True(t, providerErr.IsRetryable())
				require.Equal(t, status, providerErr.StatusCode)
				require.Equal(t, "quota exceeded", providerErr.Message)
				require.Equal(t, strings.ToLower(http.StatusText(status)), providerErr.Title)
				// NOTE: A numeric code carries no name, so the status is the error type.
				require.Equal(t, "RESOURCE_EXHAUSTED", providerErr.ErrorType)
				require.True(t, strings.HasSuffix(providerErr.URL, "/v1beta/decisions"), providerErr.URL)
				require.Equal(t, "application/json", providerErr.ResponseHeaders["content-type"])
				require.Equal(t, "application/json", providerErr.ResponseHeaders["Content-Type"])
				expectedBody, marshalErr := json.Marshal(errorBody)
				require.NoError(t, marshalErr)
				require.JSONEq(t, string(expectedBody), string(providerErr.ResponseBody))
			})
		}
	})

	t.Run("should use a string error code as the error type", func(t *testing.T) {
		t.Parallel()

		model, recorder := newGeminiDecisionModel(t, http.StatusBadRequest, map[string]any{
			"error": map[string]any{"code": "safety", "message": "blocked"},
		})

		_, err := model.Decide(t.Context(), blueCall())

		require.Len(t, recorder.requests(), 1)
		var providerErr *fantasy.ProviderError
		require.ErrorAs(t, err, &providerErr)
		require.Equal(t, http.StatusBadRequest, providerErr.StatusCode)
		require.Equal(t, "safety", providerErr.ErrorType)
		require.Equal(t, "blocked", providerErr.Message)
		require.Equal(t, "bad request", providerErr.Title)
		require.False(t, providerErr.IsRetryable())
	})

	t.Run("should fall back to the sdk message when the body is not an error envelope", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiDecisionModel(t, http.StatusBadGateway, "upstream exploded")

		_, err := model.Decide(t.Context(), blueCall())

		var providerErr *fantasy.ProviderError
		require.ErrorAs(t, err, &providerErr)
		require.Equal(t, http.StatusBadGateway, providerErr.StatusCode)
		require.NotEmpty(t, providerErr.Message)
		require.Empty(t, providerErr.ErrorType)
		require.Contains(t, string(providerErr.ResponseBody), "upstream exploded")
	})

	t.Run("should report a blocked prompt returned with a success status", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiDecisionModel(t, http.StatusOK, map[string]any{
			"answers": map[string]any{},
			"prompt_feedback": map[string]any{
				"block_reason":         "prohibited_content",
				"block_reason_message": "The input contains prohibited content",
			},
		})

		response, err := model.Decide(t.Context(), blueCall())

		require.Nil(t, response)
		requireFantasyError(t, err, "content blocked", "prohibited_content: The input contains prohibited content")
	})

	t.Run("should report a blocked prompt without a message", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiDecisionModel(t, http.StatusOK, map[string]any{
			"prompt_feedback": map[string]any{"block_reason": "safety"},
		})

		_, err := model.Decide(t.Context(), blueCall())

		var fantasyErr *fantasy.Error
		require.ErrorAs(t, err, &fantasyErr)
		require.Equal(t, "content blocked", fantasyErr.Title)
		require.Equal(t, "safety", fantasyErr.Message)
	})
}

func TestDecisionModel_Response(t *testing.T) {
	t.Parallel()

	t.Run("should map all answer types", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(map[string]any{
			"safe":   noulAnswerBody(0.03, 0.97),
			"animal": choiceAnswerBody("cat", map[string]float64{"cat": 0.8, "dog": 0.2}, 0.8),
			"quality": scoreAnswerBody(1.2, map[string]float64{"0": 0.1, "1": 0.6, "2": 0.3}, map[string]string{
				"0": "the server's text for level 0",
				"1": "the server's text for level 1",
				"2": "the server's text for level 2",
			}, 0.6),
			"unrequested": noulAnswerBody(0.5, 0.5),
		}, nil))

		response, err := model.Decide(t.Context(), questionCall(map[string]fantasy.Question{
			"safe": fantasy.NoulQuestion{Instructions: fantasy.String("Is it safe?")},
			"animal": fantasy.ChoiceQuestion{
				Instructions: fantasy.String("Which animal?"),
				Options:      []fantasy.ChoiceOption{{Name: "cat"}, {Name: "dog"}},
			},
			"quality": fantasy.ScoreQuestion{
				Instructions: fantasy.String("How good?"),
				Levels: []fantasy.ScoreLevel{
					{Label: "Bad", Description: fantasy.String("Unusable")},
					{Label: "OK"},
					{Label: "Good", Description: fantasy.String("Great work")},
				},
			},
		}))
		require.NoError(t, err)

		require.Equal(t, "gemini-decision-1", response.Model)
		require.Len(t, response.Answers, 3)
		require.Empty(t, response.Warnings)
		require.Nil(t, response.ProviderMetadata)

		noul, ok := response.Noul("safe")
		require.True(t, ok)
		require.InDelta(t, 0.03, noul.Probability, 1e-9)
		require.NotNil(t, noul.Confidence)
		require.InDelta(t, 0.97, *noul.Confidence, 1e-9)

		choice, ok := response.Choice("animal")
		require.True(t, ok)
		require.Equal(t, "cat", choice.Choice)
		require.Equal(t, map[string]float64{"cat": 0.8, "dog": 0.2}, choice.Probabilities)
		require.NotNil(t, choice.Confidence)
		require.InDelta(t, 0.8, *choice.Confidence, 1e-9)

		score, ok := response.Score("quality")
		require.True(t, ok)
		require.InDelta(t, 1.2, score.Score, 1e-9)
		require.Equal(t, map[int]float64{0: 0.1, 1: 0.6, 2: 0.3}, score.Probabilities)
		require.Equal(t, map[int]string{0: "Bad", 1: "OK", 2: "Good"}, score.Legend)
		require.NotNil(t, score.Confidence)
		require.InDelta(t, 0.6, *score.Confidence, 1e-9)
	})

	t.Run("should map usage", func(t *testing.T) {
		t.Parallel()

		for _, tc := range []struct {
			name  string
			usage map[string]any
			want  fantasy.Usage
		}{
			{
				name: "all fields",
				usage: map[string]any{
					"total_input_tokens":   100,
					"total_output_tokens":  0,
					"total_tokens":         130,
					"total_cached_tokens":  20,
					"total_thought_tokens": 30,
				},
				want: fantasy.Usage{InputTokens: 100, OutputTokens: 30, TotalTokens: 130, ReasoningTokens: 30, CacheReadTokens: 20},
			},
			{
				name:  "output already includes thoughts",
				usage: map[string]any{"total_input_tokens": 100, "total_output_tokens": 40, "total_thought_tokens": 30, "total_tokens": 140},
				want:  fantasy.Usage{InputTokens: 100, OutputTokens: 40, ReasoningTokens: 30, TotalTokens: 140},
			},
			{
				name:  "tool prompt counts in total when folding thoughts into output",
				usage: map[string]any{"total_input_tokens": 100, "total_tool_use_tokens": 5, "total_output_tokens": 10, "total_thought_tokens": 30, "total_tokens": 145},
				want:  fantasy.Usage{InputTokens: 100, OutputTokens: 40, ReasoningTokens: 30, TotalTokens: 145},
			},
			{
				name:  "partial metadata uses the Developer API inclusive fallback",
				usage: map[string]any{"total_output_tokens": 40, "total_thought_tokens": 30},
				want:  fantasy.Usage{OutputTokens: 40, ReasoningTokens: 30},
			},
			{
				name:  "missing fields count as zero",
				usage: map[string]any{"total_tokens": 7},
				want:  fantasy.Usage{TotalTokens: 7},
			},
			{
				name:  "empty usage",
				usage: map[string]any{},
				want:  fantasy.Usage{},
			},
			{
				name: "no usage",
				want: fantasy.Usage{},
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				body := decisionResponseBody(map[string]any{"blue": noulAnswerBody(0.9, 0.9)}, tc.usage)
				model, _ := newGeminiDecisionModel(t, http.StatusOK, body)

				response, err := model.Decide(t.Context(), blueCall())
				require.NoError(t, err)

				require.Equal(t, tc.want, response.Usage)
			})
		}
	})

	t.Run("should fail with an invalid response", func(t *testing.T) {
		t.Parallel()

		for _, tc := range []struct {
			name    string
			answers map[string]any
			message string
		}{
			{
				name:    "answer missing",
				answers: map[string]any{"other": noulAnswerBody(0.5, 0.5)},
				message: `no answer for question "blue"`,
			},
			{
				name:    "no answers at all",
				answers: map[string]any{},
				message: `no answer for question "blue"`,
			},
			{
				name:    "unknown answer type",
				answers: map[string]any{"blue": map[string]any{"type": "refusal"}},
				message: `answer to question "blue" has an unknown type`,
			},
			{
				name:    "answer of another type",
				answers: map[string]any{"blue": choiceAnswerBody("cat", map[string]float64{"cat": 1}, 1)},
				message: `answer to question "blue" is of type "choice", want "noul"`,
			},
		} {
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()

				model, _ := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(tc.answers, nil))

				response, err := model.Decide(t.Context(), blueCall())

				require.Nil(t, response)
				requireFantasyError(t, err, "invalid response", tc.message)
			})
		}
	})

	t.Run("should fail when a score level is not a number", func(t *testing.T) {
		t.Parallel()

		model, _ := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(map[string]any{
			"quality": scoreAnswerBody(1, map[string]float64{"0": 0.5, "high": 0.5}, map[string]string{}, 0.5),
		}, nil))

		_, err := model.Decide(t.Context(), questionCall(map[string]fantasy.Question{
			"quality": fantasy.ScoreQuestion{Instructions: fantasy.String("How good?"), Levels: scoreLevels(2)},
		}))

		requireFantasyError(t, err, "invalid response", `level "high"`)
	})
}

func TestDecisionModel_InvalidAnswerValues(t *testing.T) {
	t.Parallel()
	noul := fantasy.NoulQuestion{Instructions: fantasy.String("Is it true?")}
	choice := fantasy.ChoiceQuestion{Instructions: fantasy.String("Pick one"), Options: []fantasy.ChoiceOption{{Name: "cat"}, {Name: "dog"}}}
	score := fantasy.ScoreQuestion{Instructions: fantasy.String("Rate it"), Levels: scoreLevels(2)}
	for _, tc := range []struct {
		name     string
		question fantasy.Question
		answer   any
	}{
		{"noul with only a discriminator", noul, map[string]any{"type": "noul"}},
		{"missing noul", noul, map[string]any{"type": "noul", "confidence": 1}},
		{"null noul", noul, map[string]any{"type": "noul", "confidence": 1, "noul": nil}},
		{"missing confidence", noul, map[string]any{"type": "noul", "noul": 0}},
		{"noul below zero", noul, noulAnswerBody(-0.1, 1)},
		{"confidence above one", noul, noulAnswerBody(0.5, 1.01)},
		{"confidence overflow", noul, json.RawMessage(`{"type":"noul","noul":0.5,"confidence":1e999}`)},
		{"unknown choice winner", choice, choiceAnswerBody("bird", map[string]float64{"cat": 1}, 1)},
		{"unknown choice key", choice, choiceAnswerBody("cat", map[string]float64{"bird": 1}, 1)},
		{"choice probability above one", choice, choiceAnswerBody("cat", map[string]float64{"cat": 1.01}, 1)},
		{"choice confidence below zero", choice, choiceAnswerBody("cat", map[string]float64{"cat": 1}, -0.1)},
		{"score negative level", score, scoreAnswerBody(0, map[string]float64{"-1": 1}, nil, 1)},
		{"score too high level", score, scoreAnswerBody(0, map[string]float64{"2": 1}, nil, 1)},
		{"score noncanonical level", score, scoreAnswerBody(0, map[string]float64{"01": 1}, nil, 1)},
		{"score overlapping levels", score, scoreAnswerBody(1, map[string]float64{"1": 0.5, "01": 0.5}, nil, 1)},
		{"score noncanonical legend level", score, scoreAnswerBody(1, map[string]float64{"1": 1}, map[string]string{"01": "high"}, 1)},
		{"score repeated identical key", score, json.RawMessage(`{"type":"score","score":1,"confidence":1,"probabilities":{"1":0.1,"1":0.9}}`)},
		{"score probability below zero", score, scoreAnswerBody(0, map[string]float64{"0": -0.1}, nil, 1)},
		{"null score probability", score, json.RawMessage(`{"type":"score","score":0,"confidence":1,"probabilities":{"0":null}}`)},
		{"non-finite score probability", score, json.RawMessage(`{"type":"score","score":0,"confidence":1,"probabilities":{"0":"NaN"}}`)},
		{"score confidence above one", score, scoreAnswerBody(0, map[string]float64{"0": 1}, nil, 1.1)},
		{"score outside rubric", score, scoreAnswerBody(2, map[string]float64{"1": 1}, nil, 1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			model, _ := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(map[string]any{"q": tc.answer}, nil))
			response, err := model.Decide(t.Context(), questionCall(map[string]fantasy.Question{"q": tc.question}))
			require.Nil(t, response)
			var target *fantasy.Error
			require.ErrorAs(t, err, &target)
			require.Equal(t, "invalid response", target.Title)
		})
	}
}

func TestDecisionModel_ExplicitZeroAndRoundedProbabilities(t *testing.T) {
	t.Parallel()
	model, _ := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(map[string]any{
		"blue":    noulAnswerBody(0, 0),
		"quality": scoreAnswerBody(0.5, map[string]float64{"0": 0.51, "1": 0.5}, nil, 0.51),
	}, nil))
	call := blueCall()
	call.Questions["quality"] = fantasy.ScoreQuestion{Instructions: fantasy.String("Rate it"), Levels: scoreLevels(2)}
	response, err := model.Decide(t.Context(), call)
	require.NoError(t, err)
	answer, ok := response.Noul("blue")
	require.True(t, ok)
	require.Equal(t, fantasy.NoulAnswer{Probability: 0, Confidence: new(0.0)}, answer)
	score, ok := response.Score("quality")
	require.True(t, ok)
	require.Equal(t, map[int]float64{0: 0.51, 1: 0.5}, score.Probabilities)
}

func TestDecisionModel_ProviderOptions(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name string
		opts fantasy.ProviderOptions
		warn bool
	}{
		{"other only", fantasy.ProviderOptions{"other": &ProviderOptions{}}, false},
		{"Google and other", fantasy.ProviderOptions{Name: &ProviderOptions{}, "other": &ProviderOptions{}}, true},
		{"Google key with nil value", fantasy.ProviderOptions{Name: nil}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			model, _ := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody(), WithName("my-google"))
			call := blueCall()
			call.ProviderOptions = tc.opts
			response, err := model.Decide(t.Context(), call)
			require.NoError(t, err)
			if !tc.warn {
				require.Empty(t, response.Warnings)
				return
			}
			require.Len(t, response.Warnings, 1)
			warning := response.Warnings[0]
			require.Equal(t, fantasy.CallWarningTypeUnsupportedSetting, warning.Type)
			require.Equal(t, "provider_options.google", warning.Setting)
			require.Contains(t, warning.Message, "do not support Google provider options")
		})
	}
}

func TestDecisionModel_ErrorHeaderValues(t *testing.T) {
	t.Parallel()
	err := toProviderErr(&apierrors.APIError{StatusCode: http.StatusTooManyRequests, RawResponse: &http.Response{
		Header: http.Header{"Retry-After": {"3", "7"}, "X-Empty": {}},
	}})
	var providerErr *fantasy.ProviderError
	require.ErrorAs(t, err, &providerErr)
	require.Equal(t, map[string]string{"Retry-After": "7", "retry-after": "7"}, providerErr.ResponseHeaders)
}

func TestDecisionModel_BackendSelection(t *testing.T) {
	for _, key := range []string{"GOOGLE_GENAI_USE_ENTERPRISE", "GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_API_VERSION", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_REGION", "GEMINI_API_KEY"} {
		t.Setenv(key, "")
		require.NoError(t, os.Unsetenv(key))
	}
	t.Setenv("GOOGLE_API_KEY", "env-key")
	for _, tc := range []struct {
		name                       string
		vertex, enterprise         *string
		explicitGemini, wantVertex bool
	}{
		{name: "explicit Gemini overrides environment Vertex", vertex: new("true"), explicitGemini: true},
		{name: "only GOOGLE_API_KEY"},
		{name: "Vertex from environment", vertex: new("TrUe"), wantVertex: true},
		{name: "enterprise selects Vertex", enterprise: new("1"), wantVertex: true},
		{name: "empty enterprise flag overrides Vertex", enterprise: new(""), vertex: new("true")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if tc.vertex != nil {
				t.Setenv("GOOGLE_GENAI_USE_VERTEXAI", *tc.vertex)
			}
			if tc.enterprise != nil {
				t.Setenv("GOOGLE_GENAI_USE_ENTERPRISE", *tc.enterprise)
			}
			server, recorder := newDecisionTestServer(t, http.StatusOK, blueResponseBody())
			opts := []Option{WithBaseURL(server.URL)}
			wantKey := "env-key"
			if tc.explicitGemini {
				wantKey = "explicit-key"
				opts = append(opts, WithGeminiAPIKey(wantKey))
			}
			p, err := New(opts...)
			require.NoError(t, err)
			model, err := p.(fantasy.DecisionModelProvider).DecisionModel(t.Context(), "")
			if tc.wantVertex {
				requireFantasyError(t, err, "unsupported", "Gemini Developer API")
				require.Nil(t, model)
				client, err := p.(*provider).newClient(t.Context())
				require.NoError(t, err)
				cfg := client.ClientConfig()
				require.Equal(t, genai.BackendVertexAI, cfg.Backend)
				require.Equal(t, "v1beta1", cfg.HTTPOptions.APIVersion)
				require.Empty(t, recorder.requests())
				return
			}
			require.NoError(t, err)
			_, err = model.Decide(t.Context(), blueCall())
			require.NoError(t, err)
			request := recorder.only(t)
			require.Equal(t, "/v1beta/decisions", request.path)
			require.Equal(t, wantKey, request.header.Get("X-Goog-Api-Key"))
		})
	}
}

func TestDecisionModel_OptionNames(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name, option string
		valid        bool
	}{
		{"empty", "", false},
		{"too many characters", strings.Repeat("a", 129), false},
		{"too many bytes", strings.Repeat("\U00020000", 128) + "a", false},
		{"leading space", " a", false},
		{"trailing space", "a ", false},
		{"newline", "a\nb", false},
		{"non-whitespace control", "a\u007fb", false},
		{"non-ASCII space", "a\u00a0b", false},
		{"invalid UTF-8", "a\xff", false},
		{"internal ASCII spaces", "é  b", true},
		{"exactly 128 characters and 512 bytes", strings.Repeat("\U00020000", 128), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			model, recorder := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(map[string]any{
				"q": choiceAnswerBody(tc.option, map[string]float64{tc.option: 1}, 1),
			}, nil))
			response, err := model.Decide(t.Context(), questionCall(map[string]fantasy.Question{
				"q": fantasy.ChoiceQuestion{Instructions: fantasy.String("Pick one"), Options: []fantasy.ChoiceOption{{Name: tc.option}, {Name: "other"}}},
			}))
			if tc.valid {
				require.NoError(t, err)
				criteria := recorder.only(t).body["questions"].(map[string]any)["q"].(map[string]any)["criteria"]
				require.Contains(t, criteria, tc.option)
				return
			}
			require.Empty(t, recorder.requests())
			require.Nil(t, response)
			requireFantasyError(t, err, "invalid argument", "option")
		})
	}
}

func TestDecisionModel_Vertex(t *testing.T) {
	t.Parallel()

	server, recorder := newDecisionTestServer(t, http.StatusOK, blueResponseBody())
	p, err := New(WithVertex("test-project", "us-central1"), WithBaseURL(server.URL), WithSkipAuth(true))
	require.NoError(t, err)
	decisionProvider, ok := p.(fantasy.DecisionModelProvider)
	require.True(t, ok)

	model, err := decisionProvider.DecisionModel(t.Context(), "")

	require.Nil(t, model)
	requireFantasyError(t, err, "unsupported", "only available on the Gemini Developer API")
	require.Empty(t, recorder.requests())
}

type customQuestion struct{}

func (customQuestion) GetType() fantasy.QuestionType { return "custom" }

func TestDecisionModel_Validation(t *testing.T) {
	t.Parallel()

	manyQuestions := func(n int) map[string]fantasy.Question {
		questions := make(map[string]fantasy.Question, n)
		for i := range n {
			questions["q"+strconv.Itoa(i)] = fantasy.NoulQuestion{Instructions: fantasy.String("Is it true?")}
		}
		return questions
	}
	named := func(name string) fantasy.DecisionCall {
		return questionCall(map[string]fantasy.Question{
			name: fantasy.NoulQuestion{Instructions: fantasy.String("Is it true?")},
		})
	}
	withInput := func(parts ...fantasy.MessagePart) fantasy.DecisionCall {
		call := blueCall()
		call.Input = parts
		return call
	}

	for _, tc := range []struct {
		name    string
		call    fantasy.DecisionCall
		title   string
		message string
	}{
		{
			name:    "no input",
			call:    withInput(),
			title:   "invalid argument",
			message: "input must have at least one part",
		},
		{
			name:    "empty single text input",
			call:    withInput(fantasy.TextPart{}),
			title:   "invalid argument",
			message: "input part 0",
		},
		{
			name:    "empty text in a multipart input",
			call:    withInput(fantasy.TextPart{Text: "ok"}, &fantasy.TextPart{}),
			title:   "invalid argument",
			message: "input part 1",
		},
		{
			name:    "custom question type",
			call:    questionCall(map[string]fantasy.Question{"custom": customQuestion{}}),
			title:   "unsupported",
			message: `question "custom"`,
		},
		{
			name: "built-in question passed by pointer",
			call: questionCall(map[string]fantasy.Question{
				"pointer": &fantasy.NoulQuestion{Instructions: fantasy.String("Is it true?")},
			}),
			title:   "invalid argument",
			message: `question "pointer"`,
		},
		{
			name: "one choice option",
			call: questionCall(map[string]fantasy.Question{
				"choice": fantasy.ChoiceQuestion{Instructions: fantasy.String("Pick one"), Options: choiceOptions(1)},
			}),
			title:   "invalid argument",
			message: `question "choice": choice questions need 2 to 255 options, got 1`,
		},
		{
			name: "too many choice options",
			call: questionCall(map[string]fantasy.Question{
				"choice": fantasy.ChoiceQuestion{Instructions: fantasy.String("Pick one"), Options: choiceOptions(256)},
			}),
			title:   "invalid argument",
			message: `question "choice": choice questions need 2 to 255 options, got 256`,
		},
		{
			name: "one score level",
			call: questionCall(map[string]fantasy.Question{
				"score": fantasy.ScoreQuestion{Instructions: fantasy.String("Rate it"), Levels: scoreLevels(1)},
			}),
			title:   "invalid argument",
			message: `question "score": score questions need 2 to 255 levels, got 1`,
		},
		{
			name: "too many score levels",
			call: questionCall(map[string]fantasy.Question{
				"score": fantasy.ScoreQuestion{Instructions: fantasy.String("Rate it"), Levels: scoreLevels(256)},
			}),
			title:   "invalid argument",
			message: `question "score": score questions need 2 to 255 levels, got 256`,
		},
		{
			name: "empty noul instructions",
			call: questionCall(map[string]fantasy.Question{
				"noul": fantasy.NoulQuestion{Instructions: fantasy.String("")},
			}),
			title:   "invalid argument",
			message: `question "noul": instructions must not be empty`,
		},
		{
			name: "missing choice instructions",
			call: questionCall(map[string]fantasy.Question{
				"choice": fantasy.ChoiceQuestion{Options: choiceOptions(2)},
			}),
			title:   "invalid argument",
			message: `question "choice": instructions must not be empty`,
		},
		{
			name: "empty score instructions",
			call: questionCall(map[string]fantasy.Question{
				"score": fantasy.ScoreQuestion{Instructions: fantasy.String(""), Levels: scoreLevels(2)},
			}),
			title:   "invalid argument",
			message: `question "score": instructions must not be empty`,
		},
		{
			name:    "question name with a space",
			call:    named("bad name"),
			title:   "invalid argument",
			message: `question name "bad name"`,
		},
		{
			name:    "question name that is too long",
			call:    named(strings.Repeat("a", 65)),
			title:   "invalid argument",
			message: "question name",
		},
		{
			name:    "question name with a non-ascii letter",
			call:    named("naïve"),
			title:   "invalid argument",
			message: `question name "naïve"`,
		},
		{
			name:    "too many questions",
			call:    questionCall(manyQuestions(1001)),
			title:   "invalid argument",
			message: "at most 1000 questions",
		},
		{
			name:    "unsupported image type",
			call:    withInput(fantasy.FilePart{Data: []byte("picture-bytes"), MediaType: "image/svg+xml"}),
			title:   "invalid argument",
			message: `image media type "image/svg+xml" is not supported`,
		},
		{
			name:    "file that is not an image",
			call:    withInput(fantasy.FilePart{Data: []byte("document-bytes"), MediaType: "application/pdf"}),
			title:   "invalid argument",
			message: "files must be images",
		},
	} {
		t.Run("should reject "+tc.name+" before sending a request", func(t *testing.T) {
			t.Parallel()

			model, recorder := newGeminiDecisionModel(t, http.StatusOK, blueResponseBody())

			response, err := model.Decide(t.Context(), tc.call)

			require.Nil(t, response)
			requireFantasyError(t, err, tc.title, tc.message)
			require.Empty(t, recorder.requests())
		})
	}

	t.Run("should accept the largest supported questions", func(t *testing.T) {
		t.Parallel()

		longChoice := "a" + strings.Repeat("B-", 31) + "9"
		require.Len(t, longChoice, 64)
		questions := manyQuestions(998)
		questions[longChoice] = fantasy.ChoiceQuestion{Instructions: fantasy.String("Pick one"), Options: choiceOptions(255)}
		questions["score"] = fantasy.ScoreQuestion{Instructions: fantasy.String("Rate it"), Levels: scoreLevels(255)}

		answers := make(map[string]any, len(questions))
		for name := range questions {
			answers[name] = noulAnswerBody(0.5, 0.5)
		}
		answers[longChoice] = choiceAnswerBody("option0", map[string]float64{"option0": 1}, 1)
		answers["score"] = scoreAnswerBody(0, map[string]float64{"0": 1}, map[string]string{}, 1)
		model, recorder := newGeminiDecisionModel(t, http.StatusOK, decisionResponseBody(answers, nil))

		response, err := model.Decide(t.Context(), questionCall(questions))
		require.NoError(t, err)

		require.Len(t, response.Answers, 1000)
		require.Len(t, recorder.only(t).body["questions"], 1000)
	})
}
