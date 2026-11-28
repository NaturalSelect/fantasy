package typesafe

import (
	"net/http"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func allQuestionsCall() fantasy.DecisionCall {
	return fantasy.DecisionCall{
		Input: []fantasy.MessagePart{fantasy.TextPart{Text: "an email"}, pngPart("logo")},
		Questions: map[string]fantasy.Question{
			"tone": fantasy.ChoiceQuestion{
				Instructions: fantasy.String("What is the tone?"),
				Options:      []fantasy.ChoiceOption{{Name: "angry"}, {Name: "calm"}, {Name: "excited"}},
			},
			"urgency": fantasy.ScoreQuestion{
				Instructions: fantasy.String("How urgent?"),
				Levels: []fantasy.ScoreLevel{
					{Label: "Can wait", Description: fantasy.String("no rush")},
					{Label: "Soon"},
					{Label: "Now", Description: fantasy.String("drop everything")},
				},
			},
			"spam": fantasy.NoulQuestion{Instructions: fantasy.String("Is it spam?")},
			"logo": BoxQuestion{Instructions: fantasy.String("Find the logo")},
		},
	}
}

func TestDecide_Answers(t *testing.T) {
	t.Parallel()

	t.Run("all question types", func(t *testing.T) {
		t.Parallel()

		model, _ := newTestModel(t, respond(http.StatusOK, `{
			"model": "jev-1.14.0",
			"answers": {
				"tone": {"type": "choice", "choice": "calm", "confidence": 0.7, "probabilities": {"angry": 0.15, "calm": 0.8, "excited": 0.05}},
				"urgency": {
					"type": "score", "score": 1.7, "confidence": 0.55,
					"legend": {"0": "server zero", "1": "server one", "2": "server two"},
					"probabilities": {"0": 0.1, "1": 0.1, "2": 0.8}
				},
				"spam": {"type": "noul", "noul": 0.98},
				"logo": {"type": "box", "box": {"top": 0.2, "bottom": 0.66, "left": 0.19, "right": 0.68}, "confidence": null}
			},
			"usage": {"input_tokens": 120, "output_tokens": 12}
		}`, "Content-Type", "application/json", "X-Typesafe-Request-Id", "req_123"))

		response, err := model.Decide(t.Context(), allQuestionsCall())
		require.NoError(t, err)

		require.Equal(t, "jev-1.14.0", response.Model)
		require.Len(t, response.Answers, 4)
		require.Empty(t, response.Warnings)

		confidence := 0.7
		require.Equal(t, fantasy.ChoiceAnswer{
			Choice:        "calm",
			Probabilities: map[string]float64{"angry": 0.15, "calm": 0.8, "excited": 0.05},
			Confidence:    &confidence,
		}, response.Answers["tone"])

		confidence = 0.55
		require.Equal(t, fantasy.ScoreAnswer{
			Score:         1.7,
			Probabilities: map[int]float64{0: 0.1, 1: 0.1, 2: 0.8},
			Legend:        map[int]string{0: "Can wait", 1: "Soon", 2: "Now"},
			Confidence:    &confidence,
		}, response.Answers["urgency"])

		require.Equal(t, fantasy.NoulAnswer{Probability: 0.98}, response.Answers["spam"])

		box, ok := fantasy.AsAnswer[BoxAnswer](response.Answers["logo"])
		require.True(t, ok)
		require.Equal(t, BoxAnswer{Top: 0.2, Bottom: 0.66, Left: 0.19, Right: 0.68}, box)

		noul, ok := response.Noul("spam")
		require.True(t, ok)
		require.Equal(t, 0.98, noul.Probability)
		choice, ok := response.Choice("tone")
		require.True(t, ok)
		require.Equal(t, "calm", choice.Choice)
		score, ok := response.Score("urgency")
		require.True(t, ok)
		require.Equal(t, 1.7, score.Score)
	})

	t.Run("noul has no confidence", func(t *testing.T) {
		t.Parallel()

		model, _ := newTestModel(t, respond(http.StatusOK, `{"model":"jev-latest","answers":{"q":{"type":"noul","noul":0.98}}}`))
		response, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)

		noul, ok := response.Noul("q")
		require.True(t, ok)
		require.Nil(t, noul.Confidence)
		require.Equal(t, "jev-latest", response.Model, "an alias is passed through")
	})

	t.Run("box confidence", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name       string
			confidence string
			want       *float64
		}{
			{name: "null", confidence: `"confidence": null`, want: nil},
			{name: "missing", confidence: `"unrelated": 1`, want: nil},
			{name: "number", confidence: `"confidence": 0.4`, want: new(0.4)},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()

				model, _ := newTestModel(t, respond(http.StatusOK,
					`{"model":"m","answers":{"q":{"type":"box","box":{"top":0,"bottom":1,"left":0,"right":1},`+tt.confidence+`}}}`))
				call := imageCall(BoxQuestion{})
				response, err := model.Decide(t.Context(), call)
				require.NoError(t, err)

				box, ok := fantasy.AsAnswer[BoxAnswer](response.Answers["q"])
				require.True(t, ok)
				require.Equal(t, tt.want, box.Confidence)
			})
		}
	})

	t.Run("usage", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name  string
			usage string
			want  fantasy.Usage
		}{
			{name: "both counts", usage: `"usage":{"input_tokens":120,"output_tokens":12}`, want: fantasy.Usage{InputTokens: 120, OutputTokens: 12, TotalTokens: 132}},
			{name: "usage missing", usage: `"other":1`, want: fantasy.Usage{}},
			{name: "usage null", usage: `"usage":null`, want: fantasy.Usage{}},
			{name: "usage empty", usage: `"usage":{}`, want: fantasy.Usage{}},
			{name: "count null", usage: `"usage":{"input_tokens":null,"output_tokens":7}`, want: fantasy.Usage{OutputTokens: 7, TotalTokens: 7}},
			{name: "count missing", usage: `"usage":{"input_tokens":5}`, want: fantasy.Usage{InputTokens: 5, TotalTokens: 5}},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()

				model, _ := newTestModel(t, respond(http.StatusOK,
					`{"model":"m","answers":{"q":{"type":"noul","noul":0.1}},`+tt.usage+`}`))
				response, err := model.Decide(t.Context(), noulCall())
				require.NoError(t, err)
				require.Equal(t, tt.want, response.Usage)
			})
		}
	})

	t.Run("request id metadata", func(t *testing.T) {
		t.Parallel()

		body := `{"model":"m","answers":{"q":{"type":"noul","noul":0.1}}}`

		model, _ := newTestModel(t, respond(http.StatusOK, body, "x-typesafe-request-id", "req_abc"))
		response, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.Equal(t, fantasy.ProviderMetadata{Name: &ProviderMetadata{RequestID: "req_abc"}}, response.ProviderMetadata)

		model, _ = newTestModel(t, respond(http.StatusOK, body))
		response, err = model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.Empty(t, response.ProviderMetadata, "no metadata entry without a request id")
	})

	t.Run("extra answers are ignored", func(t *testing.T) {
		t.Parallel()

		model, _ := newTestModel(t, respond(http.StatusOK,
			`{"model":"m","answers":{"q":{"type":"noul","noul":0.1},"unasked":{"type":"noul","noul":0.9}}}`))
		response, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.Len(t, response.Answers, 1)
	})
}

func TestDecide_InvalidResponse(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		call fantasy.DecisionCall
		body string
		want string
	}{
		{name: "missing answer", call: noulCall(), body: `{"model":"m","answers":{}}`, want: `no answer for question "q"`},
		{name: "answers missing", call: noulCall(), body: `{"model":"m"}`, want: `no answer for question "q"`},
		{name: "null answer", call: noulCall(), body: `{"model":"m","answers":{"q":null}}`, want: `has type ""`},
		{name: "unknown type", call: noulCall(), body: `{"model":"m","answers":{"q":{"type":"banana","noul":0.5}}}`, want: `has type "banana", expected "noul"`},
		{name: "mismatched type", call: noulCall(), body: `{"model":"m","answers":{"q":{"type":"choice","choice":"a"}}}`, want: `has type "choice", expected "noul"`},
		{name: "refusal is not expected", call: noulCall(), body: `{"model":"m","answers":{"q":{"type":"refusal"}}}`, want: `has type "refusal"`},
		{name: "noul value missing", call: noulCall(), body: `{"model":"m","answers":{"q":{"type":"noul"}}}`, want: "no noul value"},
		{
			name: "choice missing",
			call: textCall(map[string]fantasy.Question{"q": fantasy.ChoiceQuestion{Options: []fantasy.ChoiceOption{{Name: "a"}}}}),
			body: `{"model":"m","answers":{"q":{"type":"choice"}}}`,
			want: "no choice",
		},
		{
			name: "score missing",
			call: textCall(map[string]fantasy.Question{"q": fantasy.ScoreQuestion{Levels: []fantasy.ScoreLevel{{Label: "a"}}}}),
			body: `{"model":"m","answers":{"q":{"type":"score"}}}`,
			want: "no score",
		},
		{
			name: "score level is not a number",
			call: textCall(map[string]fantasy.Question{"q": fantasy.ScoreQuestion{Levels: []fantasy.ScoreLevel{{Label: "a"}}}}),
			body: `{"model":"m","answers":{"q":{"type":"score","score":0,"probabilities":{"low":1}}}}`,
			want: `level "low"`,
		},
		{
			name: "box missing",
			call: imageCall(BoxQuestion{}),
			body: `{"model":"m","answers":{"q":{"type":"box"}}}`,
			want: "no box",
		},
		{name: "not json", call: noulCall(), body: `not json`, want: "decode response"},
		{name: "wrong field type", call: noulCall(), body: `{"model":"m","answers":{"q":{"type":"noul","noul":"high"}}}`, want: `answer "q"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, _ := newTestModel(t, respond(http.StatusOK, tt.body))
			response, err := model.Decide(t.Context(), tt.call)
			require.Nil(t, response)

			var fantasyErr *fantasy.Error
			require.ErrorAs(t, err, &fantasyErr)
			require.Equal(t, "invalid response", fantasyErr.Title)
			require.Contains(t, fantasyErr.Message, tt.want)
		})
	}
}
