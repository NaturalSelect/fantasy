package fantasy

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

type customQuestion struct {
	Threshold int `json:"threshold"`
}

func (customQuestion) GetType() QuestionType { return "test.custom" }

type customAnswer struct {
	Result bool `json:"result"`
}

func (customAnswer) GetType() QuestionType { return "test.custom" }

func init() {
	RegisterQuestionType("test.custom", func(data []byte) (Question, error) {
		var q customQuestion
		err := json.Unmarshal(data, &q)
		return q, err
	})
	RegisterAnswerType("test.custom", func(data []byte) (Answer, error) {
		var a customAnswer
		err := json.Unmarshal(data, &a)
		return a, err
	})
}

func TestDecisionCall_JSON(t *testing.T) {
	t.Parallel()

	want := validDecisionCall()
	want.Input = append(want.Input, FilePart{MediaType: "image/png", Data: []byte{42}})
	want.ProviderOptions = ProviderOptions{"test": new(roundTripProviderData("options"))}
	want.Questions["likely"] = NoulQuestion{
		Instructions:     String("likely?"),
		TrueDescription:  String("yes"),
		FalseDescription: String(""),
	}
	want.Questions["rating"] = ScoreQuestion{
		Instructions: String("rate"),
		Levels:       []ScoreLevel{{Label: "low"}, {Label: "high", Description: String("urgent")}},
	}
	require.Equal(t, want, roundTripJSON(t, want))
}

func TestDecisionResponse_JSON(t *testing.T) {
	t.Parallel()

	want := DecisionResponse{
		Model:    "test",
		Warnings: []CallWarning{{Message: "note", Tool: FunctionTool{Name: "tool"}}},
		Usage:    Usage{InputTokens: 4, OutputTokens: 2},
		Answers: map[string]Answer{
			"likely": NoulAnswer{Probability: 0.8, Confidence: new(0.0)},
			"tone":   ChoiceAnswer{Choice: "calm", Probabilities: map[string]float64{"calm": 0.9, "angry": 0.1}},
			"rating": ScoreAnswer{
				Score:         0.3,
				Probabilities: map[int]float64{0: 0.7, 1: 0.3},
				Legend:        map[int]string{0: "low", 1: "high"},
				Confidence:    new(0.91),
			},
			"declined":             RefusalAnswer{},
			"tone-with-confidence": ChoiceAnswer{Choice: "angry", Confidence: new(0.5)},
			"no-score-confidence":  ScoreAnswer{},
			"no-noul-confidence":   NoulAnswer{},
		},
	}
	for _, metadata := range []ProviderMetadata{nil, {}, {"test": new(roundTripProviderData("metadata"))}} {
		want.ProviderMetadata = metadata
		require.Equal(t, want, roundTripJSON(t, want))
	}
}

func TestDecisionJSON_CustomTypes(t *testing.T) {
	t.Parallel()

	call := DecisionCall{Questions: map[string]Question{"q": customQuestion{Threshold: 17}}}
	encoded, err := json.Marshal(call)
	require.NoError(t, err)
	require.JSONEq(t, `{"input":null,"questions":{"q":{"type":"test.custom","data":{"threshold":17}}},"provider_options":null}`, string(encoded))
	require.Equal(t, call, roundTripJSON(t, call))

	response := DecisionResponse{Answers: map[string]Answer{"q": customAnswer{Result: true}}}
	encoded, err = json.Marshal(response)
	require.NoError(t, err)
	var raw struct {
		Answers json.RawMessage `json:"answers"`
	}
	require.NoError(t, json.Unmarshal(encoded, &raw))
	require.JSONEq(t, `{"q":{"type":"test.custom","data":{"result":true}}}`, string(raw.Answers))
	require.Equal(t, response, roundTripJSON(t, response))
}

func TestDecisionJSON_UnknownTypes(t *testing.T) {
	t.Parallel()

	err := json.Unmarshal([]byte(`{"questions":{"q":{"type":"unregistered","data":{}}}}`), new(DecisionCall))
	require.ErrorContains(t, err, "unknown question type: unregistered")
	err = json.Unmarshal([]byte(`{"answers":{"q":{"type":"unregistered","data":{}}}}`), new(DecisionResponse))
	require.ErrorContains(t, err, "unknown answer type: unregistered")
}

func TestDecisionJSON_RenderedText(t *testing.T) {
	t.Parallel()

	text := JSONText([]string{"a", "b"})
	want := String(`["a","b"]`)
	choice := roundTripJSON(t, ChoiceQuestion{Instructions: text, Options: []ChoiceOption{{Description: text}}})
	require.Equal(t, ChoiceQuestion{Instructions: want, Options: []ChoiceOption{{Description: want}}}, choice)

	score := roundTripJSON(t, ScoreQuestion{Instructions: text, Levels: []ScoreLevel{{Description: text}}})
	require.Equal(t, ScoreQuestion{Instructions: want, Levels: []ScoreLevel{{Description: want}}}, score)

	noul := roundTripJSON(t, NoulQuestion{Instructions: text, TrueDescription: text, FalseDescription: text})
	require.Equal(t, NoulQuestion{Instructions: want, TrueDescription: want, FalseDescription: want}, noul)
}

func TestDecisionJSON_NilText(t *testing.T) {
	t.Parallel()

	for _, absent := range []Text{nil, (*String)(nil)} {
		q := NoulQuestion{TrueDescription: absent, FalseDescription: String("")}
		encoded, err := json.Marshal(q)
		require.NoError(t, err)
		require.JSONEq(t, `{"instructions":null,"true_description":null,"false_description":""}`, string(encoded))
		require.Equal(t, NoulQuestion{FalseDescription: String("")}, roundTripJSON(t, q))
	}
}

func TestDecisionJSON_RenderError(t *testing.T) {
	t.Parallel()

	for _, q := range []Question{
		NoulQuestion{TrueDescription: failingText{}},
		ChoiceQuestion{Options: []ChoiceOption{{Description: failingText{}}}},
		ScoreQuestion{Levels: []ScoreLevel{{Description: failingText{}}}},
	} {
		_, err := json.Marshal(DecisionCall{Questions: map[string]Question{"q": q}})
		require.ErrorContains(t, err, "boom")
	}
}
