package fantasy

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

type temperature string

type boxQuestion struct{}

func (boxQuestion) GetType() QuestionType { return "box" }

type boxAnswer struct{ Top float64 }

func (boxAnswer) GetType() QuestionType { return "box" }

type failingText struct{}

func (failingText) DecisionText() (string, error) { return "", errors.New("boom") }

func validDecisionCall() DecisionCall {
	return DecisionCall{
		Input: []MessagePart{TextPart{Text: "hello"}},
		Questions: map[string]Question{
			"tone": ChoiceQuestion{
				Instructions: String("What is the tone?"),
				Options:      []ChoiceOption{{Name: "calm"}, {Name: "angry", Description: String("hostile")}},
			},
		},
	}
}

func TestText(t *testing.T) {
	t.Parallel()

	t.Run("string", func(t *testing.T) {
		t.Parallel()
		text, err := RenderText(String("plain"))
		require.NoError(t, err)
		require.Equal(t, "plain", text)
	})

	t.Run("nil renders empty", func(t *testing.T) {
		t.Parallel()
		text, err := RenderText(nil)
		require.NoError(t, err)
		require.Empty(t, text)
	})

	t.Run("custom string type", func(t *testing.T) {
		t.Parallel()
		text, err := RenderText(TextOf(temperature("hot")))
		require.NoError(t, err)
		require.Equal(t, "hot", text)
	})

	t.Run("json", func(t *testing.T) {
		t.Parallel()
		text, err := RenderText(JSONText(map[string]any{"b": 1, "a": []string{"x"}}))
		require.NoError(t, err)
		require.JSONEq(t, `{"a":["x"],"b":1}`, text)
		require.NotContains(t, text, " ")
	})

	t.Run("json marshal error", func(t *testing.T) {
		t.Parallel()
		_, err := RenderText(JSONText(make(chan int)))
		require.Error(t, err)
	})
}

func TestRenderScoreLevel(t *testing.T) {
	t.Parallel()

	text, err := RenderScoreLevel(ScoreLevel{Label: "low"})
	require.NoError(t, err)
	require.Equal(t, "low", text)

	text, err = RenderScoreLevel(ScoreLevel{Label: "high", Description: String("needs attention today")})
	require.NoError(t, err)
	require.Equal(t, "high: needs attention today", text)

	_, err = RenderScoreLevel(ScoreLevel{Label: "x", Description: failingText{}})
	require.Error(t, err)
}

func TestDecisionCall_Validate(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		mutate  func(*DecisionCall)
		wantErr string
	}{
		{name: "valid", mutate: func(*DecisionCall) {}},
		{
			name: "image input",
			mutate: func(c *DecisionCall) {
				c.Input = append(c.Input, &FilePart{MediaType: "image/png", Data: []byte{1}})
			},
		},
		{
			name: "unknown question type is left to the provider",
			mutate: func(c *DecisionCall) {
				c.Questions["box"] = boxQuestion{}
			},
		},
		{
			name:    "no input",
			mutate:  func(c *DecisionCall) { c.Input = nil },
			wantErr: "input must have at least one part",
		},
		{
			name:    "nil input part",
			mutate:  func(c *DecisionCall) { c.Input = []MessagePart{nil} },
			wantErr: "input part 0: part is nil",
		},
		{
			name: "non-image file",
			mutate: func(c *DecisionCall) {
				c.Input = append(c.Input, FilePart{MediaType: "application/pdf", Data: []byte{1}})
			},
			wantErr: "input part 1: files must be images",
		},
		{
			name: "image without data",
			mutate: func(c *DecisionCall) {
				c.Input = append(c.Input, FilePart{MediaType: "image/png"})
			},
			wantErr: "input part 1: image has no data",
		},
		{
			name: "unsupported part type",
			mutate: func(c *DecisionCall) {
				c.Input = append(c.Input, ReasoningPart{Text: "hmm"})
			},
			wantErr: "input part 1: unsupported part type",
		},
		{
			name:    "no questions",
			mutate:  func(c *DecisionCall) { c.Questions = nil },
			wantErr: "at least one question is required",
		},
		{
			name:    "empty question name",
			mutate:  func(c *DecisionCall) { c.Questions[""] = NoulQuestion{} },
			wantErr: "question names must not be empty",
		},
		{
			name:    "nil question",
			mutate:  func(c *DecisionCall) { c.Questions["x"] = nil },
			wantErr: `question "x": question is nil`,
		},
		{
			name: "choice without options",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = ChoiceQuestion{}
			},
			wantErr: `question "x": at least one option is required`,
		},
		{
			name: "choice option without name",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = ChoiceQuestion{Options: []ChoiceOption{{Name: "a"}, {}}}
			},
			wantErr: `question "x": option 1 has no name`,
		},
		{
			name: "choice duplicate option",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = ChoiceQuestion{Options: []ChoiceOption{{Name: "a"}, {Name: "a"}}}
			},
			wantErr: `question "x": option name "a" is used more than once`,
		},
		{
			name: "choice description fails to render",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = ChoiceQuestion{Options: []ChoiceOption{{Name: "a", Description: failingText{}}}}
			},
			wantErr: `question "x": option "a": boom`,
		},
		{
			name: "score without levels",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = ScoreQuestion{}
			},
			wantErr: `question "x": at least one level is required`,
		},
		{
			name: "score level without label",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = ScoreQuestion{Levels: []ScoreLevel{{Label: "a"}, {}}}
			},
			wantErr: `question "x": level 1 has no label`,
		},
		{
			name: "score description fails to render",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = ScoreQuestion{Levels: []ScoreLevel{{Label: "a", Description: failingText{}}}}
			},
			wantErr: `question "x": level 0: boom`,
		},
		{
			name: "noul instructions fail to render",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = NoulQuestion{Instructions: failingText{}}
			},
			wantErr: `question "x": boom`,
		},
		{
			name: "noul criteria fail to render",
			mutate: func(c *DecisionCall) {
				c.Questions["x"] = NoulQuestion{FalseDescription: failingText{}}
			},
			wantErr: `question "x": boom`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			call := validDecisionCall()
			tt.mutate(&call)
			err := call.Validate()
			if tt.wantErr == "" {
				require.NoError(t, err)
				return
			}
			require.ErrorContains(t, err, tt.wantErr)
			var fantasyErr *Error
			require.ErrorAs(t, err, &fantasyErr)
			require.Equal(t, "invalid argument", fantasyErr.Title)
		})
	}
}

func TestDecisionCall_ValidateReportsFirstBadQuestionByName(t *testing.T) {
	t.Parallel()

	call := validDecisionCall()
	call.Questions["b"] = ChoiceQuestion{}
	call.Questions["a"] = ScoreQuestion{}

	// NOTE: Questions are checked in name order so the error is deterministic.
	require.ErrorContains(t, call.Validate(), `question "a"`)
}

func TestDecisionResponse_TypedAccessors(t *testing.T) {
	t.Parallel()

	confidence := 0.7
	response := &DecisionResponse{
		Answers: map[string]Answer{
			"billing": NoulAnswer{Probability: 0.98},
			"tone":    ChoiceAnswer{Choice: "calm", Probabilities: map[string]float64{"calm": 0.8}, Confidence: &confidence},
			"urgency": ScoreAnswer{Score: 1.7, Probabilities: map[int]float64{1: 0.8}, Legend: map[int]string{1: "mid"}},
			"pointer": &NoulAnswer{Probability: 0.5},
			"refused": RefusalAnswer{},
			"logo":    boxAnswer{Top: 0.2},
		},
	}

	noul, ok := response.Noul("billing")
	require.True(t, ok)
	require.InDelta(t, 0.98, noul.Probability, 1e-9)
	require.Nil(t, noul.Confidence)

	choice, ok := response.Choice("tone")
	require.True(t, ok)
	require.Equal(t, "calm", choice.Choice)
	require.InDelta(t, 0.7, *choice.Confidence, 1e-9)

	score, ok := response.Score("urgency")
	require.True(t, ok)
	require.InDelta(t, 1.7, score.Score, 1e-9)
	require.Equal(t, "mid", score.Legend[1])

	pointer, ok := response.Noul("pointer")
	require.True(t, ok)
	require.InDelta(t, 0.5, pointer.Probability, 1e-9)

	_, ok = response.Choice("billing")
	require.False(t, ok, "wrong type")
	_, ok = response.Noul("missing")
	require.False(t, ok, "missing question")

	require.True(t, response.Refused("refused"))
	require.False(t, response.Refused("billing"))
	require.False(t, response.Refused("missing"))
	_, ok = response.Noul("refused")
	require.False(t, ok, "refused answers carry no values")

	box, ok := AsAnswer[boxAnswer](response.Answers["logo"])
	require.True(t, ok)
	require.InDelta(t, 0.2, box.Top, 1e-9)
	_, ok = AsAnswer[boxAnswer](nil)
	require.False(t, ok)
}

func TestAnswerAndQuestionTypes(t *testing.T) {
	t.Parallel()

	require.Equal(t, QuestionTypeNoul, NoulQuestion{}.GetType())
	require.Equal(t, QuestionTypeChoice, ChoiceQuestion{}.GetType())
	require.Equal(t, QuestionTypeScore, ScoreQuestion{}.GetType())
	require.Equal(t, QuestionTypeNoul, NoulAnswer{}.GetType())
	require.Equal(t, QuestionTypeChoice, ChoiceAnswer{}.GetType())
	require.Equal(t, QuestionTypeScore, ScoreAnswer{}.GetType())
	require.Equal(t, AnswerTypeRefusal, RefusalAnswer{}.GetType())
}
