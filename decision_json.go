package fantasy

import (
	"encoding/json"
	"fmt"
)

func init() {
	RegisterQuestionType(QuestionTypeNoul, decodeBuiltinQuestion[NoulQuestion])
	RegisterQuestionType(QuestionTypeChoice, decodeBuiltinQuestion[ChoiceQuestion])
	RegisterQuestionType(QuestionTypeScore, decodeBuiltinQuestion[ScoreQuestion])
	RegisterAnswerType(QuestionTypeNoul, decodeBuiltinAnswer[NoulAnswer])
	RegisterAnswerType(QuestionTypeChoice, decodeBuiltinAnswer[ChoiceAnswer])
	RegisterAnswerType(QuestionTypeScore, decodeBuiltinAnswer[ScoreAnswer])
	RegisterAnswerType(AnswerTypeRefusal, decodeBuiltinAnswer[RefusalAnswer])
}

func decodeBuiltinQuestion[T Question](data []byte) (Question, error) {
	var q T
	err := json.Unmarshal(data, &q)
	return q, err
}

func decodeBuiltinAnswer[T Answer](data []byte) (Answer, error) {
	var a T
	err := json.Unmarshal(data, &a)
	return a, err
}

type decisionTextJSON struct{ Text }

func (t decisionTextJSON) MarshalJSON() ([]byte, error) {
	if nilOrNilPointer(t.Text) {
		return []byte("null"), nil
	}
	s, err := t.DecisionText()
	if err != nil {
		return nil, err
	}
	return json.Marshal(s)
}

func (t *decisionTextJSON) UnmarshalJSON(data []byte) error {
	var value *String
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	t.Text = nil
	if value != nil {
		t.Text = *value
	}
	return nil
}

type choiceOptionJSON struct {
	Name        string           `json:"name"`
	Description decisionTextJSON `json:"description"`
}

func (o ChoiceOption) MarshalJSON() ([]byte, error) {
	return json.Marshal(choiceOptionJSON{o.Name, decisionTextJSON{o.Description}})
}

func (o *ChoiceOption) UnmarshalJSON(data []byte) error {
	var wire choiceOptionJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*o = ChoiceOption{Name: wire.Name, Description: wire.Description.Text}
	return nil
}

type choiceQuestionJSON struct {
	Instructions decisionTextJSON `json:"instructions"`
	Options      []ChoiceOption   `json:"options"`
}

func (q ChoiceQuestion) MarshalJSON() ([]byte, error) {
	return json.Marshal(choiceQuestionJSON{decisionTextJSON{q.Instructions}, q.Options})
}

func (q *ChoiceQuestion) UnmarshalJSON(data []byte) error {
	var wire choiceQuestionJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*q = ChoiceQuestion{Instructions: wire.Instructions.Text, Options: wire.Options}
	return nil
}

type scoreLevelJSON struct {
	Label       string           `json:"label"`
	Description decisionTextJSON `json:"description"`
}

func (l ScoreLevel) MarshalJSON() ([]byte, error) {
	return json.Marshal(scoreLevelJSON{l.Label, decisionTextJSON{l.Description}})
}

func (l *ScoreLevel) UnmarshalJSON(data []byte) error {
	var wire scoreLevelJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*l = ScoreLevel{Label: wire.Label, Description: wire.Description.Text}
	return nil
}

type scoreQuestionJSON struct {
	Instructions decisionTextJSON `json:"instructions"`
	Levels       []ScoreLevel     `json:"levels"`
}

func (q ScoreQuestion) MarshalJSON() ([]byte, error) {
	return json.Marshal(scoreQuestionJSON{decisionTextJSON{q.Instructions}, q.Levels})
}

func (q *ScoreQuestion) UnmarshalJSON(data []byte) error {
	var wire scoreQuestionJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*q = ScoreQuestion{Instructions: wire.Instructions.Text, Levels: wire.Levels}
	return nil
}

type noulQuestionJSON struct {
	Instructions     decisionTextJSON `json:"instructions"`
	TrueDescription  decisionTextJSON `json:"true_description"`
	FalseDescription decisionTextJSON `json:"false_description"`
}

func (q NoulQuestion) MarshalJSON() ([]byte, error) {
	return json.Marshal(noulQuestionJSON{
		Instructions:     decisionTextJSON{q.Instructions},
		TrueDescription:  decisionTextJSON{q.TrueDescription},
		FalseDescription: decisionTextJSON{q.FalseDescription},
	})
}

func (q *NoulQuestion) UnmarshalJSON(data []byte) error {
	var wire noulQuestionJSON
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	*q = NoulQuestion{
		Instructions:     wire.Instructions.Text,
		TrueDescription:  wire.TrueDescription.Text,
		FalseDescription: wire.FalseDescription.Text,
	}
	return nil
}

type decisionValueJSON[T any] struct {
	Type QuestionType `json:"type"`
	Data T            `json:"data"`
}

type rawDecisionValues map[string]decisionValueJSON[json.RawMessage]

func wrapDecisionValues[T interface{ GetType() QuestionType }](values map[string]T, kind string) (map[string]decisionValueJSON[T], error) {
	if values == nil {
		return nil, nil
	}
	wrapped := make(map[string]decisionValueJSON[T], len(values))
	for name, v := range values {
		if nilOrNilPointer(v) {
			return nil, fmt.Errorf("%s %q is nil", kind, name)
		}
		wrapped[name] = decisionValueJSON[T]{Type: v.GetType(), Data: v}
	}
	return wrapped, nil
}

func decodeDecisionValues[T any](raw rawDecisionValues, kind string, decode func(QuestionType, []byte) (T, error)) (map[string]T, error) {
	if raw == nil {
		return nil, nil
	}
	values := make(map[string]T, len(raw))
	for name, v := range raw {
		item, err := decode(v.Type, v.Data)
		if err != nil {
			return nil, fmt.Errorf("%s %q: %w", kind, name, err)
		}
		values[name] = item
	}
	return values, nil
}

func (c DecisionCall) MarshalJSON() ([]byte, error) {
	questions, err := wrapDecisionValues(c.Questions, "question")
	if err != nil {
		return nil, err
	}
	type plain DecisionCall
	return json.Marshal(struct {
		plain
		Questions map[string]decisionValueJSON[Question] `json:"questions"`
	}{plain(c), questions})
}

func (c *DecisionCall) UnmarshalJSON(data []byte) error {
	var wire struct {
		Input           rawMessageParts   `json:"input"`
		Questions       rawDecisionValues `json:"questions"`
		ProviderOptions rawProviderData   `json:"provider_options"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	input, err := wire.Input.decode()
	if err != nil {
		return err
	}
	questions, err := decodeDecisionValues(wire.Questions, "question", decodeQuestion)
	if err != nil {
		return err
	}
	options, err := wire.ProviderOptions.options()
	if err != nil {
		return err
	}
	c.Input = input
	c.Questions = questions
	c.ProviderOptions = options
	return nil
}

func (r DecisionResponse) MarshalJSON() ([]byte, error) {
	answers, err := wrapDecisionValues(r.Answers, "answer")
	if err != nil {
		return nil, err
	}
	type plain DecisionResponse
	return json.Marshal(struct {
		plain
		Answers map[string]decisionValueJSON[Answer] `json:"answers"`
	}{plain(r), answers})
}

func (r *DecisionResponse) UnmarshalJSON(data []byte) error {
	var wire struct {
		Answers          rawDecisionValues `json:"answers"`
		Model            string            `json:"model"`
		Usage            Usage             `json:"usage"`
		Warnings         rawWarnings       `json:"warnings"`
		ProviderMetadata rawProviderData   `json:"provider_metadata"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	answers, err := decodeDecisionValues(wire.Answers, "answer", decodeAnswer)
	if err != nil {
		return err
	}
	metadata, err := wire.ProviderMetadata.metadata()
	if err != nil {
		return err
	}
	warnings, err := wire.Warnings.decode()
	if err != nil {
		return err
	}
	*r = DecisionResponse{
		Model:            wire.Model,
		Answers:          answers,
		Usage:            wire.Usage,
		Warnings:         warnings,
		ProviderMetadata: metadata,
	}
	return nil
}
