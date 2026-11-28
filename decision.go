package fantasy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"slices"
	"strings"
)

// Text is descriptive text sent to a decision model, such as instructions or
// an option description. Use String for plain text, or any type that can
// render itself as text.
type Text interface {
	DecisionText() (string, error)
}

// String is plain text that implements Text.
type String string

// DecisionText implements Text.
func (s String) DecisionText() (string, error) {
	return string(s), nil
}

// TextOf converts any string-based type, such as a custom enum, to Text.
// NOTE: A union like `~string | fmt.Stringer` does not compile in Go, hence the generic constructor.
func TextOf[S ~string](s S) String {
	return String(s)
}

// JSONText renders v as compact JSON when the text is needed. Marshal errors
// surface from DecisionText.
func JSONText(v any) Text {
	return jsonText{value: v}
}

type jsonText struct {
	value any
}

func (j jsonText) DecisionText() (string, error) {
	data, err := json.Marshal(j.value)
	if err != nil {
		return "", fmt.Errorf("marshal text as json: %w", err)
	}
	return string(data), nil
}

// RenderText returns the string form of t. A nil t renders as "".
func RenderText(t Text) (string, error) {
	if t == nil {
		return "", nil
	}
	return t.DecisionText()
}

// QuestionType identifies the kind of a decision question or answer.
type QuestionType string

const (
	// QuestionTypeNoul asks for the probability that a statement is true.
	QuestionTypeNoul QuestionType = "noul"
	// QuestionTypeChoice asks to pick one option from a set.
	QuestionTypeChoice QuestionType = "choice"
	// QuestionTypeScore asks to rate the input on ordered levels.
	QuestionTypeScore QuestionType = "score"

	// AnswerTypeRefusal marks a question the provider declined to answer.
	// It is only used for answers, never for questions.
	AnswerTypeRefusal QuestionType = "refusal"
)

// Question is one question asked of a decision model. Providers can add their
// own question types by implementing it. Pass the built-in questions by value.
type Question interface {
	GetType() QuestionType
}

// ChoiceOption is one option of a ChoiceQuestion.
type ChoiceOption struct {
	Name string `json:"name"`
	// Description is optional. Without one the model interprets the option by name.
	Description Text `json:"description"`
}

// ChoiceQuestion asks the model to pick one option.
type ChoiceQuestion struct {
	Instructions Text `json:"instructions"`
	// Options are evaluated in the given order by TypeSafe and OpenAI, and
	// TypeSafe leans toward the first option. Gemini sorts them by name on
	// the server, so the order has no effect there.
	Options []ChoiceOption `json:"options"`
}

// GetType implements Question.
func (ChoiceQuestion) GetType() QuestionType { return QuestionTypeChoice }

// ScoreLevel is one level of a ScoreQuestion.
type ScoreLevel struct {
	Label       string `json:"label"`
	Description Text   `json:"description"`
}

// ScoreQuestion asks the model to rate the input on ordered levels.
type ScoreQuestion struct {
	Instructions Text `json:"instructions"`
	// Levels are ordered from lowest to highest. The index is the level number, starting at 0.
	Levels []ScoreLevel `json:"levels"`
}

// GetType implements Question.
func (ScoreQuestion) GetType() QuestionType { return QuestionTypeScore }

// NoulQuestion asks the model for the probability that a statement is true.
type NoulQuestion struct {
	Instructions Text `json:"instructions"`
	// TrueDescription and FalseDescription are optional and describe what true and false mean.
	TrueDescription  Text `json:"true_description"`
	FalseDescription Text `json:"false_description"`
}

// GetType implements Question.
func (NoulQuestion) GetType() QuestionType { return QuestionTypeNoul }

// DecisionCall represents a call to a decision model. Every question is
// answered independently against the same input.
type DecisionCall struct {
	// Input is the shared context for all questions. It accepts TextPart and
	// FilePart values, where a FilePart must be an image.
	Input []MessagePart `json:"input"`
	// Questions are keyed by a name of your choice, which keys the answers too.
	Questions map[string]Question `json:"questions"`

	// UserAgent overrides the provider-level User-Agent header for this call.
	UserAgent string `json:"-"`

	// Headers overrides matching provider-level headers for this call.
	Headers map[string]string `json:"-"`

	// for provider specific options, the key is the provider id
	ProviderOptions ProviderOptions `json:"provider_options"`
}

// Validate checks the provider-independent structure of the call. Limits that
// differ between providers are checked by each provider. Question types it
// does not know are left for the provider to accept or reject.
func (c DecisionCall) Validate() error {
	if len(c.Input) == 0 {
		return invalidDecisionCall("input must have at least one part")
	}
	for i, part := range c.Input {
		if err := validateDecisionInputPart(part); err != nil {
			return invalidDecisionCall("input part %d: %v", i, err)
		}
	}
	if len(c.Questions) == 0 {
		return invalidDecisionCall("at least one question is required")
	}
	for _, name := range slices.Sorted(maps.Keys(c.Questions)) {
		if name == "" {
			return invalidDecisionCall("question names must not be empty")
		}
		if err := validateQuestion(c.Questions[name]); err != nil {
			return invalidDecisionCall("question %q: %v", name, err)
		}
	}
	return nil
}

func validateDecisionInputPart(part MessagePart) error {
	if _, ok := AsMessagePart[TextPart](part); ok {
		return nil
	}
	if file, ok := AsMessagePart[FilePart](part); ok {
		if !strings.HasPrefix(file.MediaType, "image/") {
			return fmt.Errorf("files must be images, got media type %q", file.MediaType)
		}
		if len(file.Data) == 0 {
			return errors.New("image has no data")
		}
		return nil
	}
	if part == nil {
		return errors.New("part is nil")
	}
	return fmt.Errorf("unsupported part type %q, use text or image files", part.GetType())
}

func validateQuestion(question Question) error {
	switch q := question.(type) {
	case nil:
		return errors.New("question is nil")
	case NoulQuestion:
		return validateTexts(q.Instructions, q.TrueDescription, q.FalseDescription)
	case ChoiceQuestion:
		return validateChoiceQuestion(q)
	case ScoreQuestion:
		return validateScoreQuestion(q)
	default:
		return nil
	}
}

func validateChoiceQuestion(q ChoiceQuestion) error {
	if len(q.Options) == 0 {
		return errors.New("at least one option is required")
	}
	if err := validateTexts(q.Instructions); err != nil {
		return err
	}
	seen := make(map[string]struct{}, len(q.Options))
	for i, option := range q.Options {
		if option.Name == "" {
			return fmt.Errorf("option %d has no name", i)
		}
		if _, dup := seen[option.Name]; dup {
			return fmt.Errorf("option name %q is used more than once", option.Name)
		}
		seen[option.Name] = struct{}{}
		if err := validateTexts(option.Description); err != nil {
			return fmt.Errorf("option %q: %w", option.Name, err)
		}
	}
	return nil
}

func validateScoreQuestion(q ScoreQuestion) error {
	if len(q.Levels) == 0 {
		return errors.New("at least one level is required")
	}
	if err := validateTexts(q.Instructions); err != nil {
		return err
	}
	for i, level := range q.Levels {
		if level.Label == "" {
			return fmt.Errorf("level %d has no label", i)
		}
		if err := validateTexts(level.Description); err != nil {
			return fmt.Errorf("level %d: %w", i, err)
		}
	}
	return nil
}

func validateTexts(texts ...Text) error {
	for _, text := range texts {
		if _, err := RenderText(text); err != nil {
			return err
		}
	}
	return nil
}

func invalidDecisionCall(format string, args ...any) error {
	return &Error{Title: "invalid argument", Message: fmt.Sprintf(format, args...)}
}

// RenderScoreLevel renders a level as "Label: Description", or just the label
// when it has no description. It is for providers whose levels are plain text.
func RenderScoreLevel(level ScoreLevel) (string, error) {
	description, err := RenderText(level.Description)
	if err != nil {
		return "", err
	}
	if description == "" {
		return level.Label, nil
	}
	return level.Label + ": " + description, nil
}

// Answer is the answer to one decision question. Providers can add their own
// answer types by implementing it.
type Answer interface {
	GetType() QuestionType
}

// NoulAnswer answers a NoulQuestion.
type NoulAnswer struct {
	// Probability that the statement is true, from 0 to 1.
	Probability float64 `json:"probability"`
	// Confidence is nil when the provider does not report one for this question type.
	Confidence *float64 `json:"confidence"`
}

// GetType implements Answer.
func (NoulAnswer) GetType() QuestionType { return QuestionTypeNoul }

// ChoiceAnswer answers a ChoiceQuestion.
type ChoiceAnswer struct {
	// Choice is the name of the chosen option.
	Choice string `json:"choice"`
	// Probabilities are keyed by option name.
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    *float64           `json:"confidence"`
}

// GetType implements Answer.
func (ChoiceAnswer) GetType() QuestionType { return QuestionTypeChoice }

// ScoreAnswer answers a ScoreQuestion.
type ScoreAnswer struct {
	// Score is the probability-weighted mean level, so it can be fractional.
	Score float64 `json:"score"`
	// Probabilities are keyed by level number.
	Probabilities map[int]float64 `json:"probabilities"`
	// Legend maps each level number to the label of the level in the question.
	Legend     map[int]string `json:"legend"`
	Confidence *float64       `json:"confidence"`
}

// GetType implements Answer.
func (ScoreAnswer) GetType() QuestionType { return QuestionTypeScore }

// RefusalAnswer marks a question the provider declined to answer. The other
// questions of the call are still answered.
type RefusalAnswer struct{}

// GetType implements Answer.
func (RefusalAnswer) GetType() QuestionType { return AnswerTypeRefusal }

// DecisionResponse represents a response from a decision model.
type DecisionResponse struct {
	// Model is the model that answered. It can be an alias when the provider reports one.
	Model string `json:"model"`
	// Answers has exactly one entry per question, under the same name.
	Answers  map[string]Answer `json:"answers"`
	Warnings []CallWarning     `json:"warnings"`
	Usage    Usage             `json:"usage"`

	// for provider specific response metadata, the key is the provider id
	ProviderMetadata ProviderMetadata `json:"provider_metadata"`
}

// Noul returns the answer to a Noul question. It reports false when the
// question is missing, was refused, or is of another type.
func (r *DecisionResponse) Noul(name string) (NoulAnswer, bool) {
	return AsAnswer[NoulAnswer](r.Answers[name])
}

// Choice returns the answer to a Choice question. It reports false when the
// question is missing, was refused, or is of another type.
func (r *DecisionResponse) Choice(name string) (ChoiceAnswer, bool) {
	return AsAnswer[ChoiceAnswer](r.Answers[name])
}

// Score returns the answer to a Score question. It reports false when the
// question is missing, was refused, or is of another type.
func (r *DecisionResponse) Score(name string) (ScoreAnswer, bool) {
	return AsAnswer[ScoreAnswer](r.Answers[name])
}

// Refused reports whether the provider declined to answer the question.
func (r *DecisionResponse) Refused(name string) bool {
	_, ok := AsAnswer[RefusalAnswer](r.Answers[name])
	return ok
}

// AsAnswer converts an Answer to a specific answer type, such as a
// provider-specific one.
func AsAnswer[T Answer](answer Answer) (T, bool) {
	var zero T
	if answer == nil {
		return zero, false
	}
	switch v := any(answer).(type) {
	case T:
		return v, true
	case *T:
		return *v, true
	default:
		return zero, false
	}
}

// DecisionModel represents a model that answers typed questions with
// probabilities instead of generating text.
type DecisionModel interface {
	Decide(context.Context, DecisionCall) (*DecisionResponse, error)

	Provider() string
	Model() string
}

// DecisionModelProvider is implemented by providers that offer decision models.
// Check for it with a type assertion on a Provider.
type DecisionModelProvider interface {
	DecisionModel(ctx context.Context, modelID string) (DecisionModel, error)
}
