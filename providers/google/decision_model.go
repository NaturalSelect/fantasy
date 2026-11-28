package google

import (
	"cmp"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strconv"

	"charm.land/fantasy"
	"google.golang.org/genai"
	"google.golang.org/genai/interactions/models/decisions"
	gi "google.golang.org/genai/interactions/models/interactions"
	"google.golang.org/genai/interactions/models/operations"
	"google.golang.org/genai/interactions/retry"
)

// DefaultDecisionModel is the decision model used when no model ID is given.
// It is the only model the Gemini Decisions API offers.
const DefaultDecisionModel = "gemini-decision-1"

const (
	maxDecisionQuestions = 1000
	minDecisionEntries   = 2
	maxDecisionEntries   = 255
)

var decisionQuestionName = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)

type decisionModel struct {
	modelID  string
	provider string
	client   *genai.Client
}

var _ fantasy.DecisionModelProvider = (*provider)(nil)

// DecisionModel implements fantasy.DecisionModelProvider using the Gemini
// Decisions API. It is only available on the Gemini Developer API backend.
func (a *provider) DecisionModel(ctx context.Context, modelID string) (fantasy.DecisionModel, error) {
	if a.options.backend == genai.BackendVertexAI {
		return nil, &fantasy.Error{Title: "unsupported", Message: "the Gemini Decisions API is only available on the Gemini Developer API backend"}
	}
	client, err := a.newClient(ctx)
	if err != nil {
		return nil, err
	}
	return &decisionModel{
		modelID:  cmp.Or(modelID, DefaultDecisionModel),
		provider: a.options.name,
		client:   client,
	}, nil
}

// Model implements fantasy.DecisionModel.
func (m *decisionModel) Model() string {
	return m.modelID
}

// Provider implements fantasy.DecisionModel.
func (m *decisionModel) Provider() string {
	return m.provider
}

// Decide implements fantasy.DecisionModel.
func (m *decisionModel) Decide(ctx context.Context, call fantasy.DecisionCall) (*fantasy.DecisionResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, err
	}
	input, err := decisionInput(call.Input)
	if err != nil {
		return nil, err
	}
	questions, err := decisionQuestions(call.Questions)
	if err != nil {
		return nil, err
	}

	// NOTE: The SDK retries 408, 409, 429 and 5XX by default, but retries belong to the agent.
	result, err := m.client.Decisions.Create(withDecisionCallUA(ctx, call), operations.CreateDecisionRequest{
		Body: decisions.CreateDecisionRequest{
			Model:     m.modelID,
			Input:     input,
			Questions: questions,
		},
	}, operations.WithRetries(retry.Config{Strategy: "none"}))
	if err != nil {
		return nil, toProviderErr(err)
	}
	return m.toResponse(call.Questions, result.CreateDecisionResponse)
}

func invalidDecisionArgument(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid argument", Message: fmt.Sprintf(format, args...)}
}

func invalidDecisionResponse(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid response", Message: fmt.Sprintf(format, args...)}
}

func decisionInput(parts []fantasy.MessagePart) (decisions.Input, error) {
	if len(parts) == 1 {
		if text, ok := fantasy.AsMessagePart[fantasy.TextPart](parts[0]); ok {
			return decisions.NewInput(text.Text), nil
		}
	}

	contents := make([]decisions.DecisionInputContent, 0, len(parts))
	for i, part := range parts {
		if text, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
			contents = append(contents, decisions.NewDecisionInputContent(gi.TextContent{Text: text.Text}))
			continue
		}
		file, ok := fantasy.AsMessagePart[fantasy.FilePart](part)
		if !ok {
			return decisions.Input{}, invalidDecisionArgument("input part %d: unsupported part type", i)
		}
		mimeType := gi.ImageContentMimeType(file.MediaType)
		if !mimeType.IsExact() {
			return decisions.Input{}, invalidDecisionArgument("input part %d: image media type %q is not supported, use png, jpeg, webp, heic, heif, gif, bmp or tiff", i, file.MediaType)
		}
		data := base64.StdEncoding.EncodeToString(file.Data)
		contents = append(contents, decisions.NewDecisionInputContent(gi.ImageContent{Data: &data, MimeType: &mimeType}))
	}
	return decisions.NewInput(contents), nil
}

func decisionQuestions(questions map[string]fantasy.Question) (map[string]decisions.Question, error) {
	if len(questions) > maxDecisionQuestions {
		return nil, invalidDecisionArgument("at most %d questions are supported, got %d", maxDecisionQuestions, len(questions))
	}

	converted := make(map[string]decisions.Question, len(questions))
	for _, name := range slices.Sorted(maps.Keys(questions)) {
		if !decisionQuestionName.MatchString(name) {
			return nil, invalidDecisionArgument("question name %q must be 1 to 64 characters of letters, digits, '_' or '-'", name)
		}
		question, err := decisionQuestion(questions[name])
		if err != nil {
			return nil, nameQuestionError(name, err)
		}
		converted[name] = question
	}
	return converted, nil
}

func nameQuestionError(name string, err error) error {
	fantasyErr, ok := errors.AsType[*fantasy.Error](err)
	if !ok {
		return invalidDecisionArgument("question %q: %v", name, err)
	}
	return &fantasy.Error{Title: fantasyErr.Title, Message: fmt.Sprintf("question %q: %s", name, fantasyErr.Message), Cause: fantasyErr.Cause}
}

func decisionQuestion(question fantasy.Question) (decisions.Question, error) {
	switch q := question.(type) {
	case fantasy.ChoiceQuestion:
		return choiceQuestion(q)
	case fantasy.ScoreQuestion:
		return scoreQuestion(q)
	case fantasy.NoulQuestion:
		return noulQuestion(q)
	default:
		return decisions.Question{}, &fantasy.Error{
			Title:   "unsupported",
			Message: fmt.Sprintf("question type %T is not supported by the Gemini Decisions API, use noul, choice or score questions by value", question),
		}
	}
}

func renderInstructions(instructions fantasy.Text) (string, error) {
	text, err := fantasy.RenderText(instructions)
	if err != nil {
		return "", invalidDecisionArgument("instructions: %v", err)
	}
	if text == "" {
		return "", invalidDecisionArgument("instructions must not be empty")
	}
	return text, nil
}

func choiceQuestion(q fantasy.ChoiceQuestion) (decisions.Question, error) {
	if len(q.Options) < minDecisionEntries || len(q.Options) > maxDecisionEntries {
		return decisions.Question{}, invalidDecisionArgument("choice questions need %d to %d options, got %d", minDecisionEntries, maxDecisionEntries, len(q.Options))
	}
	instructions, err := renderInstructions(q.Instructions)
	if err != nil {
		return decisions.Question{}, err
	}
	criteria := make(map[string]string, len(q.Options))
	for _, option := range q.Options {
		description, err := fantasy.RenderText(option.Description)
		if err != nil {
			return decisions.Question{}, invalidDecisionArgument("option %q: %v", option.Name, err)
		}
		criteria[option.Name] = description
	}
	return decisions.NewQuestion(decisions.ChoiceQuestion{
		Instructions: decisions.NewChoiceQuestionInstructions(instructions),
		Criteria:     criteria,
	}), nil
}

func scoreQuestion(q fantasy.ScoreQuestion) (decisions.Question, error) {
	if len(q.Levels) < minDecisionEntries || len(q.Levels) > maxDecisionEntries {
		return decisions.Question{}, invalidDecisionArgument("score questions need %d to %d levels, got %d", minDecisionEntries, maxDecisionEntries, len(q.Levels))
	}
	instructions, err := renderInstructions(q.Instructions)
	if err != nil {
		return decisions.Question{}, err
	}
	criteria := make([]string, 0, len(q.Levels))
	for i, level := range q.Levels {
		rendered, err := fantasy.RenderScoreLevel(level)
		if err != nil {
			return decisions.Question{}, invalidDecisionArgument("level %d: %v", i, err)
		}
		criteria = append(criteria, rendered)
	}
	return decisions.NewQuestion(decisions.ScoreQuestion{
		Instructions: decisions.NewScoreQuestionInstructions(instructions),
		Criteria:     criteria,
	}), nil
}

func noulQuestion(q fantasy.NoulQuestion) (decisions.Question, error) {
	instructions, err := renderInstructions(q.Instructions)
	if err != nil {
		return decisions.Question{}, err
	}
	trueDescription, err := fantasy.RenderText(q.TrueDescription)
	if err != nil {
		return decisions.Question{}, invalidDecisionArgument("true description: %v", err)
	}
	falseDescription, err := fantasy.RenderText(q.FalseDescription)
	if err != nil {
		return decisions.Question{}, invalidDecisionArgument("false description: %v", err)
	}

	noul := decisions.NoulQuestion{Instructions: decisions.NewNoulQuestionInstructions(instructions)}
	if trueDescription != "" || falseDescription != "" {
		noul.Criteria = &decisions.NoulCriteria{}
		if trueDescription != "" {
			noul.Criteria.TrueDescription = &trueDescription
		}
		if falseDescription != "" {
			noul.Criteria.FalseDescription = &falseDescription
		}
	}
	return decisions.NewQuestion(noul), nil
}

func (m *decisionModel) toResponse(questions map[string]fantasy.Question, result *decisions.CreateDecisionResponse) (*fantasy.DecisionResponse, error) {
	if result == nil {
		return nil, invalidDecisionResponse("the response was empty")
	}
	if feedback := result.PromptFeedback; feedback != nil && feedback.BlockReason != nil && *feedback.BlockReason != "" {
		message := string(*feedback.BlockReason)
		if feedback.BlockReasonMessage != nil && *feedback.BlockReasonMessage != "" {
			message += ": " + *feedback.BlockReasonMessage
		}
		return nil, &fantasy.Error{Title: "content blocked", Message: message}
	}

	answers := make(map[string]fantasy.Answer, len(questions))
	for _, name := range slices.Sorted(maps.Keys(questions)) {
		raw, ok := result.Answers[name]
		if !ok {
			return nil, invalidDecisionResponse("the response has no answer for question %q", name)
		}
		if raw.IsUnknown() {
			return nil, invalidDecisionResponse("the answer to question %q has an unknown type", name)
		}
		answer, err := toAnswer(name, questions[name], raw)
		if err != nil {
			return nil, err
		}
		answers[name] = answer
	}

	return &fantasy.DecisionResponse{
		Model:   m.modelID,
		Answers: answers,
		Usage:   decisionUsage(result.Usage),
	}, nil
}

func toAnswer(name string, question fantasy.Question, raw decisions.Answer) (fantasy.Answer, error) {
	switch q := question.(type) {
	case fantasy.NoulQuestion:
		if raw.NoulAnswer != nil {
			return fantasy.NoulAnswer{
				Probability: raw.NoulAnswer.Noul,
				Confidence:  &raw.NoulAnswer.Confidence,
			}, nil
		}
	case fantasy.ChoiceQuestion:
		if raw.ChoiceAnswer != nil {
			return fantasy.ChoiceAnswer{
				Choice:        raw.ChoiceAnswer.Choice,
				Probabilities: raw.ChoiceAnswer.Probabilities,
				Confidence:    &raw.ChoiceAnswer.Confidence,
			}, nil
		}
	case fantasy.ScoreQuestion:
		if raw.ScoreAnswer != nil {
			return toScoreAnswer(name, q, raw.ScoreAnswer)
		}
	}
	return nil, invalidDecisionResponse("the answer to question %q is of type %q, want %q", name, raw.Type, question.GetType())
}

func toScoreAnswer(name string, question fantasy.ScoreQuestion, raw *decisions.ScoreAnswer) (fantasy.Answer, error) {
	probabilities := make(map[int]float64, len(raw.Probabilities))
	for key, probability := range raw.Probabilities {
		level, err := strconv.Atoi(key)
		if err != nil {
			return nil, invalidDecisionResponse("the answer to question %q has the level %q, which is not a number", name, key)
		}
		probabilities[level] = probability
	}
	// NOTE: The legend comes from the request, so labels are exactly what the caller wrote.
	legend := make(map[int]string, len(question.Levels))
	for i, level := range question.Levels {
		legend[i] = level.Label
	}
	return fantasy.ScoreAnswer{
		Score:         raw.Score,
		Probabilities: probabilities,
		Legend:        legend,
		Confidence:    &raw.Confidence,
	}, nil
}

func decisionUsage(usage *gi.Usage) fantasy.Usage {
	if usage == nil {
		return fantasy.Usage{}
	}
	count := func(tokens *int) int64 {
		if tokens == nil {
			return 0
		}
		return int64(*tokens)
	}
	// NOTE: Like the other Gemini models, the input tokens include the cached ones.
	return fantasy.Usage{
		InputTokens:     count(usage.TotalInputTokens),
		OutputTokens:    count(usage.TotalOutputTokens),
		TotalTokens:     count(usage.TotalTokens),
		ReasoningTokens: count(usage.TotalThoughtTokens),
		CacheReadTokens: count(usage.TotalCachedTokens),
	}
}
