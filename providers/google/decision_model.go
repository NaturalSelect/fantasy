package google

import (
	"cmp"
	"context"
	"encoding/base64"
	"encoding/json/v2"
	"errors"
	"fmt"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"

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
	backend := a.effectiveBackend()
	if backend != genai.BackendGeminiAPI {
		return nil, &fantasy.Error{Title: "unsupported", Message: "the Gemini Decisions API is only available on the Gemini Developer API backend"}
	}
	client, err := a.newClientForBackend(ctx, backend)
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
		// NOTE: genai does not export a distinct error type for response decoding failures.
		if strings.HasPrefix(err.Error(), "error unmarshaling json response body:") {
			return nil, &fantasy.Error{Title: "invalid response", Message: "could not decode the decision response", Cause: err}
		}
		return nil, toProviderErr(err)
	}
	defer result.HTTPMeta.Response.Body.Close()
	return m.toResponse(call, result)
}

func invalidDecisionArgument(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid argument", Message: fmt.Sprintf(format, args...)}
}

func invalidDecisionResponse(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid response", Message: fmt.Sprintf(format, args...)}
}

func decisionInput(parts []fantasy.MessagePart) (decisions.Input, error) {
	contents := make([]decisions.DecisionInputContent, 0, len(parts))
	for i, part := range parts {
		if text, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
			if text.Text == "" {
				return decisions.Input{}, invalidDecisionArgument("input part %d: text must not be empty", i)
			}
			if len(parts) == 1 {
				return decisions.NewInput(text.Text), nil
			}
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
		name := option.Name
		if !utf8.ValidString(name) || len(name) > 512 || utf8.RuneCountInString(name) > 128 {
			return decisions.Question{}, invalidDecisionArgument("option name must be valid UTF-8 of at most 128 characters and 512 bytes")
		}
		if strings.HasPrefix(name, " ") || strings.HasSuffix(name, " ") || strings.ContainsFunc(name, func(r rune) bool {
			return unicode.IsControl(r) || r != ' ' && unicode.IsSpace(r)
		}) {
			return decisions.Question{}, invalidDecisionArgument("option name %q contains a disallowed space or control character", name)
		}
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

func (m *decisionModel) toResponse(call fantasy.DecisionCall, result *operations.CreateDecisionResponse) (*fantasy.DecisionResponse, error) {
	body := result.CreateDecisionResponse
	if body == nil {
		return nil, invalidDecisionResponse("the response was empty")
	}
	if feedback := body.PromptFeedback; feedback != nil && feedback.BlockReason != nil && *feedback.BlockReason != "" {
		message := string(*feedback.BlockReason)
		if feedback.BlockReasonMessage != nil && *feedback.BlockReasonMessage != "" {
			message += ": " + *feedback.BlockReasonMessage
		}
		return nil, &fantasy.Error{Title: "content blocked", Message: message}
	}

	// NOTE: The SDK flattens missing numbers to zero and collapses duplicate keys. Its retained response body preserves both distinctions.
	var wire struct {
		Answers map[string]decisionAnswer `json:"answers"`
	}
	if err := json.UnmarshalRead(result.HTTPMeta.Response.Body, &wire); err != nil {
		return nil, invalidDecisionResponse("could not decode answers: %v", err)
	}
	answers := make(map[string]fantasy.Answer, len(call.Questions))
	for _, name := range slices.Sorted(maps.Keys(call.Questions)) {
		raw, ok := wire.Answers[name]
		if !ok {
			return nil, invalidDecisionResponse("the response has no answer for question %q", name)
		}
		answer, err := toAnswer(name, call.Questions[name], raw)
		if err != nil {
			return nil, err
		}
		answers[name] = answer
	}

	response := &fantasy.DecisionResponse{Model: m.modelID, Answers: answers, Usage: decisionUsage(body.Usage)}
	if _, ok := call.ProviderOptions[Name]; ok {
		response.Warnings = []fantasy.CallWarning{{
			Type:    fantasy.CallWarningTypeUnsupportedSetting,
			Setting: "provider_options.google",
			Message: "Gemini decision models do not support Google provider options",
		}}
	}
	return response, nil
}

type decisionAnswer struct {
	Type          fantasy.QuestionType `json:"type"`
	Confidence    *float64             `json:"confidence"`
	Noul          *float64             `json:"noul"`
	Choice        string               `json:"choice"`
	Score         *float64             `json:"score"`
	Probabilities map[string]*float64  `json:"probabilities"`
	Legend        map[string]string    `json:"legend"`
}

func validDecisionProbability(p *float64) bool {
	return p != nil && *p >= 0 && *p <= 1
}

func toAnswer(name string, question fantasy.Question, raw decisionAnswer) (fantasy.Answer, error) {
	switch raw.Type {
	case fantasy.QuestionTypeNoul, fantasy.QuestionTypeChoice, fantasy.QuestionTypeScore:
	default:
		return nil, invalidDecisionResponse("the answer to question %q has an unknown type %q", name, raw.Type)
	}
	if raw.Type != question.GetType() {
		return nil, invalidDecisionResponse("the answer to question %q is of type %q, want %q", name, raw.Type, question.GetType())
	}
	if !validDecisionProbability(raw.Confidence) {
		return nil, invalidDecisionResponse("the answer to question %q has invalid confidence", name)
	}
	for key, probability := range raw.Probabilities {
		if !validDecisionProbability(probability) {
			return nil, invalidDecisionResponse("the answer to question %q has an invalid probability for %q", name, key)
		}
	}
	switch q := question.(type) {
	case fantasy.NoulQuestion:
		if !validDecisionProbability(raw.Noul) {
			return nil, invalidDecisionResponse("the answer to question %q has an invalid noul probability", name)
		}
		return fantasy.NoulAnswer{Probability: *raw.Noul, Confidence: raw.Confidence}, nil
	case fantasy.ChoiceQuestion:
		options := make(map[string]bool, len(q.Options))
		for _, opt := range q.Options {
			options[opt.Name] = true
		}
		if !options[raw.Choice] {
			return nil, invalidDecisionResponse("the answer to question %q chose unknown option %q", name, raw.Choice)
		}
		probabilities := make(map[string]float64, len(raw.Probabilities))
		for key, probability := range raw.Probabilities {
			if !options[key] {
				return nil, invalidDecisionResponse("the answer to question %q has a probability for unknown option %q", name, key)
			}
			probabilities[key] = *probability
		}
		return fantasy.ChoiceAnswer{Choice: raw.Choice, Probabilities: probabilities, Confidence: raw.Confidence}, nil
	case fantasy.ScoreQuestion:
		return toScoreAnswer(name, q, raw)
	}
	return nil, invalidDecisionResponse("the answer to question %q has an unsupported question type", name)
}

func toScoreAnswer(name string, question fantasy.ScoreQuestion, raw decisionAnswer) (fantasy.Answer, error) {
	if raw.Score == nil || !(*raw.Score >= 0 && *raw.Score <= float64(len(question.Levels)-1)) {
		return nil, invalidDecisionResponse("the answer to question %q has an invalid score", name)
	}
	parseLevel := func(key string) (int, error) {
		level, err := strconv.Atoi(key)
		if err != nil || strconv.Itoa(level) != key || level < 0 || level >= len(question.Levels) {
			return 0, invalidDecisionResponse("the answer to question %q has an invalid level %q", name, key)
		}
		return level, nil
	}
	probabilities := make(map[int]float64, len(raw.Probabilities))
	for key, probability := range raw.Probabilities {
		level, err := parseLevel(key)
		if err != nil {
			return nil, err
		}
		probabilities[level] = *probability
	}
	for key := range raw.Legend {
		if _, err := parseLevel(key); err != nil {
			return nil, err
		}
	}
	// NOTE: The legend comes from the request, so labels are exactly what the caller wrote.
	legend := make(map[int]string, len(question.Levels))
	for i, level := range question.Levels {
		legend[i] = level.Label
	}
	return fantasy.ScoreAnswer{
		Score:         *raw.Score,
		Probabilities: probabilities,
		Legend:        legend,
		Confidence:    raw.Confidence,
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
	input := count(usage.TotalInputTokens)
	output := count(usage.TotalOutputTokens)
	total := count(usage.TotalTokens)
	reasoning := count(usage.TotalThoughtTokens)
	// NOTE: Gemini can report thoughts either inside or outside output. Only fold them in when the total accounts for them separately.
	if reasoning > 0 && input+count(usage.TotalToolUseTokens)+output+reasoning == total {
		output += reasoning
	}
	return fantasy.Usage{
		InputTokens:     input,
		OutputTokens:    output,
		TotalTokens:     total,
		ReasoningTokens: reasoning,
		CacheReadTokens: count(usage.TotalCachedTokens),
	}
}
