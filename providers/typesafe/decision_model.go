package typesafe

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
)

const (
	decisionPath = "/v1/systemone"

	imagesHeader      = "TypeSafe-Danger-Zone-Images"
	longContextHeader = "TypeSafe-Danger-Zone-Long-Context"
	requestIDHeader   = "x-typesafe-request-id"

	maxChoiceOptions = 255
	maxScoreLevels   = 10

	maxErrorBodyBytes    = 1 << 20
	maxResponseBodyBytes = 8 << 20
)

type decisionModel struct {
	provider string
	modelID  string
	options  options
}

// Model implements fantasy.DecisionModel.
func (m *decisionModel) Model() string {
	return m.modelID
}

// Provider implements fantasy.DecisionModel.
func (m *decisionModel) Provider() string {
	return m.provider
}

// Decide implements fantasy.DecisionModel. It sends one request, without
// retries, and returns BoxAnswer next to the answers of the core question types.
func (m *decisionModel) Decide(ctx context.Context, call fantasy.DecisionCall) (*fantasy.DecisionResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, err
	}
	options, err := decisionProviderOptions(call)
	if err != nil {
		return nil, err
	}
	request, err := m.buildRequest(call)
	if err != nil {
		return nil, err
	}
	body, err := json.Marshal(request)
	if err != nil {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "encode request: " + err.Error(), Cause: err}
	}
	hasImages := len(request.Assets) > 0

	httpRequest, err := m.newHTTPRequest(ctx, call, options, body, hasImages)
	if err != nil {
		return nil, err
	}
	httpResponse, err := m.options.client.Do(httpRequest)
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return nil, transportError(err)
	}
	defer httpResponse.Body.Close()

	if httpResponse.StatusCode < http.StatusOK || httpResponse.StatusCode >= http.StatusMultipleChoices {
		// NOTE: Even an incomplete error body must not hide the status or retry headers.
		responseBody, _ := io.ReadAll(io.LimitReader(httpResponse.Body, maxErrorBodyBytes))
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		providerErr := toProviderErr(httpResponse, responseBody, httpRequest.URL.String())
		if !hasImages {
			providerErr.RequestBody = body
		}
		return nil, providerErr
	}
	responseBody, err := io.ReadAll(io.LimitReader(httpResponse.Body, maxResponseBodyBytes+1))
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if len(responseBody) > maxResponseBodyBytes {
		return nil, invalidResponsef("response body exceeds %d bytes", maxResponseBodyBytes)
	}
	if err != nil {
		return nil, transportError(err)
	}
	return toDecisionResponse(call.Questions, responseBody, httpResponse.Header.Get(requestIDHeader))
}

func decisionProviderOptions(call fantasy.DecisionCall) (*DecisionProviderOptions, error) {
	v := call.ProviderOptions[Name]
	if v == nil {
		return &DecisionProviderOptions{}, nil
	}
	options, ok := v.(*DecisionProviderOptions)
	if !ok {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "typesafe provider options should be *typesafe.DecisionProviderOptions"}
	}
	if options == nil {
		return &DecisionProviderOptions{}, nil
	}
	return options, nil
}

func (m *decisionModel) newHTTPRequest(ctx context.Context, call fantasy.DecisionCall, options *DecisionProviderOptions, body []byte, hasImages bool) (*http.Request, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, m.options.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, &fantasy.Error{Title: "invalid argument", Message: "create request: " + err.Error(), Cause: err}
	}

	if m.options.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+m.options.apiKey)
	}
	defaultUA := httpheaders.DefaultUserAgent(fantasy.Version)
	for key, value := range httpheaders.ResolveHeaders(m.options.headers, m.options.userAgent, defaultUA) {
		request.Header.Set(key, value)
	}
	if ua, ok := httpheaders.CallUserAgent(call.UserAgent); ok {
		request.Header.Set("User-Agent", ua)
	}
	if headers, ok := httpheaders.CallHeaders(call.Headers); ok {
		for key, value := range headers {
			request.Header.Set(key, value)
		}
	}

	request.Header.Set("Content-Type", "application/json")
	// NOTE: A box question requires an image, so the images header is needed exactly when there are assets.
	if hasImages {
		request.Header.Set(imagesHeader, "true")
	}
	if options.LongContext != nil && *options.LongContext {
		request.Header.Set(longContextHeader, "true")
	}
	return request, nil
}

func transportError(err error) error {
	if errors.Is(err, context.Canceled) {
		return err
	}
	return fantasy.WrapTransportError(err)
}

type wireRequest struct {
	Model     string                  `json:"model"`
	State     string                  `json:"state"`
	Questions map[string]wireQuestion `json:"questions"`
	Assets    map[string]wireAsset    `json:"assets,omitempty"`
}

type wireAsset struct {
	Type   string `json:"type"`
	Format string `json:"format"`
	Data   []byte `json:"data_base64"`
}

type wireQuestion struct {
	Type         fantasy.QuestionType `json:"type"`
	Instructions string               `json:"instructions,omitempty"`
	Criteria     any                  `json:"criteria,omitempty"`
}

type noulCriteria struct {
	True  string `json:"true,omitempty"`
	False string `json:"false,omitempty"`
}

type choiceCriterion struct {
	name        string
	description *string
}

// NOTE: TypeSafe evaluates options in the given order and leans toward the
// first one, so the criteria cannot be a Go map, whose order is random.
type choiceCriteria []choiceCriterion

func (c choiceCriteria) MarshalJSON() ([]byte, error) {
	var out bytes.Buffer
	out.WriteByte('{')
	for i, criterion := range c {
		if i > 0 {
			out.WriteByte(',')
		}
		key, err := json.Marshal(criterion.name)
		if err != nil {
			return nil, err
		}
		value, err := json.Marshal(criterion.description)
		if err != nil {
			return nil, err
		}
		out.Write(key)
		out.WriteByte(':')
		out.Write(value)
	}
	out.WriteByte('}')
	return out.Bytes(), nil
}

func (m *decisionModel) buildRequest(call fantasy.DecisionCall) (wireRequest, error) {
	state, assets, err := buildState(call.Input)
	if err != nil {
		return wireRequest{}, err
	}
	questions := make(map[string]wireQuestion, len(call.Questions))
	for _, name := range slices.Sorted(maps.Keys(call.Questions)) {
		question, err := buildQuestion(name, call.Questions[name], len(assets) > 0)
		if err != nil {
			return wireRequest{}, err
		}
		questions[name] = question
	}
	return wireRequest{Model: m.modelID, State: state, Questions: questions, Assets: assets}, nil
}

// buildState joins the input into one state string. Each image is replaced by
// a reference to its asset, which TypeSafe resolves in the state.
func buildState(input []fantasy.MessagePart) (string, map[string]wireAsset, error) {
	segments := make([]string, 0, len(input))
	assets := map[string]wireAsset{}
	for _, part := range input {
		if text, ok := fantasy.AsMessagePart[fantasy.TextPart](part); ok {
			segments = append(segments, text.Text)
			continue
		}
		file, _ := fantasy.AsMessagePart[fantasy.FilePart](part)
		format, ok := imageFormat(file.MediaType)
		if !ok {
			return "", nil, invalidArgumentf("image type %q is not supported, use image/png or image/jpeg", file.MediaType)
		}
		sum := sha256.Sum256(file.Data)
		ref := "image_" + hex.EncodeToString(sum[:])[:16]
		assets[ref] = wireAsset{Type: "image", Format: format, Data: file.Data}
		segments = append(segments, ref)
	}
	return strings.Join(segments, "\n\n"), assets, nil
}

func imageFormat(mediaType string) (string, bool) {
	switch mediaType {
	case "image/png":
		return "png", true
	case "image/jpeg", "image/jpg":
		return "jpeg", true
	default:
		return "", false
	}
}

func buildQuestion(name string, question fantasy.Question, hasImages bool) (wireQuestion, error) {
	switch q := question.(type) {
	case fantasy.NoulQuestion:
		return noulQuestion(name, q)
	case fantasy.ChoiceQuestion:
		return choiceQuestion(name, q)
	case fantasy.ScoreQuestion:
		return scoreQuestion(name, q)
	case BoxQuestion:
		if !hasImages {
			return wireQuestion{}, invalidArgumentf("question %q: a box question needs at least one image in the input", name)
		}
		return boxQuestion(name, q)
	default:
		return wireQuestion{}, &fantasy.Error{Title: "unsupported", Message: fmt.Sprintf("question %q: %T is not supported by typesafe", name, question)}
	}
}

func noulQuestion(name string, q fantasy.NoulQuestion) (wireQuestion, error) {
	var r textRenderer
	question := wireQuestion{Type: fantasy.QuestionTypeNoul, Instructions: r.text(q.Instructions)}
	criteria := noulCriteria{True: r.text(q.TrueDescription), False: r.text(q.FalseDescription)}
	if r.err != nil {
		return wireQuestion{}, invalidQuestion(name, r.err)
	}
	if criteria != (noulCriteria{}) {
		question.Criteria = criteria
	}
	return question, nil
}

func choiceQuestion(name string, q fantasy.ChoiceQuestion) (wireQuestion, error) {
	if len(q.Options) > maxChoiceOptions {
		return wireQuestion{}, invalidArgumentf("question %q: a choice question has at most %d options, got %d", name, maxChoiceOptions, len(q.Options))
	}
	var r textRenderer
	criteria := make(choiceCriteria, len(q.Options))
	for i, option := range q.Options {
		criteria[i] = choiceCriterion{name: option.Name}
		if description := r.text(option.Description); description != "" {
			criteria[i].description = &description
		}
	}
	question := wireQuestion{Type: fantasy.QuestionTypeChoice, Instructions: r.text(q.Instructions), Criteria: criteria}
	if r.err != nil {
		return wireQuestion{}, invalidQuestion(name, r.err)
	}
	return question, nil
}

func scoreQuestion(name string, q fantasy.ScoreQuestion) (wireQuestion, error) {
	if len(q.Levels) > maxScoreLevels {
		return wireQuestion{}, invalidArgumentf("question %q: a score question has at most %d levels, got %d", name, maxScoreLevels, len(q.Levels))
	}
	var r textRenderer
	criteria := make([]string, len(q.Levels))
	for i, level := range q.Levels {
		criteria[i] = r.level(level)
	}
	question := wireQuestion{Type: fantasy.QuestionTypeScore, Instructions: r.text(q.Instructions), Criteria: criteria}
	if r.err != nil {
		return wireQuestion{}, invalidQuestion(name, r.err)
	}
	return question, nil
}

func boxQuestion(name string, q BoxQuestion) (wireQuestion, error) {
	var r textRenderer
	question := wireQuestion{Type: QuestionTypeBox, Instructions: r.text(q.Instructions)}
	if r.err != nil {
		return wireQuestion{}, invalidQuestion(name, r.err)
	}
	return question, nil
}

// textRenderer renders texts and keeps the first error, so that a question
// can be built without checking every text.
type textRenderer struct {
	err error
}

func (r *textRenderer) text(t fantasy.Text) string {
	s, err := fantasy.RenderText(t)
	if err != nil && r.err == nil {
		r.err = err
	}
	return s
}

func (r *textRenderer) level(level fantasy.ScoreLevel) string {
	s, err := fantasy.RenderScoreLevel(level)
	if err != nil && r.err == nil {
		r.err = err
	}
	return s
}

func invalidArgumentf(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid argument", Message: fmt.Sprintf(format, args...)}
}

func invalidQuestion(name string, err error) error {
	return &fantasy.Error{Title: "invalid argument", Message: fmt.Sprintf("question %q: %v", name, err), Cause: err}
}

type wireResponse struct {
	Model   string                     `json:"model"`
	Answers map[string]json.RawMessage `json:"answers"`
	Usage   struct {
		InputTokens  int64 `json:"input_tokens"`
		OutputTokens int64 `json:"output_tokens"`
	} `json:"usage"`
}

type wireAnswer struct {
	Type          fantasy.QuestionType `json:"type"`
	Choice        *string              `json:"choice"`
	Score         *float64             `json:"score"`
	Noul          *float64             `json:"noul"`
	Box           *wireBox             `json:"box"`
	Confidence    *float64             `json:"confidence"`
	Probabilities wireProbabilities    `json:"probabilities"`
}

type wireBox struct {
	Top    *float64 `json:"top"`
	Bottom *float64 `json:"bottom"`
	Left   *float64 `json:"left"`
	Right  *float64 `json:"right"`
}

func inRange(v *float64, max float64) bool {
	return v != nil && *v >= 0 && *v <= max
}

type wireProbabilities map[string]float64

func (p *wireProbabilities) UnmarshalJSON(data []byte) error {
	// NOTE: Decoding straight into a map would silently discard duplicate levels.
	dec := json.NewDecoder(bytes.NewReader(data))
	start, err := dec.Token()
	if err != nil {
		return err
	}
	if start == nil {
		*p = nil
		return nil
	}
	if start != json.Delim('{') {
		return errors.New("expected an object of probabilities")
	}
	values := wireProbabilities{}
	for dec.More() {
		token, err := dec.Token()
		if err != nil {
			return err
		}
		key := token.(string)
		if _, exists := values[key]; exists {
			return fmt.Errorf("duplicate probability key %q", key)
		}
		var value *float64
		if err := dec.Decode(&value); err != nil {
			return err
		}
		if !inRange(value, 1) {
			return fmt.Errorf("probability for %q must be a number in [0, 1]", key)
		}
		values[key] = *value
	}
	*p = values
	return nil
}

func toDecisionResponse(questions map[string]fantasy.Question, body []byte, requestID string) (*fantasy.DecisionResponse, error) {
	var wire wireResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, &fantasy.Error{Title: "invalid response", Message: "decode response: " + err.Error(), Cause: err}
	}

	answers := make(map[string]fantasy.Answer, len(questions))
	for _, name := range slices.Sorted(maps.Keys(questions)) {
		raw, ok := wire.Answers[name]
		if !ok {
			return nil, invalidResponsef("no answer for question %q", name)
		}
		answer, err := toAnswer(name, questions[name], raw)
		if err != nil {
			return nil, err
		}
		answers[name] = answer
	}

	response := &fantasy.DecisionResponse{
		Model:   wire.Model,
		Answers: answers,
		Usage: fantasy.Usage{
			InputTokens:  wire.Usage.InputTokens,
			OutputTokens: wire.Usage.OutputTokens,
			TotalTokens:  wire.Usage.InputTokens + wire.Usage.OutputTokens,
		},
	}
	if requestID != "" {
		response.ProviderMetadata = fantasy.ProviderMetadata{Name: &ProviderMetadata{RequestID: requestID}}
	}
	return response, nil
}

func toAnswer(name string, question fantasy.Question, raw json.RawMessage) (fantasy.Answer, error) {
	var wire wireAnswer
	if err := json.Unmarshal(raw, &wire); err != nil {
		return nil, &fantasy.Error{Title: "invalid response", Message: fmt.Sprintf("answer %q: %v", name, err), Cause: err}
	}
	if want := question.GetType(); wire.Type != want {
		return nil, invalidResponsef("answer %q has type %q, expected %q", name, wire.Type, want)
	}

	switch q := question.(type) {
	case fantasy.NoulQuestion:
		if wire.Noul == nil {
			return nil, invalidResponsef("answer %q has no noul value", name)
		}
		if !inRange(wire.Noul, 1) {
			return nil, invalidResponsef("answer %q has noul outside [0, 1]", name)
		}
		return fantasy.NoulAnswer{Probability: *wire.Noul}, nil
	case fantasy.ChoiceQuestion:
		if wire.Choice == nil {
			return nil, invalidResponsef("answer %q has no choice", name)
		}
		options := make(map[string]bool, len(q.Options))
		for _, opt := range q.Options {
			options[opt.Name] = true
		}
		if !options[*wire.Choice] {
			return nil, invalidResponsef("answer %q has unknown choice %q", name, *wire.Choice)
		}
		for key := range wire.Probabilities {
			if !options[key] {
				return nil, invalidResponsef("answer %q has probability for unknown choice %q", name, key)
			}
		}
		return fantasy.ChoiceAnswer{Choice: *wire.Choice, Probabilities: wire.Probabilities, Confidence: wire.Confidence}, nil
	case fantasy.ScoreQuestion:
		if wire.Score == nil {
			return nil, invalidResponsef("answer %q has no score", name)
		}
		if !inRange(wire.Score, float64(len(q.Levels)-1)) {
			return nil, invalidResponsef("answer %q has score outside the requested levels", name)
		}
		probabilities := make(map[int]float64, len(wire.Probabilities))
		for key, probability := range wire.Probabilities {
			level, err := strconv.Atoi(key)
			if err != nil || strconv.Itoa(level) != key || level < 0 || level >= len(q.Levels) {
				return nil, invalidResponsef("answer %q has invalid probability level %q", name, key)
			}
			probabilities[level] = probability
		}
		legend := make(map[int]string, len(q.Levels))
		for i, level := range q.Levels {
			legend[i] = level.Label
		}
		return fantasy.ScoreAnswer{Score: *wire.Score, Probabilities: probabilities, Legend: legend, Confidence: wire.Confidence}, nil
	case BoxQuestion:
		if wire.Box == nil {
			return nil, invalidResponsef("answer %q has no box", name)
		}
		box := wire.Box
		if !inRange(box.Top, 1) || !inRange(box.Bottom, 1) || !inRange(box.Left, 1) || !inRange(box.Right, 1) {
			return nil, invalidResponsef("answer %q must have four box edges in [0, 1]", name)
		}
		return BoxAnswer{
			Top: *box.Top, Bottom: *box.Bottom, Left: *box.Left, Right: *box.Right,
			Confidence: wire.Confidence,
		}, nil
	default:
		return nil, &fantasy.Error{Title: "unsupported", Message: fmt.Sprintf("question %q: %T is not supported by typesafe", name, question)}
	}
}

func invalidResponsef(format string, args ...any) error {
	return &fantasy.Error{Title: "invalid response", Message: fmt.Sprintf(format, args...)}
}
