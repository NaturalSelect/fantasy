package google

import (
	"cmp"
	"encoding/json"
	"errors"
	"regexp"
	"strconv"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
	"google.golang.org/genai"
	"google.golang.org/genai/interactions/models/apierrors"
)

var googleContextPattern = regexp.MustCompile(`input token count.*?(\d+).*?exceeds.*?maximum.*?(\d+)`)

func toProviderErr(err error) error {
	if apiErr, ok := errors.AsType[*apierrors.APIError](err); ok {
		return toInteractionsProviderErr(err, apiErr)
	}

	apiErr, ok := errors.AsType[genai.APIError](err)
	if !ok {
		// Wrap transient transport failures so `.IsRetryable()` works.
		return fantasy.WrapTransportError(err)
	}

	providerErr := &fantasy.ProviderError{
		Message:      apiErr.Message,
		Title:        cmp.Or(fantasy.ErrorTitleForStatusCode(apiErr.Code), "provider request failed"),
		Cause:        err,
		StatusCode:   apiErr.Code,
		ResponseBody: []byte(apiErr.Message),
		ErrorType:    apiErr.Status,
	}

	parseContextTooLargeError(apiErr.Message, providerErr)

	return providerErr
}

// interactionsErrorBody is the error envelope of the Interactions-style APIs
// such as Decisions. Unlike the generateContent API, its code can be a string
// like "safety" instead of a number.
type interactionsErrorBody struct {
	Error struct {
		Code    json.RawMessage `json:"code"`
		Message string          `json:"message"`
		Status  string          `json:"status"`
	} `json:"error"`
}

func toInteractionsProviderErr(err error, apiErr *apierrors.APIError) *fantasy.ProviderError {
	var body interactionsErrorBody
	_ = json.Unmarshal([]byte(apiErr.Body), &body) // best-effort; falls back to the SDK message on parse failure.

	// NOTE: A numeric code stays empty here, so the status name is used instead.
	var code string
	_ = json.Unmarshal(body.Error.Code, &code)

	message := cmp.Or(body.Error.Message, apiErr.Message)
	providerErr := &fantasy.ProviderError{
		Message:      message,
		Title:        cmp.Or(fantasy.ErrorTitleForStatusCode(apiErr.StatusCode), "provider request failed"),
		Cause:        err,
		StatusCode:   apiErr.StatusCode,
		ResponseBody: []byte(apiErr.Body),
		ErrorType:    cmp.Or(code, body.Error.Status),
	}
	if response := apiErr.RawResponse; response != nil {
		providerErr.ResponseHeaders = httpheaders.ResponseHeaders(response.Header)
		if response.Request != nil && response.Request.URL != nil {
			providerErr.URL = response.Request.URL.String()
		}
	}

	parseContextTooLargeError(message, providerErr)

	return providerErr
}

func parseContextTooLargeError(message string, providerErr *fantasy.ProviderError) {
	matches := googleContextPattern.FindStringSubmatch(message)
	if matches == nil {
		return
	}
	providerErr.ContextTooLargeErr = true
	providerErr.ContextUsedTokens, _ = strconv.Atoi(matches[1])
	providerErr.ContextMaxTokens, _ = strconv.Atoi(matches[2])
}
