package typesafe

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"charm.land/fantasy"
	"charm.land/fantasy/providers/internal/httpheaders"
)

// toProviderErr converts a non-2xx response into a ProviderError. A body with
// a "detail" field, documented for 400 and 422, is summarised, and any other
// body is reported as it came.
func toProviderErr(resp *http.Response, body []byte, requestURL string) *fantasy.ProviderError {
	message, errorType, maxTokens := parseErrorBody(resp, body)
	return &fantasy.ProviderError{
		Title:              cmp.Or(fantasy.ErrorTitleForStatusCode(resp.StatusCode), "provider request failed"),
		Message:            message,
		URL:                requestURL,
		StatusCode:         resp.StatusCode,
		ResponseHeaders:    httpheaders.ResponseHeaders(resp.Header),
		ResponseBody:       body,
		ErrorType:          errorType,
		ContextTooLargeErr: resp.StatusCode == http.StatusBadRequest && maxTokens,
	}
}

func parseErrorBody(resp *http.Response, body []byte) (message, errorType string, maxTokens bool) {
	const code = "max_tokens_exceeded"
	var envelope struct {
		Detail json.RawMessage `json:"detail"`
		Error  json.RawMessage `json:"error"`
	}
	if json.Unmarshal(body, &envelope) == nil {
		var errorCode string
		maxTokens = json.Unmarshal(envelope.Error, &errorCode) == nil && errorCode == code
		if detail := bytes.TrimSpace(envelope.Detail); len(detail) > 0 {
			switch detail[0] {
			case '"':
				var text string
				if json.Unmarshal(detail, &text) == nil {
					kind, _, _ := strings.Cut(text, ":")
					return text, "", maxTokens || kind == code
				}
			case '[':
				if msg, typ, ok := validationIssues(detail); ok {
					return msg, typ, maxTokens
				}
			case '{':
				var object struct {
					Code string `json:"code"`
					Type string `json:"type"`
				}
				_ = json.Unmarshal(detail, &object) // best-effort; the error type is only a hint.
				kind := cmp.Or(object.Code, object.Type)
				return compactJSON(detail), kind, maxTokens || kind == code
			}
		}
	}
	if text := strings.TrimSpace(string(body)); text != "" {
		return text, "", maxTokens
	}
	return cmp.Or(http.StatusText(resp.StatusCode), resp.Status), "", maxTokens
}

// validationIssues formats the "detail" list of a 422 response as
// "loc: msg" entries and returns the type of the first one.
func validationIssues(detail []byte) (message, errorType string, ok bool) {
	var issues []struct {
		Loc  []any  `json:"loc"`
		Msg  string `json:"msg"`
		Type string `json:"type"`
	}
	if json.Unmarshal(detail, &issues) != nil || len(issues) == 0 {
		return "", "", false
	}
	entries := make([]string, 0, len(issues))
	for _, issue := range issues {
		loc := make([]string, 0, len(issue.Loc))
		for _, part := range issue.Loc {
			loc = append(loc, fmt.Sprint(part))
		}
		if len(loc) == 0 {
			entries = append(entries, issue.Msg)
			continue
		}
		entries = append(entries, strings.Join(loc, ".")+": "+issue.Msg)
	}
	return strings.Join(entries, "; "), issues[0].Type, true
}

func compactJSON(data []byte) string {
	var out bytes.Buffer
	if err := json.Compact(&out, data); err != nil {
		return string(data)
	}
	return out.String()
}
