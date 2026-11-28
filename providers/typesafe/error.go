package typesafe

import (
	"bytes"
	"cmp"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"charm.land/fantasy"
)

// toProviderErr converts a non-2xx response into a ProviderError. A body with
// a "detail" field, documented for 400 and 422, is summarised, and any other
// body is reported as it came.
func toProviderErr(resp *http.Response, body []byte, requestURL string) *fantasy.ProviderError {
	message, errorType := parseErrorBody(resp, body)
	return &fantasy.ProviderError{
		Title:           cmp.Or(fantasy.ErrorTitleForStatusCode(resp.StatusCode), "provider request failed"),
		Message:         message,
		URL:             requestURL,
		StatusCode:      resp.StatusCode,
		ResponseHeaders: toHeaderMap(resp.Header),
		ResponseBody:    body,
		ErrorType:       errorType,
		// NOTE: The docs mention max_tokens_exceeded without saying where it
		// appears in the body, so match the raw body.
		ContextTooLargeErr: bytes.Contains(body, []byte("max_tokens_exceeded")),
	}
}

func parseErrorBody(resp *http.Response, body []byte) (message, errorType string) {
	var envelope struct {
		Detail json.RawMessage `json:"detail"`
	}
	if json.Unmarshal(body, &envelope) == nil {
		if detail := bytes.TrimSpace(envelope.Detail); len(detail) > 0 {
			switch detail[0] {
			case '"':
				var text string
				if json.Unmarshal(detail, &text) == nil {
					return text, ""
				}
			case '[':
				if msg, typ, ok := validationIssues(detail); ok {
					return msg, typ
				}
			case '{':
				var object struct {
					Code string `json:"code"`
					Type string `json:"type"`
				}
				_ = json.Unmarshal(detail, &object) // best-effort; the error type is only a hint.
				return compactJSON(detail), cmp.Or(object.Code, object.Type)
			}
		}
	}
	if text := strings.TrimSpace(string(body)); text != "" {
		return text, ""
	}
	return cmp.Or(http.StatusText(resp.StatusCode), resp.Status), ""
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

func toHeaderMap(in http.Header) (out map[string]string) {
	out = make(map[string]string, len(in)*2)
	for k, v := range in {
		if l := len(v); l > 0 {
			out[k] = v[l-1]
			out[strings.ToLower(k)] = v[l-1]
		}
	}
	return out
}
