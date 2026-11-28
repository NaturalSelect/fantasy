package httpheaders

import (
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestResponseHeaders(t *testing.T) {
	t.Parallel()

	in := http.Header{}
	in.Add("Retry-After", "5")
	in.Add("X-Request-Id", "first")
	in.Add("X-Request-Id", "last")
	in["Empty"] = nil

	out := ResponseHeaders(in)

	require.Equal(t, map[string]string{
		"Retry-After":  "5",
		"retry-after":  "5",
		"X-Request-Id": "last",
		"x-request-id": "last",
	}, out)
}

func TestResponseHeaders_Nil(t *testing.T) {
	t.Parallel()
	require.Empty(t, ResponseHeaders(nil))
}
