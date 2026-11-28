package fantasy

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestImageCall_Validate(t *testing.T) {
	t.Parallel()

	png := FilePart{Data: []byte("png"), MediaType: "image/png"}

	tests := []struct {
		name    string
		call    ImageCall
		wantErr string
	}{
		{name: "prompt only", call: ImageCall{Prompt: "a cat"}},
		{name: "edit with mask", call: ImageCall{Prompt: "a hat", Files: []FilePart{png}, Mask: &png}},
		{name: "n set", call: ImageCall{Prompt: "a cat", N: new(int64(2))}},
		{
			name:    "zero n",
			call:    ImageCall{Prompt: "a cat", N: new(int64(0))},
			wantErr: "n must be greater than zero, got 0",
		},
		{
			name:    "negative n",
			call:    ImageCall{Prompt: "a cat", N: new(int64(-1))},
			wantErr: "n must be greater than zero, got -1",
		},
		{
			name:    "file without data",
			call:    ImageCall{Prompt: "a cat", Files: []FilePart{png, {MediaType: "image/png"}}},
			wantErr: "file 1 has no data",
		},
		{
			name:    "mask without file",
			call:    ImageCall{Prompt: "a cat", Mask: &png},
			wantErr: "a mask needs a file to edit",
		},
		{
			name:    "mask without data",
			call:    ImageCall{Prompt: "a cat", Files: []FilePart{png}, Mask: &FilePart{MediaType: "image/png"}},
			wantErr: "mask has no data",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			err := tt.call.Validate()
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
