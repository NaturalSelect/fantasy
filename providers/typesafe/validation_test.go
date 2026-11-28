package typesafe

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

type customQuestion struct{}

func (customQuestion) GetType() fantasy.QuestionType { return "custom" }

func TestDecide_ValidationBeforeSending(t *testing.T) {
	t.Parallel()

	tooManyOptions := make([]fantasy.ChoiceOption, 256)
	for i := range tooManyOptions {
		tooManyOptions[i] = fantasy.ChoiceOption{Name: fmt.Sprintf("option-%d", i)}
	}
	tooManyLevels := make([]fantasy.ScoreLevel, 11)
	for i := range tooManyLevels {
		tooManyLevels[i] = fantasy.ScoreLevel{Label: fmt.Sprintf("level-%d", i)}
	}

	tests := []struct {
		name      string
		call      fantasy.DecisionCall
		wantTitle string
		want      string
	}{
		{
			name:      "box without image",
			call:      textCall(map[string]fantasy.Question{"logo": BoxQuestion{Instructions: fantasy.String("Find it")}}),
			wantTitle: "invalid argument",
			want:      `question "logo": a box question needs at least one image`,
		},
		{
			name:      "256 choice options",
			call:      textCall(map[string]fantasy.Question{"tone": fantasy.ChoiceQuestion{Options: tooManyOptions}}),
			wantTitle: "invalid argument",
			want:      `question "tone": a choice question has at most 255 options, got 256`,
		},
		{
			name:      "11 score levels",
			call:      textCall(map[string]fantasy.Question{"urgency": fantasy.ScoreQuestion{Levels: tooManyLevels}}),
			wantTitle: "invalid argument",
			want:      `question "urgency": a score question has at most 10 levels, got 11`,
		},
		{
			name: "webp image",
			call: fantasy.DecisionCall{
				Input:     []fantasy.MessagePart{fantasy.FilePart{Data: []byte("webp"), MediaType: "image/webp"}},
				Questions: map[string]fantasy.Question{"q": fantasy.NoulQuestion{}},
			},
			wantTitle: "invalid argument",
			want:      `image type "image/webp" is not supported`,
		},
		{
			name:      "custom question type",
			call:      textCall(map[string]fantasy.Question{"odd": customQuestion{}}),
			wantTitle: "unsupported",
			want:      `question "odd"`,
		},
		{
			name:      "pointer to a core question type",
			call:      textCall(map[string]fantasy.Question{"odd": &fantasy.NoulQuestion{}}),
			wantTitle: "invalid argument",
			want:      `question "odd"`,
		},
		{
			name:      "pointer to the box question type",
			call:      imageCall(&BoxQuestion{}),
			wantTitle: "unsupported",
			want:      `question "q"`,
		},
		{
			name:      "no input",
			call:      fantasy.DecisionCall{Questions: map[string]fantasy.Question{"q": fantasy.NoulQuestion{}}},
			wantTitle: "invalid argument",
			want:      "input must have at least one part",
		},
		{
			name:      "no questions",
			call:      fantasy.DecisionCall{Input: []fantasy.MessagePart{fantasy.TextPart{Text: "hello"}}},
			wantTitle: "invalid argument",
			want:      "at least one question is required",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			model, server := newTestModel(t, answerAll)
			response, err := model.Decide(t.Context(), tt.call)
			require.Nil(t, response)

			var fantasyErr *fantasy.Error
			require.ErrorAs(t, err, &fantasyErr)
			require.Equal(t, tt.wantTitle, fantasyErr.Title)
			require.Contains(t, fantasyErr.Message, tt.want)
			require.Zero(t, server.count(), "no request may be sent")
		})
	}
}

func TestDecide_Canceled(t *testing.T) {
	t.Parallel()

	t.Run("before the request", func(t *testing.T) {
		t.Parallel()

		model, _ := newTestModel(t, answerAll)
		ctx, cancel := context.WithCancel(t.Context())
		cancel()

		_, err := model.Decide(ctx, noulCall())
		require.ErrorIs(t, err, context.Canceled)
		var providerErr *fantasy.ProviderError
		require.False(t, errors.As(err, &providerErr), "a cancellation is not a provider error")
	})

	t.Run("while waiting for the response", func(t *testing.T) {
		t.Parallel()

		started := make(chan struct{})
		model, _ := newTestModel(t, func(_ http.ResponseWriter, r *http.Request) {
			close(started)
			<-r.Context().Done()
		})
		ctx, cancel := context.WithCancel(t.Context())
		go func() {
			<-started
			cancel()
		}()

		_, err := model.Decide(ctx, noulCall())
		require.ErrorIs(t, err, context.Canceled)
		var providerErr *fantasy.ProviderError
		require.False(t, errors.As(err, &providerErr), "a cancellation is not a provider error")
	})
}
