package typesafe

import (
	"encoding/base64"
	"net/http"
	"testing"

	"charm.land/fantasy"
	"github.com/stretchr/testify/require"
)

func TestDecide_RequestBody(t *testing.T) {
	t.Parallel()

	t.Run("choice criteria keep the option order", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		_, err := model.Decide(t.Context(), textCall(map[string]fantasy.Question{
			"tone": fantasy.ChoiceQuestion{
				Instructions: fantasy.String("What is the tone?"),
				Options: []fantasy.ChoiceOption{
					{Name: "zeta"},
					{Name: "alpha", Description: fantasy.String("first")},
				},
			},
		}))
		require.NoError(t, err)

		request := server.last(t)
		require.Equal(t, http.MethodPost, request.method)
		require.Equal(t, "/v1/systemone", request.path)
		require.Equal(
			t,
			`{"model":"jev-latest","state":"hello","questions":{"tone":{"type":"choice","instructions":"What is the tone?","criteria":{"zeta":null,"alpha":"first"}}}}`,
			string(request.body),
		)
	})

	t.Run("choice names and descriptions are escaped and rendered", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		_, err := model.Decide(t.Context(), textCall(map[string]fantasy.Question{
			"q": fantasy.ChoiceQuestion{
				Options: []fantasy.ChoiceOption{
					{Name: `say "hi"`, Description: fantasy.String("two\nlines")},
					{Name: "json", Description: fantasy.JSONText(map[string]int{"a": 1})},
					{Name: "empty", Description: fantasy.String("")},
				},
			},
		}))
		require.NoError(t, err)

		require.Equal(
			t,
			`{"type":"choice","criteria":{"say \"hi\"":"two\nlines","json":"{\"a\":1}","empty":null}}`,
			server.last(t).question(t, "q"),
		)
	})

	t.Run("score criteria are an ordered array of rendered levels", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		_, err := model.Decide(t.Context(), textCall(map[string]fantasy.Question{
			"urgency": fantasy.ScoreQuestion{
				Instructions: fantasy.String("How urgent?"),
				Levels: []fantasy.ScoreLevel{
					{Label: "Low"},
					{Label: "Mid", Description: fantasy.String("middle")},
					{Label: "High", Description: fantasy.String("")},
				},
			},
		}))
		require.NoError(t, err)

		require.Equal(
			t,
			`{"type":"score","instructions":"How urgent?","criteria":["Low","Mid: middle","High"]}`,
			server.last(t).question(t, "urgency"),
		)
	})

	t.Run("noul criteria", func(t *testing.T) {
		t.Parallel()

		tests := []struct {
			name     string
			question fantasy.NoulQuestion
			want     string
		}{
			{
				name: "both descriptions",
				question: fantasy.NoulQuestion{
					Instructions:     fantasy.String("Is it spam?"),
					TrueDescription:  fantasy.String("It is spam"),
					FalseDescription: fantasy.String("It is not spam"),
				},
				want: `{"type":"noul","instructions":"Is it spam?","criteria":{"true":"It is spam","false":"It is not spam"}}`,
			},
			{
				name: "only true",
				question: fantasy.NoulQuestion{
					Instructions:    fantasy.String("Is it spam?"),
					TrueDescription: fantasy.String("It is spam"),
				},
				want: `{"type":"noul","instructions":"Is it spam?","criteria":{"true":"It is spam"}}`,
			},
			{
				name: "only false",
				question: fantasy.NoulQuestion{
					Instructions:     fantasy.String("Is it spam?"),
					FalseDescription: fantasy.String("It is not spam"),
				},
				want: `{"type":"noul","instructions":"Is it spam?","criteria":{"false":"It is not spam"}}`,
			},
			{
				name:     "no descriptions",
				question: fantasy.NoulQuestion{Instructions: fantasy.String("Is it spam?")},
				want:     `{"type":"noul","instructions":"Is it spam?"}`,
			},
			{
				name: "empty descriptions",
				question: fantasy.NoulQuestion{
					Instructions:     fantasy.String("Is it spam?"),
					TrueDescription:  fantasy.String(""),
					FalseDescription: fantasy.String(""),
				},
				want: `{"type":"noul","instructions":"Is it spam?"}`,
			},
		}
		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				t.Parallel()

				model, server := newTestModel(t, answerAll)
				_, err := model.Decide(t.Context(), textCall(map[string]fantasy.Question{"q": tt.question}))
				require.NoError(t, err)
				require.Equal(t, tt.want, server.last(t).question(t, "q"))
			})
		}
	})

	t.Run("box has no criteria", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		call := fantasy.DecisionCall{
			Input:     []fantasy.MessagePart{pngPart("logo")},
			Questions: map[string]fantasy.Question{"logo": BoxQuestion{Instructions: fantasy.String("Find the logo")}},
		}
		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)
		require.Equal(t, `{"type":"box","instructions":"Find the logo"}`, server.last(t).question(t, "logo"))
	})

	t.Run("empty instructions are omitted", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		_, err := model.Decide(t.Context(), textCall(map[string]fantasy.Question{
			"nil":   fantasy.NoulQuestion{},
			"empty": fantasy.ChoiceQuestion{Instructions: fantasy.String(""), Options: []fantasy.ChoiceOption{{Name: "a"}}},
		}))
		require.NoError(t, err)

		request := server.last(t)
		require.Equal(t, `{"type":"noul"}`, request.question(t, "nil"))
		require.Equal(t, `{"type":"choice","criteria":{"a":null}}`, request.question(t, "empty"))
	})

	t.Run("only the documented top level fields are sent", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		_, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)

		var keys []string
		for key := range server.last(t).fields(t) {
			keys = append(keys, key)
		}
		require.ElementsMatch(t, []string{"model", "state", "questions"}, keys)
	})

	t.Run("limits are inclusive", func(t *testing.T) {
		t.Parallel()

		options := make([]fantasy.ChoiceOption, 255)
		for i := range options {
			options[i] = fantasy.ChoiceOption{Name: "option-" + string(rune('A'+i/26)) + string(rune('a'+i%26))}
		}
		levels := make([]fantasy.ScoreLevel, 10)
		for i := range levels {
			levels[i] = fantasy.ScoreLevel{Label: "level-" + string(rune('a'+i))}
		}

		model, server := newTestModel(t, answerAll)
		_, err := model.Decide(t.Context(), textCall(map[string]fantasy.Question{
			"choice": fantasy.ChoiceQuestion{Options: options},
			"score":  fantasy.ScoreQuestion{Levels: levels},
		}))
		require.NoError(t, err)
		require.Equal(t, 1, server.count())
	})
}

func TestDecide_Input(t *testing.T) {
	t.Parallel()

	t.Run("text and images are interleaved in the state", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		call := noulCall()
		call.Input = []fantasy.MessagePart{
			fantasy.TextPart{Text: "before"},
			pngPart("first image"),
			fantasy.TextPart{Text: "between"},
			fantasy.FilePart{Data: []byte("second image"), MediaType: "image/jpeg"},
			fantasy.TextPart{Text: "after"},
		}
		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)

		first, second := imageRef("first image"), imageRef("second image")
		fields := server.last(t).fields(t)
		require.JSONEq(t, `"before\n\n`+first+`\n\nbetween\n\n`+second+`\n\nafter"`, string(fields["state"]))
		require.JSONEq(t, `{
			"`+first+`": {"type": "image", "format": "png", "data_base64": "`+base64.StdEncoding.EncodeToString([]byte("first image"))+`"},
			"`+second+`": {"type": "image", "format": "jpeg", "data_base64": "`+base64.StdEncoding.EncodeToString([]byte("second image"))+`"}
		}`, string(fields["assets"]))
	})

	t.Run("pointer parts work like value parts", func(t *testing.T) {
		t.Parallel()

		valueModel, valueServer := newTestModel(t, answerAll)
		pointerModel, pointerServer := newTestModel(t, answerAll)

		valueCall := noulCall()
		valueCall.Input = []fantasy.MessagePart{fantasy.TextPart{Text: "see"}, pngPart("image")}
		_, err := valueModel.Decide(t.Context(), valueCall)
		require.NoError(t, err)

		pointerCall := noulCall()
		image := pngPart("image")
		pointerCall.Input = []fantasy.MessagePart{&fantasy.TextPart{Text: "see"}, &image}
		_, err = pointerModel.Decide(t.Context(), pointerCall)
		require.NoError(t, err)

		require.Equal(t, string(valueServer.last(t).body), string(pointerServer.last(t).body))
		require.Contains(t, string(pointerServer.last(t).body), `"state":"see\n\n`+imageRef("image")+`"`)
	})

	t.Run("the same image twice yields one asset", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		call := noulCall()
		call.Input = []fantasy.MessagePart{pngPart("same"), pngPart("same")}
		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)

		ref := imageRef("same")
		fields := server.last(t).fields(t)
		require.JSONEq(t, `"`+ref+`\n\n`+ref+`"`, string(fields["state"]))
		require.JSONEq(
			t,
			`{"`+ref+`": {"type": "image", "format": "png", "data_base64": "`+base64.StdEncoding.EncodeToString([]byte("same"))+`"}}`,
			string(fields["assets"]),
		)
	})

	t.Run("jpg is sent as jpeg", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		call := noulCall()
		call.Input = []fantasy.MessagePart{fantasy.FilePart{Data: []byte("photo"), MediaType: "image/jpg"}}
		_, err := model.Decide(t.Context(), call)
		require.NoError(t, err)
		require.Contains(t, string(server.last(t).fields(t)["assets"]), `"format":"jpeg"`)
	})

	t.Run("text only input has no assets", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		_, err := model.Decide(t.Context(), noulCall())
		require.NoError(t, err)
		require.NotContains(t, server.last(t).fields(t), "assets")
	})

	t.Run("the body is identical for identical input", func(t *testing.T) {
		t.Parallel()

		model, server := newTestModel(t, answerAll)
		options := make([]fantasy.ChoiceOption, 20)
		for i := range options {
			options[i] = fantasy.ChoiceOption{Name: string(rune('t' - i)), Description: fantasy.String("d")}
		}
		call := fantasy.DecisionCall{
			Input: []fantasy.MessagePart{
				fantasy.TextPart{Text: "text"},
				pngPart("one"), pngPart("two"), pngPart("three"), pngPart("four"),
			},
			Questions: map[string]fantasy.Question{
				"a": fantasy.NoulQuestion{Instructions: fantasy.String("a")},
				"b": fantasy.ChoiceQuestion{Options: options},
				"c": fantasy.ScoreQuestion{Levels: []fantasy.ScoreLevel{{Label: "x"}, {Label: "y"}}},
				"d": BoxQuestion{Instructions: fantasy.String("d")},
				"e": fantasy.NoulQuestion{Instructions: fantasy.String("e")},
				"f": fantasy.NoulQuestion{Instructions: fantasy.String("f")},
			},
		}

		var bodies []string
		for range 20 {
			_, err := model.Decide(t.Context(), call)
			require.NoError(t, err)
			bodies = append(bodies, string(server.last(t).body))
		}
		for _, body := range bodies {
			require.Equal(t, bodies[0], body)
		}
	})
}
