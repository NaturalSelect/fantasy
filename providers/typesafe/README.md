# TypeSafe Provider

[TypeSafe](https://typesafe.ai) serves System One models such as Jev. They do
not generate text. You send one shared input and several typed questions, and
each answer comes back as a value with probabilities that your code can branch
on.

This provider implements `fantasy.DecisionModelProvider`, not
`fantasy.LanguageModel`. Calling `LanguageModel` returns an error.

## API key

Create a key in the [TypeSafe console](https://console.typesafe.ai/home). The
TypeSafe SDKs read it from `TYPESAFE_API_KEY`. This provider does not read the
environment, so pass the key yourself.

```go
p, err := typesafe.New(typesafe.WithAPIKey(os.Getenv("TYPESAFE_API_KEY")))
if err != nil {
	return err
}
model, err := p.(fantasy.DecisionModelProvider).DecisionModel(ctx, "") // "" is jev-latest
if err != nil {
	return err
}

resp, err := model.Decide(ctx, fantasy.DecisionCall{
	Input: []fantasy.MessagePart{fantasy.TextPart{Text: "I was charged twice this month."}},
	Questions: map[string]fantasy.Question{
		"billing": fantasy.NoulQuestion{Instructions: fantasy.String("Is this message about billing?")},
		"tone": fantasy.ChoiceQuestion{
			Instructions: fantasy.String("What is the tone of this message?"),
			Options: []fantasy.ChoiceOption{
				{Name: "angry", Description: fantasy.String("An upset or hostile message")},
				{Name: "calm", Description: fantasy.String("A neutral or polite message")},
			},
		},
	},
})
if err != nil {
	return err
}
tone, _ := resp.Choice("tone")
fmt.Println(tone.Choice, tone.Probabilities)
```

## Behaviour to know

- **Noul has no confidence.** `NoulAnswer.Confidence` is nil. Choice, Score and
  Box answers carry one.
- **Choice order matters.** Options are sent in the order you give them, and
  TypeSafe leans toward the first option.
- **Limits.** A choice question has 1 to 255 options and a score question has 1
  to 10 levels. Both are checked before the request is sent.
- **Images.** PNG and JPEG only. Each image is sent as an asset and referenced
  from the input text. Size limits are checked by the server.
- **Score levels.** A level is sent as `Label: Description`, or just the label
  when it has no description. `ScoreAnswer.Legend` is built from your labels.
- **Model name.** The response `model` field can echo an alias such as
  `jev-latest` instead of a version.

## Differences from other decision providers

The Google provider also implements `fantasy.DecisionModelProvider` with the
Gemini Decisions API. The core types are the same, but the providers differ.

| | TypeSafe | Gemini |
|---|---|---|
| Default model | `jev-latest` | `gemini-decision-1` |
| Noul confidence | nil | set |
| Choice options | 1 to 255, order kept | 2 to 255, sorted by name by the server |
| Score levels | 1 to 10 | 2 to 255 |
| Instructions | optional | required |
| Image formats | PNG, JPEG | PNG, JPEG, WebP, HEIC, HEIF, GIF, BMP, TIFF |
| Box questions | yes | no, returns an unsupported error |
| Blocked content | not reported | returns a content blocked error |
| Backend | `api.typesafe.ai` | Gemini Developer API only, not Vertex AI |

## Long context

Send a longer input by turning on TypeSafe's long-context mode. It raises the
context limit to 262k tokens and is slower and less accurate.

```go
call.ProviderOptions = typesafe.NewDecisionProviderOptions(&typesafe.DecisionProviderOptions{
	LongContext: new(true),
})
```

## Box questions

`typesafe.BoxQuestion` finds something in an image. It needs at least one image
in the input and a model from Jev 1.14 on, so `jev-1.13.0` rejects it. The
other providers do not support it.

```go
box, ok := fantasy.AsAnswer[typesafe.BoxAnswer](resp.Answers["logo"])
```

`Top` and `Bottom` are fractions of the image height measured from the top.
`Left` and `Right` are fractions of the width measured from the left.

## Retries

The provider sends one request and does not retry. Errors with status 429 and
529 report `IsRetryable()`, and the `retry-after` header is kept on the error.
Wrap the call to retry:

```go
retry := fantasy.RetryWithExponentialBackoffRespectingRetryHeaders[*fantasy.DecisionResponse](fantasy.DefaultRetryOptions())
resp, err := retry(ctx, func() (*fantasy.DecisionResponse, error) {
	return model.Decide(ctx, call)
})
```
