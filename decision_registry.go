package fantasy

import (
	"fmt"
	"sync"
)

// QuestionDecoder converts the JSON of a question into a Question.
type QuestionDecoder func([]byte) (Question, error)

// AnswerDecoder converts the JSON of an answer into an Answer.
type AnswerDecoder func([]byte) (Answer, error)

var (
	questionDecoders sync.Map
	answerDecoders   sync.Map
)

// RegisterQuestionType registers the decoder of a question type that a
// provider adds on top of the built-in ones, so that a DecisionCall holding it
// survives a JSON round trip. Question types must have a globally unique ID.
// This should only be called during package initialization (init functions).
func RegisterQuestionType(questionType QuestionType, decode QuestionDecoder) {
	questionDecoders.Store(questionType, decode)
}

// RegisterAnswerType registers the decoder of an answer type that a provider
// adds on top of the built-in ones, so that a DecisionResponse holding it
// survives a JSON round trip. Answer types share the IDs of their questions.
// This should only be called during package initialization (init functions).
func RegisterAnswerType(questionType QuestionType, decode AnswerDecoder) {
	answerDecoders.Store(questionType, decode)
}

func decodeQuestion(questionType QuestionType, data []byte) (Question, error) {
	decode, ok := questionDecoders.Load(questionType)
	if !ok {
		return nil, fmt.Errorf("unknown question type: %s", questionType)
	}
	return decode.(QuestionDecoder)(data) //nolint:forcetypeassert // type enforced by RegisterQuestionType
}

func decodeAnswer(questionType QuestionType, data []byte) (Answer, error) {
	decode, ok := answerDecoders.Load(questionType)
	if !ok {
		return nil, fmt.Errorf("unknown answer type: %s", questionType)
	}
	return decode.(AnswerDecoder)(data) //nolint:forcetypeassert // type enforced by RegisterAnswerType
}
