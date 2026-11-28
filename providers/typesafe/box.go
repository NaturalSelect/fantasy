package typesafe

import (
	"encoding/json"

	"charm.land/fantasy"
)

// QuestionTypeBox asks for the bounding box of something in an image. It is
// specific to TypeSafe and not supported by jev-1.13.0.
const QuestionTypeBox fantasy.QuestionType = "box"

// BoxQuestion asks the model to locate something in an input image. The input
// must contain at least one image. Read the answer with
// fantasy.AsAnswer[BoxAnswer].
type BoxQuestion struct {
	Instructions fantasy.Text `json:"instructions"`
}

type boxQuestionJSON struct {
	Instructions string `json:"instructions"`
}

func (q BoxQuestion) MarshalJSON() ([]byte, error) {
	text, err := fantasy.RenderText(q.Instructions)
	if err != nil {
		return nil, err
	}
	return json.Marshal(boxQuestionJSON{Instructions: text})
}

func (q *BoxQuestion) UnmarshalJSON(data []byte) error {
	var v boxQuestionJSON
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	q.Instructions = fantasy.String(v.Instructions)
	return nil
}

func init() {
	fantasy.RegisterQuestionType(QuestionTypeBox, decodeBoxQuestion)
	fantasy.RegisterAnswerType(QuestionTypeBox, decodeBoxAnswer)
}

func decodeBoxQuestion(data []byte) (fantasy.Question, error) {
	var q BoxQuestion
	if err := json.Unmarshal(data, &q); err != nil {
		return nil, err
	}
	return q, nil
}

func decodeBoxAnswer(data []byte) (fantasy.Answer, error) {
	var a BoxAnswer
	if err := json.Unmarshal(data, &a); err != nil {
		return nil, err
	}
	return a, nil
}

// GetType implements fantasy.Question.
func (BoxQuestion) GetType() fantasy.QuestionType { return QuestionTypeBox }

// BoxAnswer answers a BoxQuestion with the edges of the bounding box, as
// fractions from 0 to 1. Top and Bottom are measured from the top of the
// image, Left and Right from the left.
type BoxAnswer struct {
	Top    float64 `json:"top"`
	Bottom float64 `json:"bottom"`
	Left   float64 `json:"left"`
	Right  float64 `json:"right"`
	// Confidence is nil when TypeSafe reports none.
	Confidence *float64 `json:"confidence"`
}

// GetType implements fantasy.Answer.
func (BoxAnswer) GetType() fantasy.QuestionType { return QuestionTypeBox }
