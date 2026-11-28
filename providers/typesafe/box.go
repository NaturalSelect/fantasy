package typesafe

import "charm.land/fantasy"

// QuestionTypeBox asks for the bounding box of something in an image. It is
// specific to TypeSafe and not supported by jev-1.13.0.
const QuestionTypeBox fantasy.QuestionType = "box"

// BoxQuestion asks the model to locate something in an input image. The input
// must contain at least one image. Read the answer with
// fantasy.AsAnswer[BoxAnswer].
type BoxQuestion struct {
	Instructions fantasy.Text `json:"instructions"`
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
