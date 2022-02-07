// ## The code: A perceptron for classifying points

// ### Imports

// Besides a few standard libraries, we only need a small custom library for drawing the perceptron's output to a PNG.
package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/appliedgo/perceptron/draw"
)

/*
### The perceptron

First we define the perceptron. A new perceptron uses random weights and biases that will be modified during the training process. The perceptron performs two tasks:

* Process input signals
* Adjust the input weights as instructed by the "trainer".

*/

// Our perceptron is a simple struct that holds the input weights and the bias.
type Perceptron struct {
	weights []float32
	bias    float32
}

// This is the Heaviside Step function.
func (p *Perceptron) heaviside(f float32) int32 {
	if f < 0 {
		return 0
	}
	return 1
}

// Create a new perceptron with n inputs. Weights and bias are initialized with random values
// between -1 and 1.
func NewPerceptron(n int32) *Perceptron {
	var i int32
	w := make([]float32, n, n)
	for i = 0; i < n; i++ {
		w[i] = rand.Float32()*2 - 1
	}
	return &Perceptron{
		weights: w,
		bias:    rand.Float32()*2 - 1,
	}
}

// `Process` implements the core functionality of the perceptron. It weighs the input signals,
// sums them up, adds the bias, and runs the result through the Heaviside Step function.
// (The return value could be a boolean but is an int32 instead, so that we can directly
// use the value for adjusting the perceptron.)
func (p *Perceptron) Process(inputs []int32) int32 {
	sum := p.bias
	for i, input := range inputs {
		sum += float32(input) * p.weights[i]
	}
	return p.heaviside(sum)
}

// During the learning phase, the perceptron adjusts the weights and the bias based on how much the perceptron's answer differs from the correct answer.
func (p *Perceptron) Adjust(inputs []int32, delta int32, learningRate float32) {
	for i, input := range inputs {
		p.weights[i] += float32(input) * float32(delta) * learningRate
	}
	p.bias += float32(delta) * learningRate
}

/* ### Training

We rule out the case where the line would be vertical. This allows us to specify the line as a linear function equation:

    f(x) = ax + b

Parameter *a* specifies the gradient of the line (that is, how steep the line is), and *b* sets the offset.

By describing the line this way, checking whether a given point is above or below the line becomes very easy. For a point *(x,y)*, if the value of *y* is larger than the result of *f(x)*, then *(x,y)* is above the line.

See these examples:

![Lines expressed through y = ax + b](separationlines.png)

*/

// *a* and *b* specify the linear function that describes the separation line; see below for details.
// They are defined at global level because we need them in several places and I do not want to
// clutter the parameter lists unnecessarily.
var (
	a, b int32
)

// This function describes the separation line.
func f(x int32) int32 {
	return a*x + b
}

// Function `isAboveLine` returns 1 if the point *(x,y)* is above the line *y = ax + b*, else 0. This is our teacher's solution manual.
func isAboveLine(point []int32, f func(int32) int32) int32 {
	x := point[0]
	y := point[1]
	if y > f(x) {
		return 1
	}
	return 0
}

// Function `train` is our teacher. The teacher generates random test points and feeds them to the perceptron. Then the teacher compares the answer against the solution from the 'solution manual' and tells the perceptron how far it is off.
func train(p *Perceptron, iters int, rate float32) {

	for i := 0; i < iters; i++ {
		// Generate a random point between -100 and 100.
		point := []int32{
			rand.Int31n(201) - 101,
			rand.Int31n(201) - 101,
		}

		// Feed the point to the perceptron and evaluate the result.
		actual := p.Process(point)
		expected := isAboveLine(point, f)
		delta := expected - actual

		// Have the perceptron adjust its internal values accordingly.
		p.Adjust(point, delta, rate)
	}
}

/*
### Showtime!

Now it is time to see how well the perceptron has learned the task. Again we throw random points
at it, but this time there is no feedback from the teacher. Will the perceptron classify every
point correctly?
*/

// This is our test function. It returns the number of correct answers.
func verify(p *Perceptron) int32 {
	var correctAnswers int32 = 0

	// Create a new drawing canvas. Both *x* and *y* range from -100 to 100.
	c := draw.NewCanvas()

	for i := 0; i < 100; i++ {
		// Generate a random point between -100 and 100.
		point := []int32{
			rand.Int31n(201) - 101,
			rand.Int31n(201) - 101,
		}

		// Feed the point to the perceptron and evaluate the result.
		result := p.Process(point)
		if result == isAboveLine(point, f) {
			correctAnswers += 1
		}

		// Draw the point. The colour tells whether the perceptron answered 'is above' or 'is below'.
		c.DrawPoint(point[0], point[1], result == 1)
	}

	// Draw the separation line *y = ax + b*.
	c.DrawLinearFunction(a, b)

	// Save the image as `./result.png`.
	c.Save()

	return correctAnswers
}

// Main: Set up, train, and test the perceptron.
func main() {

	// Set up the line parameters.
	// a (the gradient of the line) can vary between -5 and 5,
	// and b (the offset) between -50 and 50.
	rand.Seed(time.Now().UnixNano())
	a = rand.Int31n(11) - 6
	b = rand.Int31n(101) - 51

	// Create a new perceptron with two inputs (one for x and one for y).
	p := NewPerceptron(2)

	// Start learning.
	iterations := 1000
	var learningRate float32 = 0.1 // Allowed range: 0 < learning rate <= 1.
	// **Try to play with these parameters!**
	train(p, iterations, learningRate)

	// Now the perceptron is ready for testing.
	successRate := verify(p)
	fmt.Printf("%d%% of the answers were correct.\n", successRate)
}
