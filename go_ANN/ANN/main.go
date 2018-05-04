package main

import (
	"fmt"
	"math/rand"

	"../ANN/NN"
)

// Idea: Update weights in Neuron |x|

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * To get this main() to work
 * go to Network.go and
 * Uncomment Line 56
 * Remove block comment around lines 19 and 76
 * Block comment main() around lines 85 and 118
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*func main() {
	//Chess Over fit data
	x := [][]float32{
		{2, 3, 4, 5, 6, 4, 3, 2,
			1, 1, 1, 1, 1, 1, 1, 1,
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			1, 1, 1, 1, 1, 1, 1, 1,
			2, 3, 4, 5, 6, 4, 3, 2},
		{2, 3, 4, 5, 6, 4, 3, 2,
			1, 1, 1, 1, 0, 1, 1, 1,
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0,
			1, 1, 1, 0, 1, 1, 1, 1,
			2, 3, 4, 5, 6, 4, 3, 2},
		{2, 3, 4, 5, 6, 4, 3, 2,
			1, 1, 1, 0, 0, 1, 1, 1,
			0, 0, 0, 0, 0, 0, 0, 0,
				0, 0, 0, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			1, 0, 0, 1, 0, 0, 0, 0,
			0, 1, 1, 0, 1, 1, 1, 1,
			2, 3, 4, 5, 6, 4, 3, 2},
		{2, 0, 4, 5, 6, 4, 3, 2,
			1, 1, 1, 0, 0, 1, 1, 1,
			0, 0, 3, 0, 0, 0, 0, 0,
				0, 0, 0, 1, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0,
			1, 0, 0, 1, 0, 0, 0, 0,
			0, 1, 1, 3, 1, 1, 1, 1,
			2, 0, 4, 5, 6, 4, 3, 2},
		{2, 0, 4, 5, 6, 4, 3, 2,
			1, 1, 1, 0, 0, 1, 1, 1,
			0, 0, 3, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0,
			1, 0, 0, 1, 0, 0, 0, 0,
			0, 1, 1, 3, 0, 1, 1, 1,
			2, 0, 4, 5, 6, 4, 3, 2}}
	y := [][]float32{{.4, 0.7, .5}, {0.1, 0.7, .5}, {0.2, 0.8, 0.1}, {0.5, 0.7, 1}, {0.4, 0.7, .2}}
	var FNN NN.NeuralNetwork
	var shape = []int{len(x[0]), 16, len(y[0])}
	FNN.NeuralNetwork(shape)
	FNN.Print()

	FNN.Test(x, y)
	for i := 0; i < 1000000; i++ {
		FNN.Train(x, y)
		FNN.Test(x, y)
	}
	FNN.SetDebug(true, false)
	FNN.Print()
	FNN.Test(x, y)
} */

/** ** * * * ** * ** ** *** * ** ** * * * *****    ***** ** *
 * For this main() to work
 * Block comment Line 19 and 76
 * Unblock the  Lines 85 and 118
 * In Network.go comment line 56 and
 * Line 27 assign NN.learnRate = 0.00005
 * * * * * *  * * * * * * *** * * * * ** ** *** ** * *** * **/
func main() {
	fmt.Println("Nothing to report.")
	rand.Seed(3)

	// 3x3 convolution
	//Learnrate = 0.0005
	//shape: 9, 6, 4, 3
	//Tiny data set for convolution
	x := [][]float32{{1, 0, 1, 1, 0, 1, 1, 0, 1}, {1, 0, 1, 0, 0, 1, 0, 0, 0}, {0, 1, 0, 0, 1, 0, 0, 1, 0}, {0, 0, 0, 1, 1, 1, 1, 1, 1}, {1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 1, 0, 1, 0, 1, 0, 0}, {0, 0, 0, 0, 1, 0, 1, 0, 1}, {0, 0, 1, 0, 1, 0, 0, 0, 1}, {1, 0, 1, 0, 0, 0, 1, 0, 1}}
	y := [][]float32{{1, 0, 0}, {0, 0, 1}, {1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}}
	//x := [][]float32{{1, 0, 1}, {1, 1, 0}, {0, 1, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 1}, {1, 1, 1}, {0, 1, 1}}
	//y := [][]float32{{1, 1, 0}, {1, 1, 1}, {0, 1, 1}, {0, 0, 1}, {1, 0, 1}, {0, 1, 0}, {0, 0, 0}, {1, 0, 0}}
	var nn NN.NeuralNetwork
	var shape = []int{len(x[0]), 9, 6, len(y[0])}
	nn.NeuralNetwork(shape)

	nn.Print()
	nn.Test(x, y)
	for i := 0; i < 400001; i++ {
		nn.Train(x, y)
		//nn.Test(x, y)
	}
	nn.SetDebug(true, false)
	nn.Test(x, y)
	x = [][]float32{{1, 0, 0, 0}, {1, 1, 1, 1}}
	y = [][]float32{{0, 0, 1}, {1, 0, 0}}
	nn.SetDebug(true, false)
	nn.Test(x, y)
	nn.Print()
	var n NN.Neuron
	n.Add(0.5)
	n.Activate()
	fmt.Println(n.Out)
}
