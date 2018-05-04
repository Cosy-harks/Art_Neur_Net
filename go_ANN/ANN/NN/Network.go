package NN

import (
	"fmt"
	"math"

	"../MyMath"
	"../MyMath/MatrixMath"
)

// NeuralNetwork -
// collection is the layers of the network
type NeuralNetwork struct {
	collection []Layer
	learnRate  float32
	lastIndex  int
	debug      debuge
}

// NeuralNetwork ctor
func (NN *NeuralNetwork) NeuralNetwork(shape []int) {
	r := len(shape)
	for i := 0; i < r-1; i++ {
		NN.collection = append(NN.collection, layerCtor(shape[i], shape[i+1]))
	}
	NN.collection = append(NN.collection, layerCtor(shape[r-1], 0))
	NN.learnRate = 0.00005
	NN.lastIndex = len(NN.collection) - 1
	NN.debug = debuge{true, false}
}

type debuge struct {
	test    bool
	forward bool
}

// Test the Network with
// start data, expected output
func (NN *NeuralNetwork) Test(starts [][]float32, expects [][]float32) {
	var errSum float32
	for loopInner := 0; loopInner < len(starts); loopInner++ {
		NN.ForewardPropagation(starts[loopInner])
		answer := NN.collection[NN.lastIndex].getOutput(false)
		err := MyMath.TotalError(answer, expects[loopInner])
		errSum += err
		if NN.debug.test {
			fmt.Printf("%v - ", expects[loopInner])
			fmt.Printf("%v\t%v\n", answer, err)

		}
	}
	errSum /= (float32)(len(starts))
	i := 1
	for ; errSum/(float32)(math.Pow10(i)) > 1.0; i++ {
	}
	//NN.learnRate = 0.1 / (float32)(math.Pow10(i))
	if NN.debug.test {
		fmt.Println()
		NN.debug.test = false
	}
}

// SetDebug sets debug bools
func (NN *NeuralNetwork) SetDebug(test, fore bool) {
	NN.debug.test = test
	NN.debug.forward = fore
}

// Train the network
func (NN *NeuralNetwork) Train(starts [][]float32, expects [][]float32) {
	//Make a batchTrain later that will batch the inputs and send them here

	// TODO: fill in the training
	for elem := 0; elem < len(starts); elem++ {
		NN.ForewardPropagation(starts[elem])
		NN.parseWeights(expects[elem], NN.backwardPropagation)
		NN.parseWeights(expects[elem], NN.clear)
		//NN.finalPass()
	}
	NN.parseWeights(expects[0], NN.update)
}

// I want to make both recursive and loopy backwardPropagation functions
func (NN *NeuralNetwork) parseWeights(expects []float32, fn func(int, int, int, []float32)) {
	// TODO: Backward propagate to try correcting network
	// Idea try going backwards with the expected values
	//figure out what input would provide the correct output.
	does := MyMath.DeltaErrors(NN.collection[NN.lastIndex].getOutput(false), expects)

	if MyMath.AbsAverage(does) > 0.00002 {
		//depth - 1. as to be able to check next layer
		for L := 0; L < len(NN.collection)-1; L++ {
			//include bias
			for N := 0; N <= len(NN.collection[L].neurons); N++ {
				// neurons of the next layer
				for W := 0; W < len(NN.collection[L+1].neurons); W++ {
					fn(L, N, W, does)
				}
			}
		}
	}

	/*	for _, v := range does {
		fmt.Printf("%v ", v)
	*/
	//fmt.Println()
}

/*
Min
dErr/dW4,0 = dErr/dOut5,0 (dOut5,0/dnet5,0 (dnet5,0/dW4,0))

dErr/dW3,4 = (dErr/dOut5,0 (dOut5,0/dnet5,0 (dnet5,0/dOut4,0))
			+ dErr/dOut5,1 (dOut5,1/dnet5,1 (dnet5,1/dOut4,0)))
			*(dOut4,0/dnet4,0 * dnet4,0/dW3,4)

dErr/dW2,3 = (dErr/dOut5,0 (dOut5,0/dnet5,0 (dnet5,0/dOut4,0 (dOut4,0/dnet4,0 (dnet4,0/dOut3,1))
										+    dnet5,0/dOut4,1 (dOut4,1/dnet4,1 (dnet4,1/dOut3,1))))
			+ dErr/dOut5,1 (dOut5,1/dnet5,1 (dnet5,1/dOut4,0 (dOut4,0/dnet4,0 (dnet4,0/dOut3,1))
										+    dnet5,1/dOut4,1 (dOut4,1/dnet4,1 (dnet4,1/dOut3,1)))))
			* (dOut3,1/dnet3,1 * dnet3,1/dW2,3)

dErr/dW1,5 =
(dErr/dOut5,0 (dOut5,0/dnet5,0 (dnet5,0/dOut4,0 (dOut4,0/dnet4,0 (dnet4,0/dOut3,0 (dOut3,0/dnet3,0 (dnet3,0/dOut2,0 (dOut2,0/dnet2,0 * dnet2,0/dW1,5)))
																+ dnet4,0/dOut3,1 (dOut3,1/dnet3,1 (dnet3,1/dOut2,0 (dOut2,0/dnet2,0 * dnet2,0/dW1,5)))))
							 +  dnet5,0/dOut4,1 (dOut4,1/dnet4,1 (dnet4,1/dOut3,0 (dOut3,0/dnet3,0 (dnet3,0/dOut2,0 (dOut2,0/dnet2,0 * dnet2,0/dW1,5)))
																+ dnet4,0/dOut3,1 (dOut3,1/dnet3,1 (dnet3,1/dOut2,0 (dOut2,0/dnet2,0 * dnet2,0/dW1,5)))))))
+dErr/dOut5,1 (dOut5,1/dnet5,1 (dnet5,1/dOut4,0 (dOut4,0/dnet4,0 (dnet4,0/dOut3,0 (dOut3,0/dnet3,0 (dnet3,0/dOut2,0 (dOut2,0/dnet2,0 * dnet2,0/dW1,5)))
																+ dnet4,0/dOut3,1 (dOut3,1/dnet3,1 (dnet3,1/dOut2,0 (dOut2,0/dnet2,0 * dnet2,0/dW1,5)))))
							 +  dnet5,1/dOut4,1 (dOut4,1/dnet4,1 (dnet4,1/dOut3,0 (dOut3,0/dnet3,0 (dnet3,0/dOut2,0 (dOut2,0/dnet2,0 * dnet2,0/dW1,5)))
																+ dnet4,0/dOut3,1 (dOut3,1/dnet3,1 (dnet3,1/dOut2,0 (dOut2,0/dnet2,0 * dnet2,0/dW1,5))))))))

1 node per layer no bias
dErr/dW0 = (dErr/doutn * doutn/dnetn * [dnetn/doutn-1) *doutn-1/dnetn-1 * (dnetn-1/doutn-2] *...* dout2/dnet2 * [dnet2/dout1) * dout1/dnet1 * dnet1/dW0]

groups of three third/first may be reused on the layer transition

if L+1 < NN.lastIndex {
		cwl := 0
		// if more than three from the last index
		if L+3 < NN.lastIndex {
			cwl = L + 3
		} else {
			cwl = NN.lastIndex
		}
		// layer loop
		for ; cwl < len(NN.collection)-1; cwl++ {
			// node loop
			for cn := 0; cn < len(NN.collection[cwl].neurons); cn++ {
				// node for next layer
				for n := 0; n < len(NN.collection[cwl+1].neurons); n++ {

					///////////////////////
					// idea: node hold the derivesum thing from c++ ANN
					///////////////////////

					// if cwl is more than three layers to the right of L
					if cwl > L+3 {
						//deriveSums[cwl][cn] += collection[cwl - 1].getWeight(n, cn)*collection[cwl - 1].dOut_dIn(n)*deriveSums[cwl - 1][n];
						NN.collection[cwl].neurons[cn].deriveSum += NN.collection[cwl-1].neurons[n].weight[cn] * NN.collection[cwl-1].neurons[n].dOutdIn * NN.collection[cwl-1].neurons[n].deriveSum
					} else if cwl == L+3 {
						//deriveSums[cwl][cn] += collection[L + 2].getWeight(n, cn)*collection[L + 2].dOut_dIn(n)*collection[L + 1].getWeight(W, n);
						NN.collection[cwl].neurons[cn].deriveSum += NN.collection[cwl-1].neurons[n].weight[cn] * NN.collection[cwl-1].neurons[n].dOutdIn * NN.collection[cwl-2].neurons[W].weight[n]
					} else if cwl == L+2 {
						//  somemore deriveSums[cwl][cn] = 1; uhhmmmm, what?
						NN.collection[cwl].neurons[cn].deriveSum = 1.0
					}
				}
			}
		}
	}


On the dnet/dout1 add dnet/dout2
*/

// ForewardPropagation processes the input data
func (NN *NeuralNetwork) ForewardPropagation(beginning []float32) {
	O := beginning
	var W [][]float32
	for i := 0; i < len(NN.collection); i++ {
		NN.collection[i].setInputs(O)
		NN.collection[i].processInput()
		if NN.debug.forward {
			fmt.Printf("Out %v: %v\n", i, NN.collection[i].getOutput(true))
		}
		W, O = NN.collection[i].makeMatrices(true)
		O = MatrixMath.MatrixMult(O, W)
		if NN.debug.forward {
			fmt.Printf("In %v: %v\n", i+1, O)
		}
	}
	return
}

func (NN *NeuralNetwork) backwardPropagation(L, N, W int, errs []float32) {
	// [L+2, NN.lastIndex]
	for i := L + 2; i <= NN.lastIndex; i++ {
		// each neuron affected by the LNW weight
		for j := 0; j < len(NN.collection[i].neurons); j++ {
			// beginning
			if i == L+2 {
				if N < len(NN.collection[L].neurons) { //dnet/dw	|x|				*			dout/din	|x|				*			dnet_i/dout_i-1		|x|			* 			dout/dnet	|x|
					NN.collection[i].neurons[j].deriveSum = NN.collection[L].neurons[N].Out * NN.collection[i-1].neurons[W].dOutdIn * NN.collection[i-1].neurons[W].weight[j] * NN.collection[i].neurons[j].dOutdIn
				} else if N == len(NN.collection[L].neurons) { //dnet/dw	|x|			*				dout/din	|x|			*			dnet_i/dout_i-1		|x|		*	 		dout/dnet	|x|
					NN.collection[i].neurons[j].deriveSum = NN.collection[L].bias.Out * NN.collection[i-1].neurons[W].dOutdIn * NN.collection[i-1].neurons[W].weight[j] * NN.collection[i].neurons[j].dOutdIn
				}
				//After beginning
			} else {
				// self.deriveSum = (sum of deriveSums from previous layer) then multiply by self.dOutdIn
				for k := 0; k < len(NN.collection[i-1].neurons); k++ {
					NN.collection[i].neurons[j].deriveSum += NN.collection[i-1].neurons[k].deriveSum * NN.collection[i-1].neurons[k].weight[j]
				}
				NN.collection[i].neurons[j].deriveSum *= NN.collection[i].neurons[j].dOutdIn //self.dOutdIn = 0 || 1
			}
		}
	}
	// Did not make it into the initial for loop
	if L+1 == NN.lastIndex {
		//Minimum work
		//One of the neurons
		if N < len(NN.collection[L].neurons) {
			NN.collection[L].neurons[N].deltaWeight[W] += -errs[W] * NN.collection[L].neurons[N].dOutdIn * NN.collection[L].neurons[N].Out // dTotalErr/dOut[W] * dOutdIn[W] * dIn/dW[W]
			//The bias
		} else if N == len(NN.collection[L].neurons) {
			NN.collection[L].bias.deltaWeight[W] += -errs[W] * NN.collection[L].bias.dOutdIn * NN.collection[L].bias.Out // dTotalErr/dOut[W] * dOutdIn[W] * dIn/dW[W]
		}
		// After initial for loop is executed
	} else {
		//Output layer
		for b := 0; b < len(NN.collection[NN.lastIndex].neurons); b++ {
			//if working on neuron weight
			if N < len(NN.collection[L].neurons) {
				NN.collection[L].neurons[N].deltaWeight[W] += -errs[b] * NN.collection[NN.lastIndex].neurons[b].deriveSum
				// else if working on bias weight
			} else if N == len(NN.collection[L].neurons) {
				NN.collection[L].bias.deltaWeight[W] += -errs[b] * NN.collection[NN.lastIndex].neurons[b].deriveSum
			}
		}
		// Allows 4 Layer networks to work
		if L+2 < NN.lastIndex {
			if N < len(NN.collection[L].neurons) {
				NN.collection[L].neurons[N].deltaWeight[W] *= -1
				// else if working on bias weight
			} else if N == len(NN.collection[L].neurons) {
				NN.collection[L].bias.deltaWeight[W] *= -1
			}
		}
	}
}

func (NN *NeuralNetwork) clear(L, N, W int, dummy []float32) {
	if N < len(NN.collection[L].neurons) {
		NN.collection[L].neurons[N].deriveSum = 0.0
	} else {
		NN.collection[L].bias.deriveSum = 0.0
	}
}

func (NN *NeuralNetwork) update(L, N, W int, dummy []float32) {
	if N < len(NN.collection[L].neurons) {
		NN.collection[L].neurons[N].weight[W] += NN.collection[L].neurons[N].deltaWeight[W] / (float32)(len(dummy)) * NN.learnRate
		NN.collection[L].neurons[N].deltaWeight[W] = 0.0
	} else {
		//fmt.Printf("dw: %v, L: %v, N: %v, W: %v", NN.collection[L].bias.deltaWeight[W], L, N, W)
		NN.collection[L].bias.weight[W] += NN.collection[L].bias.deltaWeight[W] / (float32)(len(dummy)) * NN.learnRate
		NN.collection[L].bias.deltaWeight[W] = 0.0
	}
}

// BackwardPropagation is hard
func (NN *NeuralNetwork) lastWeights() {
	last := NN.lastIndex - 1
	for n := 0; n <= len(NN.collection[last].neurons); n++ {
		for w := 0; w < len(NN.collection[last].neurons[n].weight); w++ {
			// what to do? dErr/dOut
			val := NN.collection[last].neurons[n].dOutdIn * NN.collection[last].neurons[n].weight[w] // * dErr/dOut
			NN.collection[last].neurons[n].setDeltaWeight(w, val)
		}
	}
}

// Print the Network learnRate, shape and weights
func (NN *NeuralNetwork) Print() {
	fmt.Printf("Learnrate = %v\n", NN.learnRate)
	fmt.Print("shape: ")
	for a := 0; a < len(NN.collection); a++ {
		fmt.Printf("%v", len(NN.collection[a].neurons))
		if a < len(NN.collection)-1 {
			fmt.Print(", ")
		}
	}
	fmt.Println()
	fmt.Println()
	//NN.collection[0].makeMatrices(false)
	for _, v := range NN.collection {
		ws, _ := v.makeMatrices(true)
		for i := 0; i < len(ws); i++ {
			fmt.Printf("%v: %v\n", i, ws[i])
		}

	}
}
