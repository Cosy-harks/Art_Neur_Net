package NN

import (
	"fmt"
	"math/rand"
)

/*	Methods implemented
 *
 *	func (N *Neuron) Neuron()
 *	func (N *Neuron) push()
 *	func (N *Neuron) pushes(number int)
 *	func (N *Neuron) setWeight(w int, value float32)
 *	func (N *Neuron) setDeltaWeight(w int, value float32)
 *	func (N *Neuron) setWeights(values []float32)
 *	func (N *Neuron) setDeltaWeights(values []float32)
 *	func (N *Neuron) Add(value float32)
 *	func (N *Neuron) Activate()
 *
 *
 *	Methods Needing work
 *
 *	func (N *Neuron) update()
 */

// Neuron is the smallest structure of the Neural Network
type Neuron struct {
	weight         []float32
	deltaWeight    []float32
	deriveSum      float32
	in             float32
	Out            float32
	activationF    int
	dOutdIn        float32
	activateThresh float32
}

// Neuron constructor
func (N *Neuron) Neuron() {
	N.in = 0
	N.Out = 0
	N.activateThresh = rand.Float32()/2 - 0.2

	// update value if I update number of switch option
	N.activationF = rand.Intn(3)
}

// push a weight and deltaWeight onto the neuron
func (N *Neuron) push() {
	N.weight = append(N.weight, rand.Float32())
	N.deltaWeight = append(N.deltaWeight, 0.0)
}

func (N *Neuron) pushes(number int) {
	for i := 0; i < number; i++ {
		N.push()
	}
}

// set one of the weights
func (N *Neuron) setWeight(w int, value float32) {
	if len(N.weight) > w {
		N.weight[w] = value
	} else {
		fmt.Printf("Selected weight %v is out of range", w)
	}
}

// set one of the delta weights
func (N *Neuron) setDeltaWeight(w int, value float32) {
	if len(N.deltaWeight) > w {
		N.deltaWeight[w] = value
	} else {
		fmt.Printf("Selected weight %v is out of range", w)
	}
}

// setWeights puts in new weight values if array lengths are ==
func (N *Neuron) setWeights(values []float32) {
	if len(values) != len(N.weight) {
		fmt.Println("length of given weight values != length of weights value")
		fmt.Printf("%v != %v", len(values), len(N.weight))
	} else {
		N.weight = values
	}
}

// setWeights puts in new delta weight values if array lengths are ==
func (N *Neuron) setDeltaWeights(values []float32) {
	if len(values) != len(N.deltaWeight) {
		fmt.Println("length of given weight values != length of weights value")
		fmt.Printf("%v != %v", len(values), len(N.deltaWeight))
	} else {
		N.deltaWeight = values
	}
}

// Add value to the input of this neuron
func (N *Neuron) Add(value float32) {
	N.in += value
}

// Activate this neuron
func (N *Neuron) Activate() {
	// If I update number of cases update constructor
	/*
		switch N.activationF {
		case 0: // ReLU
			// If in > 0 derivative(Out)/derivative(In) = 1
			// else derivative(Out)/derivative(In) = 0
	*/
	if N.in >= N.activateThresh {
		N.Out = N.in
		N.dOutdIn = 1
	} else {
		N.Out = 0
		N.dOutdIn = 0
	}
	N.in = 0
	/*break
	case 1: // Sigmoid
		N.Out = 1.0 / (1.0 + float32(math.Exp((float64)(-N.in))))
		N.dOutdIn = N.Out * (1 - N.Out)
		N.in = 0
		break
	case 2: // All or nothing
		if N.in < N.activateThresh {
			N.Out = 0
		} else {
			N.Out = 1
		}
		N.dOutdIn = 0
		N.in = 0
		break
	default:
		break
	}*/
}

// DeActivate rverse of Activate
func (N *Neuron) DeActivate() {
	/*switch N.activationF {
	case 0: // ReLU
		// If in > 0 derivative(Out)/derivative(In) = 1
		// else derivative(Out)/derivative(In) = 0
	*/
	if N.in >= N.activateThresh {
		N.in = N.Out
		N.dOutdIn = 1
	} else {
		N.in = 0
		N.dOutdIn = 0
	}
	N.Out = 0
	/*break
	case 1: // Sigmoid
		N.in = (float32)(math.Log((float64)(N.Out / (1.0 - N.Out))))
		N.dOutdIn = N.Out * (1 - N.Out)
		N.in = 0
		break
	case 2: // All or nothing
		if N.in < N.activateThresh {
			N.Out = 0
		} else {
			N.Out = 1
		}
		N.dOutdIn = 0
		N.in = 0
		break
	default:
		break
	}*/
}

//Update weights by deltaWeights
func (N *Neuron) update() {
	for i, v := range N.deltaWeight {
		N.weight[i] += v
	}
}
