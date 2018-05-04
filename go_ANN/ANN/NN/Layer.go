package NN

//	/////////////////////////////
//	class - Layer - holds group of neurons that connect to another layer
//	///////////////////////////

/*	Methods implemented in order
 *
 *	func 			layerCtor(neurons, weightCount int) (L Layer)
 *	func (L *Layer) pushNeuron()
 *	func (L *Layer) pushWeights()
 *	func (L *Layer) setWeightsOf(neuron int, neWeights []float32)
 *	func (L  Layer) makeMatrices(withBias bool) (weights [][]float32, outputs []float32)
 *	func (L *Layer) processInput()
 *	func (L *Layer) setInputs(ins []float32)
 *	func (L *Layer) getOutput(withBias bool) (outputs []float32)
 *	func (L  Layer) getWeights(withBias bool) (weights [][]float32)
 *
 */

// Layer is a group of Neuron objects
type Layer struct {
	neurons []Neuron
	bias    Neuron
}

func layerCtor(neurons, weightCount int) (L Layer) {
	for i := 0; i < neurons; i++ {
		L.pushNeuron()
		L.neurons[i].pushes(weightCount)
	}
	L.bias.pushes(weightCount)
	return
}

func (L *Layer) pushNeuron() {
	var n Neuron
	L.neurons = append(L.neurons, n)
}

func (L *Layer) pushWeights() {
	for i := 0; i < len(L.neurons); i++ {
		L.neurons[i].push()
	}
	L.bias.push()
}

func (L *Layer) setWeightsOf(neuron int, neWeights []float32) {
	L.neurons[neuron].setWeights(neWeights)
}

// true includes bias false does !include bias
func (L Layer) makeMatrices(withBias bool) (weights [][]float32, outputs []float32) {
	weights = L.getWeights(withBias)
	outputs = L.getOutput(withBias)
	return
}

func (L *Layer) processInput() {
	for i := 0; i < len(L.neurons); i++ {
		L.neurons[i].Activate()
	}
	L.bias.Activate()
}

func (L *Layer) setInputs(ins []float32) {
	for i := 0; i < len(ins); i++ {
		L.neurons[i].Add(ins[i])
	}
	L.bias.Add(1.0)
}

func (L *Layer) getOutput(withBias bool) (outputs []float32) {
	for i := 0; i < len(L.neurons); i++ {
		outputs = append(outputs, L.neurons[i].Out)
	}
	if withBias {
		outputs = append(outputs, L.bias.Out)
	}
	return
}

func (L Layer) getWeights(withBias bool) (weights [][]float32) {
	for i := 0; i < len(L.neurons); i++ {
		weights = append(weights, L.neurons[i].weight)
	}
	if withBias {
		weights = append(weights, L.bias.weight)
	}
	return
}
