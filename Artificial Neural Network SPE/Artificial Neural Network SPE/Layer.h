#pragma once
#include <vector>
#include "Node.h"

class Layer
{
public:
	Layer();
	Layer(unsigned int);
	~Layer();

	// add neuron and weights if needed
	void push_neuron(Node);

	// add one more weight to all the neurons
	void push_weights();
	void push_weights(int);

	// performs all the activations of the layer
	void function_of_input();

	// 0 == ReLU, x > 0 == sigmoid, index
	void setFunction(int, unsigned int);

	// creates the pair of matrices multiplied together to start the next layer.
	std::pair<std::vector<double>, std::vector<std::vector<double>>> make_matrices(bool);

	// fill in the net inputs
	void input(std::vector<double>);
	
	// get number of neurons not including bias neuron
	int getSize();

	// node, weight, value
	void setWeight(int, int, double);
	// node, weight indices
	double getWeight(int, int);

	void setdWeight(unsigned int, unsigned int, double);
	double getdWeight(unsigned int, unsigned int);

	double dOut_dIn(int);
	double getOutput(int);

private:
	std::vector<Node> neurons;
	Node bias;
};

