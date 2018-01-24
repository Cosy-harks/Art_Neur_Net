#pragma once
#include "Layer.h"

class NeuralNetwork
{
public:
	NeuralNetwork();

	NeuralNetwork(std::vector<int>);
	~NeuralNetwork();

	// push a layer with # of neurons in the layer
	void addLayer(unsigned int);

	// first < < vec > > of start values, < < vec > > expected values
	//prints the average error squared
	void test(std::vector<std::vector<double>>, std::vector<std::vector<double>>);

	void train(std::vector<std::vector<double>>, std::vector<std::vector<double>>);

	void print();

private:
	//the layers of the neural network
	std::vector<Layer> collection;

	// the calculated output rather than another layer
	std::vector<double> target;
	std::vector<double> deriveSums;

	double learningRate;
	double sqError;
	int last;

	// moves the information along the network
	void forwardPropagation();
	//layer, node, weight return deltaError_to_deltaWeight
	double backwardPropagation(int, int, int);
	// current working layer, current node, Tended to layer, deriveSum, specific index
	void wideDerivative(int, int, int, int, double &, int &);

	void averageDeltaWeights(int);

	void updateWeights();
};

