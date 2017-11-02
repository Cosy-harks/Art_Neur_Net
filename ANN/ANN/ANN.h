#pragma once
#ifndef _ANN_H
#define _ANN_H

#include <vector>
#include <iostream>
#include <random>
#include "ops.h"
#include "layer.h"

/*this class would be held in [] for hidden
*Idea: make neuron class{
*	int func; // holds a number to differentiate the function it will use out of several
*}
*/

// numInputs is the number of inputs for one set of test data
// IHWeights is the number of weight values between input and hidden nodes
// numHidden is the number of hidden nodes (only 1 layer so far)
// numOutputs is the number of returned values
// *weights points to the weight being worked with
// *weightHead points to the first weight to get back to easily
class NN
{
private:

	int	  Ls;			// # of layers from 0 . . n
	float lrnScalr;		// scale 
	float bias;			// IDK 
	
						// a layer class to hold the neurons
						// and a neuron class to hold the weights
	std::vector<layer> nlayer;
	std::vector<float> target;

	bool  round;

	// moves along the layers
	void forwardPropagation(int lay);
	void backPropagation();
	
	// Recursive function
	float derivative(int rl, int rn, int ll, int ln, int to);

	// updates with node[].nextW[]
	void updateWeights();

public:
	// Idea: take in <int> first elem inputs, last elem out, mid are hidden
	NN(std::vector<int> ls, float eta, int bias, bool roundouts);
	// input training data
	bool trainNN(std::vector<std::vector<float>> da, std::vector<std::vector<float>> ta, int n_iters);
	// Test
	std::vector<float> testANN(std::vector<float> da, std::vector<float> tag);

	// set Learning scalar
	void setLearn(float eta);
	// set Bias
	void setBias(float b);

	float totalError();

	void print();

};

#endif // !_ANN_H
