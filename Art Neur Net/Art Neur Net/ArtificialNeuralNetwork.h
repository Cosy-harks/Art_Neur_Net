#pragma once
#include <vector>
#ifndef _ArtificialNeuralNetwork_H

#include <iostream>
#include <random>

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
	int   in;			// inputs for one set of outputs
	int   IHWeights;	// weights between inputs and hidden
	int   hid;			// hidden neurons
	int	  out;			// set size of output
	float lrnScalr;		// scale 
	vector<float> dataVals;	// data values
	float *dvh;			// data values head
	float *weights;		// pointer to current working weight 
	float *weightHead;	// pointer to first weight

	bool  round;

	void setWeights(int weightCount);



public:
	// Idea: take in [int] first elem inputs, last elem out, mid are hidden
	// input #of, hidden #of, output #of, round outputs(True/False)
	NN(int numberOfInputs, int numberOfHidden, int numberOfOutputs, float eta, bool roundouts);
	// input training data
	bool trainNN(float[[]] data, float[[]] target, int n_iters);
	// set Learning scalar
	void setLearn(float eta);

	~NN();

};

#endif // !_ArtificialNeuralNetwork_H
