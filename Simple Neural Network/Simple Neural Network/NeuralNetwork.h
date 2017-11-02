#pragma once
#ifndef _NeuralNetwork_H
#define _NeuealNetwork_H

#include <vector>
#include <iostream>
#include <random>
#include <tuple>
#include "matrixOps.h"
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
	int   in;			// inputs for one set of outputs
	int   IHWeights;	// weights between inputs and hidden
	int   hid;			// hidden neurons
	int	  out;			// set size of output
	int	  Ls;			// # of layers from 0 . . n
	float lrnScalr;		// scale 
	float bias;			// IDK

	// to update out layer of weights
	float dErrtotal_dw;
	float dnet_o_dw;	// 1st
	float dout_o_dnet_o;// 2nd
	float dEt_dout_o;	// 3rd


	//Make a layer class
	std::vector<layer> nlayer;
	std::vector<float> target;

	std::vector<float> E_out;
	std::vector<float> out_o;
	std::vector<float> net_o;
	std::vector<float> out_h;
	std::vector<float> net_h;

	std::vector<float> nextWeights;
	std::vector<float> dataVals;	// data values
	std::vector<float> OutputBetas;
	std::vector<float> HiddenBetas;
	std::vector<float> HiddenVals;
	std::vector<float> weights;		// pointer to current working weight
	std::vector<float> dweights;	// change in weights'
	bool  round;

	void randWeights(int weightCount);



	// moves along the layers
	void forwardPropagation(int lay);
	std::pair<std::vector<float>, std::vector<std::vector<float>>> forwardProp(std::vector<float> input, int fromLayer);
	void backPropagation();
	void backProp(std::vector<float> output, std::vector<float> target, std::vector<float> input, int layer);

	float derivative(int rl, int rn, int ll, int ln, int to);

	void updateWeights();

public:
	// Idea: take in [int] first elem inputs, last elem out, mid are hidden
	// input #of, hidden #of, output #of, round outputs(True/False)
	NN(std::vector<int> ls, int numberOfInputs, int numberOfHidden, int numberOfOutputs, float eta, int bias, bool roundouts);
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

#endif // !_NeuralNetwork_H
