#pragma once
#ifndef _NEURON_H
#define _NEURON_H

#include <vector>
#include <random>
#include "matrixOps.h"

class neuron {
public:
	float net;
	float out;
	int activ;
	float out_wrt_net;
	//int activ; // represents the activation function to use in a switch statement
	std::vector<float> w; // connections to next layer
	std::vector<float> nextW; // update to weights
	std::vector<float> dErr_dw; // partial derivative of Total Error with respect to w#
	

	neuron(int numberOfWeights);

	void activations();

	void randWeights();
};
#endif // !_NEURON_H
