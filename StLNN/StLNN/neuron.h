#pragma once
#ifndef _NEURON_H
#define _NEURON_H

#include <vector>
#include <random>
//#include "ops.h"

class neuron {
public:
	float net;
	float out;
	int activ; // represents the activation function to use in a switch statement
	int ifout;
	bool tout;

	// axon adjustment
	float w;
	// don't send data if "out" is within "range" of this value
	float deactivate;
	// deactivate-range < out < deactivate+range 
	float range;

	//std::vector<float> nextW; // update to weights
	//std::vector<float> dErr_dw; // partial derivative of Total Error with respect to w#
	std::vector<int> toLayer;

	// picks some layers to send to and sets itself up
	neuron(int totalLayers, int outNeurons);

	void activate();
};
#endif // !_NEURON_H
