#pragma once
#ifndef _LAYER_H
#define _LAYER_H

#include "neuron.h"

class layer {
public:
	std::vector<neuron> node;
	//neuron bias = neuron(1);//bias
	//int weightCount;

	layer(int neurons, int totalLayers, int outN) {
		//weightCount = neurons*nextNeurons;
		for (int i = 0; i < neurons; i++) {
			node.push_back(neuron(totalLayers, outN));
		}
		//bias = neuron(totalLayers);
		//setBias(1);
	}

	void input(std::vector<float> input) {
		for (int i = 0; i < input.size(); i++) {
			node[i].net = input[i];
		}
	}
};
#endif // !_LAYER_H
