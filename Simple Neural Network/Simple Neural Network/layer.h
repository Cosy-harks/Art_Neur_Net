#pragma once
#ifndef _LAYER_H
#define _LAYER_H

#include "neuron.h"

class layer {
public:
	std::vector<neuron> node;
	neuron bias = neuron(1);//bias
	int weightCount;

	layer(int neurons, int nextNeurons) {
		weightCount = neurons*nextNeurons;
		for (int i = 0; i < neurons; i++) {
			node.push_back(neuron(nextNeurons));
		}
		bias = neuron(nextNeurons); // bias
		setBias(1);
	}

	void input(std::vector<float> input) {
		for (int i = 0; i < input.size(); i++) {
			node[i].out = input[i];
		}
	}

	void setBias(int bi) {// bias
		bias.out = bi;
	}
};
#endif // !_LAYER_H
