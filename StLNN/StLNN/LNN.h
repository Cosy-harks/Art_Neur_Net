#pragma once
#ifndef _LNN_H
#define _LNN_H



#include "layer.h"

class NN {
private:
	int	  Ls;
	std::vector<layer> nlayer;
	layer forwardpropagation();
	//void update();

public:
	NN(std::vector<int> layernodes);
	void train(std::vector<std::vector<float>> data, int n);
};

#endif // !_LNN_H