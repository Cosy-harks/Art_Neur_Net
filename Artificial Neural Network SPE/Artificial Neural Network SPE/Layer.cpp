#include "Layer.h"

Layer::Layer():
	neurons(std::vector<Node>()),
	bias(Node())
{
	bias.add(1.0);
}

Layer::Layer(unsigned int ns) :
	neurons(std::vector<Node>()),
	bias(Node())
{
	for(;neurons.size() < ns; )
	{
		neurons.push_back(Node());
	}
	bias.add(1.0);
}


Layer::~Layer()
{
}

// Push a node into the layer
//and makes it have at least as many
//weights as the first neuron
void Layer::push_neuron(Node n)
{
	neurons.push_back(n);
	while (neurons[0].getWeightSize() > neurons[neurons.size()-1].getWeightSize())
	{
		neurons[neurons.size() - 1].pushWeight(0.2 * neurons[neurons.size() - 1].getWeightSize());
	}
}

// pushes one extra weight onto each neuron in the layer
void Layer::push_weights()
{
	for (int i = 0; i < neurons.size(); i++)
	{
		neurons[i].pushWeight((neurons[i].getWeightSize() + 0.5)*0.2);
	}
	bias.pushWeight(1.2 * bias.getWeightSize());
}

void Layer::push_weights(int ws)
{
	for (int i = 0; i < ws; i++)
	{
		push_weights();
	}
}

void Layer::function_of_input()
{
	for (int i = 0; i < neurons.size(); i++)
	{
		neurons[i].mathFunctions();
	}
	//may remove these eventually
	//will modify the weights of it not the output.
	bias.mathFunctions();
	bias.add(1.0);
}

void Layer::setFunction(int f, unsigned int index)
{
	if (index < getSize())
	{
		neurons[index].setSwitch(f);
	}
}

int Layer::getSize()
{
	return neurons.size();
}

void Layer::setWeight(int n, int w, double v)
{
	neurons[n].setWeight(w, v);
}

double Layer::getWeight(int n, int w)
{
	return neurons[n].getWeight(w);
}

void Layer::setdWeight(unsigned int a, unsigned int b, double c)
{
	neurons[a].setDeltaWeight(b, c);
}

double Layer::getdWeight(unsigned int a, unsigned int b)
{
	return neurons[a].getDeltaWeight(b);
}

double Layer::dOut_dIn(int n)
{
	return neurons[n].dOut_dIn();
}

double Layer::getOutput(int n)
{
	return neurons[n].getOutput();
}

std::pair<std::vector<double>, std::vector<std::vector<double>>> Layer::make_matrices(bool biass)
{
	// first 1 by nodes, second nodes by weights
	std::pair<std::vector<double>, std::vector<std::vector<double>>> _1xN_NxW;
	{
		Node a;
		for (int i = 0; i < neurons.size(); i++)
		{
			a = neurons[i];
			_1xN_NxW.first.push_back(a.getOutput());

			_1xN_NxW.second.push_back(a.getWeights());
		}
		if (biass)
		{
			a = bias;
			_1xN_NxW.first.push_back(a.getOutput());

			_1xN_NxW.second.push_back(a.getWeights());
		}
	}
	return _1xN_NxW;
}

void Layer::input(std::vector<double> in)
{
	if (in.size() == getSize())
	{
		for (int i = 0; i < in.size(); i++)
		{
			neurons[i].add(in[i]);
		}
	}
}
