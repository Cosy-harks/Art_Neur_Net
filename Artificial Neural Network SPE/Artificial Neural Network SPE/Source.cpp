#include "MathOps.h"
#include "Node.h"
#include "Layer.h"
#include "NeuralNetwork.h"
#include <sstream>
#include <iostream>


int main()
{
	Node I = Node();
	auto L = Layer();
	std::vector<std::vector<double>> input = { { 0, 1, 1 },{ 0, 0, 0 },{ 0, 1, 0 },{ 1, 1, 0 },{ 0, 0, 1 } };
	std::vector<std::vector<double>> output = { { 1, 0, 0 },{ 0, 0, 1 },{ 0, 1, 1 },{ 1, 1, 1 },{ 0, 1, 0 } };
	std::vector<int> D = { (int)input[0].size(), 4, 3, 3, 4, (int)output[0].size() };
	NeuralNetwork N = NeuralNetwork(D);

	N.test(input, output);
	N.train(input, output);
	N.test(input, output);

	I.add(0.25);
	L.push_neuron(I);
	I.add(0.02);
	L.push_neuron(I);

	L.push_weights();
	L.push_weights();

	L.push_neuron(Node());
	L.push_weights();

	L.setFunction(1, 0);
	L.function_of_input();

	auto M = L.make_matrices(true);

	std::vector<double> vd;
	for (int k = 0; k < M.second[0].size(); k++)
	{
		vd.push_back(0.0);
		for (int j = 0; j < M.first.size(); j++)
		{
			vd[k] += M.first[j] * M.second[j][k];
		}
	}

	{
		std::string s = vecToString(M.first);
		std::cout << "1\n" << s << std::endl;
		for (int i = 0; i < M.second.size(); i++)
		{
			s = vecToString(M.second[i]);
			std::cout << s << std::endl;
		}
		s = vecToString(vd);
		std::cout << s << std::endl;
	}
	for (int no = 0; no < 4; no++)
	{
		std::cout << no + 2 << std::endl;
		L.input(vd);
		L.function_of_input();

		M = L.make_matrices(true);

		for (int k = 0; k < M.second[0].size(); k++)
		{
			for (int j = 0; j < M.first.size(); j++)
			{
				vd[k] += M.first[j] * M.second[j][k];
			}
		}

		{
			std::string s = vecToString(M.first);
			std::cout << s << std::endl;
			for (int i = 0; i < M.second.size(); i++)
			{
				s = vecToString(M.second[i]);
				std::cout << s << std::endl;
			}
			s = vecToString(vd);
			std::cout << s << std::endl;
		}
	}
	double d = I.getOutput();
	std::cout << d;
	I.setSwitch(2);
	I.mathFunctions();
	d = I.getOutput();
	std::cout << d << std::endl;
	system("pause");
	return 0;
}