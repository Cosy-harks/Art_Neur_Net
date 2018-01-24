#pragma once
#include <vector>

class Node
{
public:
	Node();
	~Node();

	void add(double);
	double getOutput();
	void setWeight(unsigned int, double);
	double getWeight(unsigned int);
	void setDeltaWeight(unsigned int, double);
	double getDeltaWeight(unsigned int);
	// good for creating matrix
	std::vector<double> getWeights();

	int getWeightSize();
	double dOut_dIn();
	void mathFunctions();
	void setSwitch(int);

	void pushWeight(double);
	void pushWeights(int);

private:
	double inputSum;
	double output;

	std::vector<double> weight;
	std::vector<double> deltaWeight;
	
	double derivativeOutputToInput;

	int switchValue;
	double derivativeValue;

};

