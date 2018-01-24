#include "Node.h"



Node::Node() :
	inputSum( 0.0 ),
	output( 0.0 ),
	weight( std::vector<double>() ),
	deltaWeight( std::vector<double>() ),
	derivativeOutputToInput(0.0),
	switchValue( 0 )
{
}

Node::~Node()
{
}

void Node::add(double in)
{
	inputSum += in;
}

double Node::getOutput()
{
	return output;
}

void Node::setWeight( unsigned int index, double change )
{
	if ( weight.size() >= index ) { weight[ index ] = change; }
}

double Node::getWeight(unsigned int I)
{
	return weight[I];
}

void Node::setDeltaWeight(unsigned int a, double b)
{
	deltaWeight[a] = b;
}

double Node::getDeltaWeight(unsigned int a)
{
	return deltaWeight[a];
}

std::vector<double> Node::getWeights()
{
	return weight;
}

int Node::getWeightSize()
{
	return weight.size();
}

double Node::dOut_dIn()
{
	return derivativeOutputToInput;
}

void Node::pushWeight(double w)
{
	weight.push_back(w);
	deltaWeight.push_back(0);
}

void Node::pushWeights( int ws )
{
	for ( int i = 1; i < ws + 1; i++ )
	{
		pushWeight( i * 0.12 );
	}
}

void Node::mathFunctions()
{
	switch ( switchValue )
	{
	
	case 0:
		// If inputSum > 0 derivativeInputToOutput = 1
		// else derivativeInputToOutput = 0
		output = ( inputSum > 0 ) ? inputSum : 0;
		derivativeOutputToInput = ( inputSum > 0 ) ? 1 : 0;
		inputSum = 0;
		break;
	default:
		// 
		output = 1.0 / ( 1.0 + exp( -inputSum ) );	// do/di = 1/( 2 + e^-x + e^x );
													// do/di = output * ( 1 - output );
													// output = 1/( 1 + e^-x )
													// 1/( 1 + e^-x ) * ( 1 - 1/( 1 + e^-x ) == 1/( 2 + e^-x + e^x );
		derivativeOutputToInput = 1.0 / ( 2.0 + exp( -inputSum ) + exp( inputSum ) );
		inputSum = 0;
		break;
	}
}

void Node::setSwitch(int s)
{
	switchValue = s;
}
