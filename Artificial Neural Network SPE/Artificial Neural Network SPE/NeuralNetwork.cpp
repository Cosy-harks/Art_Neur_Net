#include "NeuralNetwork.h"
#include "MathOps.h"


NeuralNetwork::NeuralNetwork():
	collection( std::vector< Layer >() ),
	target( std::vector< double >() ),
	learningRate( 0.25 ),
	last()
{
}


NeuralNetwork::NeuralNetwork( std::vector< int > v ):
	collection( std::vector< Layer >() ),
	target( std::vector< double >() ),
	learningRate( 0.05 ),
	last( v.size() - 1 )
{
	for (int i = 0; i < v.size(); i++)
	{
		collection.push_back(Layer(v[i]));
		if (i < v.size()-1)
		{
			collection[i].push_weights(v[i + 1]);
		}
	}
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::addLayer( unsigned int ns )
{
	collection.push_back( Layer( ns ) );
	last++;
}

void NeuralNetwork::test( std::vector< std::vector< double > > input, std::vector< std::vector< double > > expected )
{
	std::vector<double> err;
	for (int i = 0; i < expected.size(); i++)
	{
		target = expected[i];
		collection[0].input(input[i]);
		forwardPropagation();
		auto M = collection[last].make_matrices(false).first;
		err.push_back(sumSquaredDifference(M, target));
		std::cout <<vecToString(input[i]) << " --> " << vecToString(M) << " - " << vecToString(target) << std::endl;
	}
	std::cout << "%" << average(err) << std::endl;
}

void NeuralNetwork::train(std::vector<std::vector<double>> input, std::vector<std::vector<double>> expected)
{
	//may need to rework
	std::vector<double> err;
	for (int ii = 0; ii < 250; ii++)
	{
		for (int i = 0; i < expected.size(); i++)
		{
			target = expected[i];
			collection[0].input(input[i]);
			forwardPropagation();
			for (int L = 0; L < collection.size() - 1; L++)
			{
				for (int N = 0; N < collection[L].getSize(); N++)
				{
					for (int W = 0; W < collection[L + 1].getSize(); W++)
					{
						double d = backwardPropagation(L, N, W);
						//std::cout << d << std::endl;
						collection[L].setdWeight(N, W, collection[L].getdWeight(N, W) - d*learningRate);
					}
				}
			}
		}
		averageDeltaWeights(expected.size());
		// I did not update the weights
		updateWeights();
	}

}

void NeuralNetwork::print()
{
	for (int i = 0; i < collection.size; i++)
	{
		std::cout << collection[i].getSize();
	}
	std::cout << std::endl;
}

void NeuralNetwork::forwardPropagation()
{
	for ( int i = 0; i < collection.size(); i++ )
	{
		collection[ i ].function_of_input();
		//auto M = collection[i].make_matrices();
		//math M.first, M.second
		//auto vd = dot( collection[ i ].make_matrices() );
		if ( i + 1 < collection.size() )
		{
			collection[ i + 1 ].input(dot(collection[i].make_matrices(true)));
		}
		else
		{
			//output = dot(collection[i].make_matrices());
		}
	}
}

double NeuralNetwork::backwardPropagation(int L, int N, int W)
{
	// may need to rework
	//deriveSums.clear();
	double dErr_dLNW = 0.0;
	int dsindex = 0;

	if (L + 3 < collection.size())
	{
		for (int cwl = (L + 3 < last) ? L + 3 : last; cwl < collection.size()-1; cwl++)
		{
			for (int cn = 0; cn < collection[cwl].getSize(); cn++)
			{
				wideDerivative(cwl, cn, L, W, dErr_dLNW, dsindex);
			}
		}
		//for (int i = 0; i < output.size(); i++)
		//{
		//	for (int j = 0; j < collection[last].getSize(); j++)
		//	{
		//		dErr_dLNW += collection[last].dOut_dIn*collection[last].getWeight(j, i)*deriveSum[dsindex++];
		//	}
		//	dsindex -= output.size();
		//	deriveSum.push_back(dErr_dLNW);
		//	dErr_dLNW = 0.0;
		//	//dErr_dLNW += 2 * (output[i] - target[i]) * deriveSum[dsindex++];
		//}

		for (int i = 0; i < collection[last].getSize(); i++)
		{
			dErr_dLNW += (collection[last].getOutput(i) - target[i]) * deriveSums[dsindex++];
		}
		dErr_dLNW *= collection[L + 1].dOut_dIn(W)*collection[L].getOutput(N);
		deriveSums.push_back(dErr_dLNW);
	}
	return dErr_dLNW;
}

void NeuralNetwork::wideDerivative(int cwl, int cn, int L, int W, double & dErr_dLNW, int & dsindex)
{
	// May need to rework
	if (L + 3 == cwl)
	{
		for (int n = 0; n < collection[cwl - 1].getSize(); n++)
		{
			dErr_dLNW += collection[L + 2].getWeight(n, cn)*collection[L + 2].dOut_dIn(n)*collection[L + 1].getWeight(W, n);
		}
		deriveSums.push_back(dErr_dLNW);
		dErr_dLNW = 0.0;
		//dErr_dLNW *= collection[L + 1].dOut_dIn(W)*collection[L].getOutput(N);
	}
	else if (L + 2 == cwl)
	{
		for (int i = 0; i < collection[cwl].getSize(); i++)
		{
			deriveSums.push_back(1);
		}

	}
	else
	{
		for (int n = 0; n < collection[cwl - 1].getSize(); n++)
		{
			//dsindex is bad
			dErr_dLNW += collection[cwl - 1].getWeight(n, cn)*collection[cwl - 1].dOut_dIn(n)*deriveSums[dsindex];
			dsindex++;
		}
		if (cwl + 1 < collection.size())
		{
			dsindex -= (cn < collection[cwl + 1].getSize()) ? collection[cwl].getSize() - 1 : 0;
		}
		deriveSums.push_back(dErr_dLNW);
		dErr_dLNW = 0.0;
	}
}

void NeuralNetwork::averageDeltaWeights(int averager)
{
	for (int L = 0; L < collection.size()-1; L++)
	{
		for (int N = 0; N < collection[L].getSize(); N++)
		{
			for (int W = 0; W < collection[L + 1].getSize(); W++)
			{
				collection[L].setdWeight(N, W, collection[L].getdWeight(N, W) / (double)averager);
			}
		}
	}
}

void NeuralNetwork::updateWeights()
{
	for (int L = 0; L < collection.size()-1; L++)
	{
		for (int N = 0; N < collection[L].getSize(); N++)
		{
			for (int W = 0; W < collection[L + 1].getSize(); W++)
			{
				collection[L].setWeight(N, W, collection[L].getWeight(N, W) + collection[L].getdWeight(N, W));
				//std::cout << collection[L].getdWeight(N, W);
			}
			//std::cout << std::endl;
		}
	}
}
