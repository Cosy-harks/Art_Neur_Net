
#include "ANN.h"

/*this class would be held in [] for hidden
*Idea: make neuron class{
*	int func; // holds a number to differentiate the function it will use out of several
*}
*/

// numInputs is the number of inputs for one set of test data
// IHWeights is the number of weight values between input and hidden nodes
// numHidden is the number of hidden nodes (only 1 layer so far)
// numOutputs is the number of returned values
// *weights points to the weight being worked with
// *weightHead points to the first weight to get back to easily


// Idea: take in [int] first elem inputs, last elem out, mid are hidden
// input #of, hidden #of, output #of, round outputs(True/False)
NN::NN(std::vector<int> ls, float eta, int biase, bool roundouts)
{
	if (biase != 1 && biase != 0) {
		std::cout << "NO NET!" << std::endl;
	}

	else {
		std::cout << ls.size() << std::endl;
		Ls = ls.size() - 1;
		for (int i = 0; i < Ls; i++) {
			nlayer.push_back(layer(ls[i], ls[i + 1]));
		}
		nlayer.push_back(layer(ls[Ls], 0));

		for (int _ = 0; _ < ls[Ls]; _++) {
			target.push_back(0.0f);
		}

		lrnScalr = eta;
		round = roundouts;
		bias = biase;

		////////////////////finished//////////////////////

		// not all #'s are the same #
		// only for updates to weights going to output
		//net_h# = sum#(i_#*w_#)		vector<>
		//out_h# = 1/(1+e^(-net_h#))	vector<>
		//net_o# = sum#(out_h#*w_(#+s))	vector<>
		//out_o# = 1/(1+e^(-net_o#))	vector<>
		//Etotal = sum#(0.5(target_# - out_o#)^2)	float
		//E_out# = .5(target_# - out_o#)^2			vector<>
		//dEtotal/dw_# = dEtotal/dout_o# * dout_o#/dnet_o# * dnet_o#/dw_#	float
		//Etotal = sum#(0.5(target_# - out_o#)^2)
		//dEtotal/dout_o# = 2*0.5(target_# - out_o#)^(1-1)*(-1)				float
		//dEtotal/dout_o# = -(target_# - out_o#)
		//dout_o#/dnet_o# = out_o#(1-out_o#)		float
		//dnet_o#/dw_# = out_h#						float
		//dEtotal/dw_# = -(target_# - out_o#) * out_o#(1-out_o#) * out_h#
		//w_#b = w_# - eta*dEtotal/dw_#				vector<>

		// weights to hidden
		//dEtotal/dw_# = dEtotal/dout_h# * dout_h#/dnet_h# * dnet_h#/dw_#
		//dEtotal/dout_h# = sum_o#(dE_o#/dout_h#)
		//dE_o#/dout_h# = dE_o#/dnet_o# * dnet_o#/dout_h#
		//dE_o#/dnet_o# = dE_o#/dout_o# * dout_o#/dnet_o# = a # * a #
		//net_o# = sum#(w_(#+s)*out_h#)
		//dnet_o#/dout_h# = w_#
		//dE_o#/dout_h# = (a # * a #) * w_#
		//dEtotal/dout_h# = sum_o#(dE_o#/dout_h#) = a #
		//out_h# = 1/(1 + e^(-net_h#))
		//dout_h#/dnet_h# = out_h#(1-out_h#) = a #
		//net_h# = sum#(w_#*i_#)
		//dnet_h#/dw_# = i_#
		//dEtotal/dw_# = dEtotal/dout_h# * dout_h#/dnet_h# * dnet_h#/dw_#
		//dEtotal/dw_# = a #
		//w_#b = w_# - eta*dEtotal/dw_# = a #
		//
	}
}

// input training data vector<vector<float>> 1st elem = test #, 2nd = params 
bool NN::trainNN(std::vector<std::vector<float>> data, std::vector<std::vector<float>> Target, int n_iters)
{
	bool worked = true;

	for (int n = 0; n < n_iters; n++) {
		for (int test = 0; test < data.size(); test++) {
			nlayer[0].input(data[test]);
			forwardPropagation(0);
			target = Target[test];
			backPropagation();
			updateWeights();
		}
		std::cout << totalError() << std::endl;
		//lrnScalr *= 0.95f;
	}
	/*
	for (int n = 0; n < n_iters; n++) {
	//std::vector<std::vector<float>> answers;
	std::pair<std::vector<float>, std::vector<std::vector<float>>> o_w;
	for (int test = 0; test < data.size(); test++) {
	std::vector<float> d = data[test];
	d.push_back(bias);
	o_w = forwardProp(d, 0);
	backProp(o_w.first, Target[test], d, 2);
	//answers.push_back(o_w.first);
	//for (int elem = 0; elem < weights.size(); elem++) {
	//	weights[elem] = squash(weights[elem]);
	//}
	}*/
	///<summary>
	///Duel learning: backProp w/ 1 output & target ^,
	///then concatenate all the outputs and backProp on that v.
	///<summary>
	/*for (int e = 0; e < answers.size(); e++) {
	for (int f = 0; f < answers[0].size(); f++) {
	answers[e][f] = (answers[e][f] > 0);
	}
	}
	// Make separate BackPropagation func
	for (int i = 0; i < answers.size(); i++)
	{
	answers[i] = elemWiseSubtraction(target[i], answers[i]);
	}

	std::vector<std::vector<float>> help = dot(T(data), answers);
	for (int index = 0; index < help.size(); index++)
	{
	help[index] = scale(lrnScalr, help[index]);
	}
	std::vector<std::vector<float>> wets = T(o_w.second);
	for (int a = 0; a < help.size(); a++)
	{
	wets[a] = elemWiseSubtraction(wets[a], help[a]);
	}
	wets = T(wets);
	//std::cout << vecToString(wets[0]) << std::endl;
	for (int a = 0; a < out; a++)
	{
	OutputBetas = answers[a];
	for (int b = 0; b < hid; b++)
	{

	dweights[IHWeights + b + (a*hid)] = wets[a][b] - weights[IHWeights * 1 + b + (a*hid)];
	weights[IHWeights + b + (a*hid)] = wets[a][b];// *dweights[IHWeights * 1 + b + (a*hid)];
	}
	}
	// And learn how to do it.*/

	return false;
}

void NN::forwardPropagation(int i) {
	//ITS SO BEAUTIFUL
	for (int to = 0; i < Ls && to < nlayer[i + 1].node.size(); to++) {
		//std::cout << i << std::endl;
		nlayer[i + 1].node[to].net = 0;
		for (int from = 0; from < nlayer[i].node.size(); from++) {
			nlayer[i + 1].node[to].net += nlayer[i].node[from].out*nlayer[i].node[from].w[to];
			//std::cout << from << " " << to << " " << nlayer[i + 1].node[to].net << std::endl;
		}
		nlayer[i + 1].node[to].net += nlayer[i].bias.out*nlayer[i].bias.w[to];
		nlayer[i + 1].node[to].out = squash(nlayer[i + 1].node[to].net);
	}
	if (i < Ls - 1) {
		forwardPropagation(i + 1);
	}
	/*else {
	std::cout << "out0: " << nlayer[Ls].node[0].out << std::endl;
	std::cout << "out1: " << nlayer[Ls].node[1].out << std::endl;
	std::cout << "out2: " << nlayer[Ls].node[2].out << std::endl;
	}*/
}

void NN::backPropagation() {
	for (int fLayer = 0; fLayer < Ls; fLayer++) { //forward layer
		for (int from = 0; from < nlayer[fLayer].node.size(); from++) { // node in layer
			for (int to = 0; to < nlayer[fLayer].node[from].w.size(); to++) { // weight from node#
				// start recursion from here
				// with (LL, 0) recurse LL-1 till LL == fLayer-2
				// return the 2 multiplied derivitives + ( , ++)
				float e_w = 0;
				if (fLayer + 2 < Ls) {
					for (int i = 0; i < nlayer[Ls].node.size(); i++) {
						e_w = derivative(Ls, i, fLayer, from, to)*(target[i] - nlayer[Ls].node[i].out)*(nlayer[Ls].node[i].out*(1 - nlayer[Ls].node[i].out));
					}
					e_w *= ((from < nlayer[fLayer].node.size()) ? nlayer[fLayer].node[from].out : nlayer[fLayer].bias.out)*nlayer[fLayer + 1].node[to].out*(1 - nlayer[fLayer + 1].node[to].out);
				}
				else if (fLayer + 1 == Ls) {
					e_w = (-(target[to] - nlayer[Ls].node[to].out))*(nlayer[Ls].node[to].out*(1 - nlayer[Ls].node[to].out))*((from < nlayer[fLayer].node.size()) ? nlayer[fLayer].node[from].w[to] : nlayer[fLayer].bias.w[to]);//(nlayer[fLayer+1].node[to].out*(1 - nlayer[fLayer+1].node[to].out)*nlayer[fLayer].node[from].out);
				}
				else {// if fLayer + 2 == Ls
					for (int i = 0; i < nlayer[Ls].node.size(); i++) {
						e_w += (-(target[i] - nlayer[Ls].node[i].out)*nlayer[Ls].node[i].out*(1 - nlayer[Ls].node[i].out))*nlayer[Ls - 1].node[to].w[i];
					}
					e_w *= ((from < nlayer[fLayer].node.size()) ? nlayer[fLayer].node[from].out : nlayer[fLayer].bias.out)*nlayer[fLayer + 1].node[to].out*(1 - nlayer[fLayer + 1].node[to].out);
				}
				if (from < nlayer[fLayer].node.size()) {
					nlayer[fLayer].node[from].dErr_dw[to] = e_w;
					nlayer[fLayer].node[from].nextW[to] = nlayer[fLayer].node[from].w[to] - nlayer[fLayer].node[from].dErr_dw[to] * lrnScalr;
					//std::cout << e_w << std::endl;
					//std::cout << nlayer[fLayer].node[from].w[to] << std::endl;
				}
				else {
					nlayer[fLayer].bias.dErr_dw[to] = e_w;
					nlayer[fLayer].bias.nextW[to] = nlayer[fLayer].bias.w[to] - nlayer[fLayer].bias.dErr_dw[to] * lrnScalr;
				}
			}
		}
	}
}

float NN::derivative(int rl, int rn, int ll, int ln, int to) {
	// going all the way down may be able to update all the dErr_dw in a column.
	// would also need to multiply exterior partials to several ^ at a time.
	float dE_dw = 0.0f;
	if (rl == Ls && rl - 3 > ll) {
		for (int i = 0; i < nlayer[Ls - 1].node.size(); i++) {
			dE_dw += derivative(rl - 1, i, ll, ln, to)*(nlayer[Ls - 1].node[i].out*(1 - nlayer[Ls - 1].node[i].out))*nlayer[Ls - 1].node[i].w[rn];
		}
	}
	else if (rl - 3 > ll) {
		for (int i = 0; i < nlayer[rl - 1].node.size(); i++) {
			dE_dw += derivative(rl - 1, i, ll, ln, to)*(nlayer[rl - 1].node[i].out*(1 - nlayer[rl - 1].node[i].out))*nlayer[rl - 1].node[i].w[rn];
		}
	}
	else if (rl - 3 == ll) {
		for (int i = 0; i < nlayer[rl - 1].node.size(); i++) {
			dE_dw += nlayer[ll + 1].node[to].w[i] * nlayer[rl - 1].node[i].out*(1 - nlayer[rl - 1].node[i].out)*nlayer[rl - 1].node[i].w[rn];
		}
	}
	return dE_dw;
}

void NN::updateWeights() {
	for (int l = 0; l <= Ls; l++) {
		for (int n = 0; n < nlayer[l].node.size(); n++) {
			for (int to = 0; to < nlayer[l].node[n].w.size(); to++) {
				nlayer[l].node[n].w[to] = nlayer[l].node[n].nextW[to];
			}
		}
		for (int to = 0; to < nlayer[l].bias.w.size(); to++) {
			nlayer[l].bias.w[to] = nlayer[l].bias.nextW[to];
		}
	}
}

std::vector<float> NN::testANN(std::vector<float> test, std::vector<float> Target) {
	//test.push_back(bias);

	nlayer[0].input(test);
	target = Target;
	forwardPropagation(0);
	std::vector<float> returrn;
	for (int i = 0; i < nlayer[Ls].node.size(); i++) {
		returrn.push_back(nlayer[Ls].node[i].out);
	}
	return returrn;
}

// set Learning scalar
void NN::setLearn(float eta)
{
	lrnScalr = eta;
}

void NN::setBias(float b)
{
	bias = b;
}

float NN::totalError() {
	float err = 0.0f;
	for (int i = 0; i < nlayer[Ls].node.size(); i++) {
		err += 0.5f*pow(target[i] - nlayer[Ls].node[i].out, 2);
	}
	return err;
}

void NN::print() {
	//std::cout.precision(2);
	for (int l = 0; l <= Ls; l++) {
		for (int net = 0; net < nlayer[l].node.size(); net++) {
			std::cout << nlayer[l].node[net].net << "  ";
		}
		std::cout << std::endl;
		for (int out = 0; out < nlayer[l].node.size(); out++) {
			std::cout << nlayer[l].node[out].out << "  ";
		}
		std::cout << std::endl;
		for (int n = 0; n < nlayer[l].node.size(); n++) {
			for (int w = 0; w < nlayer[l].node[n].w.size(); w++) {
				std::cout << nlayer[l].node[n].w[w] << " ";
			}
			std::cout << std::endl;
		}
	}
}

