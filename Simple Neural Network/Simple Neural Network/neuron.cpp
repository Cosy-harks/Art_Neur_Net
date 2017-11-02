
#include "neuron.h"


neuron::neuron(int numberOfWeights) {
	net = 0.0f;
	out = 0.0f;
	for (int wc = 0; wc < numberOfWeights; wc++) {
		w.push_back(0.0f);
		nextW.push_back(0.0f);
		dErr_dw.push_back(0.0f);
	}
	randWeights();
}

void neuron::activations() {
	switch (activ) {
	case 0:// 2/(1+e^-net)-1
		out_wrt_net = out*(1 - out);
		break;
	case 1://max(0, net)
		out_wrt_net = net > 0 ? 1 : 0;
		break;
	case 2://

		break;
	case 3:

		break;
	default:

		break;
	}
}

void neuron::randWeights() {
	for (int i = 0; i < w.size(); i++) {
		w[i] = (float(rand()) / float(0x7fff));
	}
}