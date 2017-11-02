#include "neuron.h"

neuron::neuron(int tlays, int outN) {
	net = 0;
	out = 0;
	w = (float(rand()) / float(0x7fff) - 0.5f) * 2.0f;
	deactivate = (float(rand()) / float(0x7fff) - 0.5f) * 2.0f;// float(rand() % 60) - 30.0;
	range = float(rand()) / float(0x7fff*1.0f); // (float(rand()) / float(0x7fff) - 0.5f) * 2.0f;
	activ = rand() % 5;
	tout = (rand() % 3 == 1) ? true : false;
	if ((tlays != 0) && !tout) {
		for (int i = 0; i <= tlays / 3 + 1; i++) {
			toLayer.push_back(rand() % (tlays)+1);
		}
		ifout = outN;
	}
	else if (tout) {
		ifout = rand() % outN;
	}

}

void neuron::activate() {
	switch (activ){
	default:
		out = (1.f / (1.f + exp(-net)) - 0.5f)*2.f;
		net = 0;
	}
}