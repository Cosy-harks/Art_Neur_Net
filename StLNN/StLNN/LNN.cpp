#include "LNN.h"
#include <iostream>

// error all out puts are the same
// I need a way to send data to only one output,
// if sending to the last layer, consistently

NN::NN(std::vector<int> layernodes) {
	Ls = layernodes.size() - 1;

	for (int i = 0; i < Ls; i++) {
		nlayer.push_back(layer(layernodes[i], Ls, layernodes[Ls]));
	}
	nlayer.push_back(layer(layernodes[Ls], 0, layernodes[Ls]));
}

layer NN::forwardpropagation() {
	// for each layer
	for (int i = 0; i <= Ls; i++) {
		// for each node in layer[i]
		for (int j = 0; j < nlayer[i].node.size(); j++) {
			// if layer is sending places
			if (nlayer[i].node[j].toLayer.size()) {
				// if layer is in do not sent range
				if (nlayer[i].node[j].deactivate - nlayer[i].node[j].range > nlayer[i].node[j].out || nlayer[i].node[j].out > nlayer[i].node[j].deactivate + nlayer[i].node[j].range) {
					// call function in neuron to do each uniqueish activation
					nlayer[i].node[j].activate();
					// for each layer being sent to
					for (int k = 0; k < nlayer[i].node[j].toLayer.size(); k++) {
						float s = nlayer[i].node[j].out * nlayer[i].node[j].w;
						// for node in receiving layer
						for (int l = 0; l < nlayer[nlayer[i].node[j].toLayer[k]].node.size(); l++) {
							//////////////////////////////////////////////////////////////////////////////
							nlayer[nlayer[i].node[j].toLayer[k]].node[l].net += s;
						}
					}
					nlayer[i].node[j].out = 0;
				}
			}
			else if (nlayer[i].node[j].tout) {
				// call function in neuron to do each uniqueish activation
				nlayer[i].node[j].activate();
				float s = nlayer[i].node[j].out * nlayer[i].node[j].w;
				nlayer[Ls].node[nlayer[i].node[j].ifout].net += s;
			}
			else {
				nlayer[i].node[j].activate();
			}
		}
	}
	return nlayer[Ls];
}

void NN::train(std::vector<std::vector<float>> data, int n_iters) {
	layer out = layer(nlayer[Ls].node.size(), Ls, nlayer[Ls].node.size());
	for (int _ = 0; _ < n_iters; _++) {
		for (int i = 0; i < data.size(); i++) {
			nlayer[0].input(data[i]);
			out = forwardpropagation();
			for (int j = 0; j < out.node.size(); j++) {
				std::cout << out.node[j].out;
			}
			std::cout << std::endl;
		}
	}
}