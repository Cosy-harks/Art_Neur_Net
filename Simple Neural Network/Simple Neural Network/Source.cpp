#include "NeuralNetwork.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <time.h>

using namespace std;

vector<double> simple_activation(vector<vector<double>>, vector<double>);
vector<double> train(vector<vector<double>>, vector<double>, vector<double>, double, int);
vector<vector<vector<double>>> trains(vector<vector<double>> inputs, vector<double> targets, vector<vector<vector<double>>> weights, double eta, int n_iters);
vector<double> hid_activation(vector<double> in, vector<vector<double>> w);
vector<double> test(vector<vector<double>> data, vector<vector<vector<double>>> weights);
string vecToString(vector<float>);

int main() {
	srand(time(NULL));
	ifstream file("..\\2fears.csv");
	//FILE *file;
	//file = fopen("2fears.txt", "r");
	
	string a = "";
	// something is wrong here
	vector<vector<float>> x;
	vector<vector<float>> y = { vector<float>() };

	if (file) {
		while (file) {
			x.push_back(vector<float>());
			y.push_back(vector<float>());
			getline(file, a,',');
			for (int i = 0; i < 6; i++, getline(file, a, ',')) {
				x[x.size() - 1].push_back(::atof(a.c_str()));
				y[y.size() - 1].push_back(::atof(a.c_str()));
			}
		}
		y[0] = y[y.size() - 1];
		y.pop_back();
	}
	//fclose(file);


	vector<int> i = { 2, 1 };
	NN one(i, 3, 5, 3, .23f, 1, true);
	//x = { { 0., 0., 0., 1.f}, {0., 0., 1., 1.}, {0., 1., 1., 1.}, {1., 0., 0., 1.}, {1., 0., 1., 1.} };
	//y = { { 0., 0., 1.}, {0., 1., 0.}, {1., 0., 0.}, {1., 0., 1.}, {1., 1., 0.} };
	vector<vector<float>> c = { {0.05f, 0.1f, 1} };
	vector<vector<float>> b = { {0.01f, 0.99f} };
	x = { { 0., 0., 0., 1. }, { 0., 0., 1., 1. }, { 0., 1., 0., 1. }, { 0., 1., 1., 1. }, { 1., 0., 0., 1. }, { 1., 0., 1., 1. }, { 1., 1., 0., 1. } };
	y = { { 0., 0., 1. },     { 0., 1., 0. },     { 0., 1., 1. },     { 1., 0., 0. },     { 1., 0., 1. },     { 1., 1., 0. },     { 1., 1., 1. } };
	x = { { 0.f, 1.f }, { 1.f, 1.f } };
	y = { { 1.f }, { 0.f } };
	//one.print();
	
	one.trainNN(x, y, 100*25);

	
	//x = { { 0., 0., 0., 1. },{ 0., 0., 1., 1. },{ 0., 1., 0., 1. },{ 0., 1., 1., 1. },{ 1., 0., 0., 1. },{ 1., 0., 1., 1. },{1., 1., 0., 1.} };
	//y = { { 0., 0., 1. },{ 0., 1., 0. },{ 0., 1., 1. },{ 1., 0., 0. },{ 1., 0., 1. },{ 1., 1., 0. },{1., 1., 1.} };
	
	//one.print();

	for (int i = 0; i < x.size(); i++) {
		vector<float> a = one.testANN(x[i], y[i]);
		cout << vecToString(a) << " - " << vecToString(y[i]) << endl;
		cout << one.totalError() << endl;
		//cout << vecToString(elemWiseSubtraction(a, y[i])) << endl;
	}
	one.print();

	//vector<float> a = one.testANN(vector<float> {0.05f, 0.1f});
	//cout << vecToString(a);
	system("pause");
	return 0;
}

/*string vecToString(vector<float> toS){
	stringstream alf;
	alf << "{ ";
	for (int i = 0; i < toS.size() - 1; i++) {
		alf << toS[i] << ", ";
	}
	alf << toS[toS.size() - 1] << " }";
	return alf.str();
}
	/*cout << vecToString() << endl
	
vector<vector<vector<double>>> w;//{double(rand())/10000.0, double(rand())/10000.0, double(rand())/10000.0};
									 //cout << w[0] << ", " << w[1] << ", " << w[2] << endl;
	int cols = 3;// rand() % 13 + 2;
	int rows = 10;// rand() % 15 + 3;
	vector<vector<double>> Ane;

	for (int planes = 0; planes < 2; planes++) {
		w.push_back(vector<vector<double>> {});
		for (int vecs = 0; vecs < 2-planes; vecs++) {
			w[planes].push_back(vector<double> {});
		}
	}
	// Input to Hidden and Hidden to Output
	for (int plane = 0; plane < 2; plane++) {
		// # of hidden (only one layer)
		for (int i = 0; i < 2 && plane < 1; i++) {
			// # of inputs to the hidden
			for (int _ = 0; _ < cols; _++) {
				w[plane][i].push_back(rand() / 10000.0);
			}
		}
		for (int i = 0; i < 3 && plane > 0; i++) {
			w[plane][0].push_back(rand() / 10000.0);
		}
	}

	for (int i = 0; i < cols; i++) {
		Ane.push_back(vector<double>{});
		for (int j = 0; j < rows; j++) {
			Ane[i].push_back(rand() % 11 - 5);
		}
	}


	//w = { 1.124, -0.1872, 2.8117 };
	double adval = 0;
	for (int i = 0; i < Ane[0].size(); i++) {
		adval = 0;
		for (int j = 0; j < Ane.size() - 1; j++) {
			adval += Ane[j][i];
		}
		Ane[Ane.size() - 1][i] = ((adval > 2) ? 1 : 0);
	}
	// = { {1, 1, 0, 0}, {1, 0, 1, 0}, {1, 0, 0, 0} };

	vector<vector<double>> inputs;
	vector<vector<double>> f{ {} };
	for (int i = 0; i < Ane[0].size(); i++) {
		f[0].push_back(-1);
	}
	for (int i = 0; i < Ane.size() - 1; i++) {
		inputs.push_back(Ane[i]);
	}

	cout << vecToString(w[0][0]) << endl;
	vector<double> target = Ane[Ane.size() - 1];
	w = trains(inputs, target, w, 0.02, 45);

	inputs.insert(inputs.end(), f.begin(), f.end());

	cout << vecToString(target) << endl;
	cout << vecToString(w[0][0]) << endl;
	cout << vecToString(test(inputs, w)) << endl;
	//cout << vecToString(simple_activation(inputs, w[0][0])) << endl;

	system("pause");
	return 0;
}

vector<double> elemWiseSubtraction(vector<double> a, vector<double> b) {
	if (a.size() != b.size()) {
		cout << a[0] << " nope " << b[0] << endl;
		return a;
	}
	vector<double> c;
	for (int i = 0; i < a.size(); i++) {
		c.push_back(a[i] - b[i]);
	}
	return c;
}

//Transpose
vector<vector<double>> T(vector<vector<double>> vec) {
	//cout << vec[1][0] << endl;
	int col = vec[0].size() - 1;
	int row = vec.size() - 1;
	vector<vector<double>> last;
	for (int i = col; i >= 0; i--) {
		vector<double> t;
		for (int j = row; j >= 0; j--) {
			t.push_back(vec[j][i]);
		}
		last.push_back(t);
	}
	//cout << last[1][0] << endl;
	return last;
}

//matrix dot
vector<double> dot(vector<vector<double>> a, vector<double> b) {
	vector<double> c;
	for (int i = 0; i < a[0].size(); i++) {
		double val = 0;
		for (int j = 0; j < a.size(); j++) {
			val += a[j][i] * b[j];
		}
		c.push_back(val);
	}
	return c;
}
//dot
double dot(vector<double> a, vector<double> b) {
	
	double val = 0;
	for (int i = 0; i < a.size(); i++) {
		val += a[i] * b[i];
	}
	return val;
}

vector<double> scale(double s, vector<double> vec) {
	for (int i = 0; i < int(vec.size()); i++) {
		vec[i] *= s;
	}
	return vec;
}

vector<double> squash(vector<double> de_squashed) {
	for (int i = 0; i < de_squashed.size(); i++) {
		de_squashed[i] = 1 / (1 + exp(-de_squashed[i]));
	}
	return de_squashed;
}

vector<vector<vector<double>>> trains(vector<vector<double>> data, vector<double> targets, vector<vector<vector<double>>> weights, double eta, int n_iters) {
	vector<vector<double>> f = { {} };
	// Increases inputs by 1
	for (int i = 0; i < data[0].size(); i++) {
		f[0].push_back(-1);
	}
	data.insert(data.end(), f.begin(), f.end());

	// 
	for (int n = 0; n < n_iters; n++) {
		vector<double> hidden;
		vector<double> answers;
		for (int test = 0; test < data[0].size(); test++) {
			vector<double> input{};
			for (int j = 0; j < data.size(); j++) {
				input.push_back(data[j][test]);
			}
			hidden = hid_activation(input, weights[0]);
			for (int i = 1; i < weights.size(); i++) {
				hidden = hid_activation(hidden, weights[i]);
			}

			answers.push_back((hidden[0] > 0) ? 1 : 0);
		}
		answers = elemWiseSubtraction(answers, targets);
		vector<double> a = dot(T(data), answers);
		cout << answers.size() << endl;
		for (int i = 0; i < answers.size(); i++) {
			cout << answers[i];
		}
		cout << endl;
		cout << a.size() << endl;
		for (int i = 0; i < a.size(); i++) { cout << a[i]; }
		cout << endl;
		
		for (int wPlane = 0; wPlane < weights.size(); wPlane++) {
			for (int wVec = 0; wVec < weights[wPlane].size(); wVec++) {
				weights[wPlane][wVec] = elemWiseSubtraction(weights[wPlane][wVec], scale(-eta, a));
			}
		}
	}
	return weights;
}


vector<vector<double>> train(vector<vector<double>> inputs, vector<double> targets, vector<vector<double>> weights, double eta, int n_iters) {
	vector<vector<double>> f = { {} };
	for (int i = 0; i < inputs[0].size(); i++) {
		f[0].push_back(-1);
	}
	cout << inputs.size() << endl;
	inputs.insert(inputs.end(), f.begin(), f.end());
	cout << inputs.size() << endl;
	vector<vector<double>> activations = {};
	for (; n_iters > 0; n_iters--) {
		for (int outs = 0; outs < weights[0].size(); outs++) {
			//activations[outs] = simple_activation(inputs, weights[outs]);
			activations[outs] = elemWiseSubtraction(activations[outs], targets);
			weights[outs] = elemWiseSubtraction(weights[outs], scale(-eta, (dot(T(inputs), activations[outs]))));
		}
	}

	return weights;
}

vector<double> test(vector<vector<double>> data, vector<vector<vector<double>>> weights) {
	vector<double> answers;
	vector<double> hidden;
	for (int test = 0; test < data[0].size(); test++) {
		vector<double> input{};
		for (int j = 0; j < data.size(); j++) {
			input.push_back(data[j][test]);
		}
		hidden = hid_activation(input, weights[0]);
		for (int i = 1; i < weights.size(); i++) {
			hidden = hid_activation(hidden, weights[i]);
		}
		answers.push_back((hidden[0] > 0) ? 1 : 0);
	}
	return answers;
}

vector<double> hid_activation(vector<double> inputs, vector<vector<double>> weights) {
	vector<double> got;
	for (int i = 0; i < weights.size(); i++) {
		got.push_back(dot(inputs, weights[i]));
	}
	return got;
}
*/