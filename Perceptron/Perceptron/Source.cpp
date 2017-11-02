#include <iostream>
#include <sstream>
#include <random>
#include <vector>
#include <time.h>

using namespace std;

vector<double> simple_activation(vector<vector<double>>, vector<double>);
vector<double> train(vector<vector<double>>, vector<double>, vector<double>, double, int);
string vecToString(vector<double>);

int main() {
	srand(time(NULL));
	vector<double> w;//{double(rand())/10000.0, double(rand())/10000.0, double(rand())/10000.0};
	//cout << w[0] << ", " << w[1] << ", " << w[2] << endl;
	int cols = 2;// rand() % 13 + 2;
	int rows = 10;// rand() % 15 + 3;
	vector<vector<double>> Ane;
	for (int i = 0; i < cols; i++) {
		w.push_back(double(rand() / 10000.0));
		Ane.push_back(vector<double>{});
		for (int j = 0; j < rows; j++) {
			Ane[i].push_back(rand() % 11-5);
		}
	}
	//w = { 1.124, -0.1872, 2.8117 };
	int adval = 0;
	for (int i = 0; i < Ane[0].size(); i++) {
		adval = 0;
		for (int j = 0; j < Ane.size() - 1; j++) {
			adval += Ane[j][i];
		}
		Ane[Ane.size() - 1][i] = ((abs(adval) == 2) ? 1 : 0);
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

	vector<double> target = Ane[Ane.size() - 1];
	w = train(inputs, target, w, 0.09, 45);

	inputs.insert(inputs.end(), f.begin(), f.end());

	cout << vecToString(target) << endl;
	cout << vecToString(w) << endl;
	cout << vecToString(simple_activation(inputs, w)) << endl;

	system("pause");
	return 0;
}

string vecToString(vector<double> toS) {
	stringstream alf;
	alf << "{ ";
	for (int i = 0; i < toS.size() - 1; i++) {
		alf << toS[i] << ", ";
	}
	alf << toS[toS.size() - 1] << " }";
	return alf.str();
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

//Transpons
vector<vector<double>> T(vector<vector<double>> vec) {
	//cout << vec[1][0] << endl;
	int col = vec[0].size()-1;
	int row = vec.size()-1;
	vector<vector<double>> last;
	for (int i = col; i >= 0; i--) {
		vector<double> t;
		for (int j = row; j >= 0; j--){
			t.push_back(vec[j][i]);
		}
		last.push_back(t);
	}
	//cout << last[1][0] << endl;
	return last;
}

//dot
vector<double> dot(vector<vector<double>> a, vector<vector<double>> b) {
	vector<double> c;
	for (int i = 0; i < a[0].size(); i++) {
		double val = 0;
		for (int j = 0; j < a.size(); j++) {
			val += a[j][i] * b[0][j];
		}
		c.push_back(val);
	}
	return c;
}

vector<double> scale(double s, vector<double> vec) {
	for (int i = 0; i < int(vec.size()); i++) {
		vec[i] *= s;
	}
	return vec;
}

vector<double> train(vector<vector<double>> inputs, vector<double> targets, vector<double> weights, double eta, int n_iters) {
	vector<vector<double>> f = { {} };
	for (int i = 0; i < inputs[0].size(); i++) {
		f[0].push_back(-1);
	}
	cout << inputs.size() << endl;
	inputs.insert(inputs.end(), f.begin(), f.end());
	cout << inputs.size() << endl;
	for (; n_iters > 0; n_iters--) {
		vector<double> activations = simple_activation(inputs, weights);
		activations = elemWiseSubtraction(activations, targets);
		weights = elemWiseSubtraction(weights, scale(-eta, (dot(T(inputs), vector<vector<double>> {activations}))));
	}

	return weights;
}

vector<double> simple_activation(vector<vector<double>> inputs, vector<double> weights) {
	//return np.where(np.dot(inputs, weights)>0, 1, 0)
	vector<double> got = dot(inputs, vector<vector<double>>{weights});
	for (int i = 0; i < got.size(); i++) {
		got[i] = (got[i] > 0) ? 1 : 0;
	}
	return got;
}