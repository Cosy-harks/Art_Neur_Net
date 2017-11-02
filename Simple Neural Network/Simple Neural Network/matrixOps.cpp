#include "matrixOps.h"

std::string vecToString(std::vector<float> toS) {
	std::stringstream alf;
	alf << "{ ";
	for (int i = 0; i < toS.size() - 1; i++) {
		alf << toS[i] << ", ";
	}
	alf << toS[toS.size() - 1] << " }";
	return alf.str();
}

// sigmoid
std::vector<float> squash(std::vector<float> de_squashed) {
	for (int i = 0; i < de_squashed.size(); i++) {
		de_squashed[i] = squash(de_squashed[i]);
	}
	return de_squashed;
}

// sigmoid
float squash(float de_squashed) {
	return 2.0f / (1.0f + exp(-de_squashed)) - 1.0f;
}

// Mat * Mat
std::vector<std::vector<float>> dot(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b)
{
	std::vector<std::vector<float>> c;
	for (int i = 0; i < a.size(); i++) {
		std::vector<float> val;
		for (int k = 0; k < b[0].size(); k++) {
			val.push_back(.0f);
			for (int j = 0; j < a[0].size(); j++) {
				val[k] += a[i][j] * b[j][k];
			}
		}
		c.push_back(val);
	}
	return c;
}

// Mat * vec
std::vector<float> dot(std::vector<std::vector<float>> a, std::vector<float> b)
{
	std::vector<float> c;
	for (int i = 0; i < a.size(); i++) {
		float val = 0;
		for (int j = 0; j < a[0].size(); j++) {
			val += a[i][j] * b[j];
		}
		c.push_back(val);
	}
	return c;
}

std::vector<float> scale(float s, std::vector<float> vec) {
	for (int i = 0; i < int(vec.size()); i++) {
		vec[i] *= s;
	}
	return vec;
}

//Transpose
std::vector<std::vector<float>> T(std::vector<std::vector<float>> vec) {
	//cout << vec[1][0] << endl;
	int col = vec[0].size() - 1;
	int row = vec.size() - 1;
	std::vector<std::vector<float>> last;
	for (int i = col; i >= 0; i--) {
		std::vector<float> t;
		for (int j = row; j >= 0; j--) {
			t.push_back(vec[j][i]);
		}
		last.push_back(t);
	}
	//cout << last[1][0] << endl;
	return last;
}

std::vector<float> elemWiseSubtraction(std::vector<float> a, std::vector<float> b) {
	if (a.size() != b.size()) {
		std::cout << a.size() << " nope " << b.size() << std::endl;
		return a;
	}
	std::vector<float> c;
	for (int i = 0; i < a.size(); i++) {
		c.push_back(a[i] - b[i]);
	}
	return c;
}

std::vector<float> elemWiseAddition(std::vector<float> a, std::vector<float> b) {
	if (a.size() != b.size()) {
		std::cout << a.size() << " nope " << b.size() << std::endl;
		return a;
	}
	std::vector<float> c;
	for (int i = 0; i < a.size(); i++) {
		c.push_back(a[i] + b[i]);
	}
	return c;
}
