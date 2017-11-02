#include <random>
#include <time.h>
#include "LNN.h"

int main() {
	srand(time(NULL));
	std::vector<std::vector<float>> y;

	//x = { { 0., 0., 0., 1. },{ 0., 0., 1., 1. },{ 0., 1., 0., 1. },{ 0., 1., 1., 1. },{ 1., 0., 0., 1. },{ 1., 0., 1., 1. },{ 1., 1., 0., 1. } };
	y = { { 0., 0., 1. },{ 0., 1., 0. },{ 0., 1., 1. },{ 1., 0., 0. },{ 1., 0., 1. },{ 1., 1., 0. },{ 1., 1., 1. }, {0., 0., 0.} };
	std::vector<int> a = { 3, 3, 3 };
	NN one = NN(a);
	one.train({ {1.2f, 0.5f, 0.8f}, {1.0f, 2.0f, 3.0f} }, 10000);
	system("pause");
	return 0;
}