#pragma once
#ifndef _matrixOps_H
#define _matrixOps_H

#include <vector>
#include <sstream>
#include <iostream>

std::vector<float> squash(std::vector<float> de_squashed);

float squash(float de_squashed);

// Mat * Mat == Mat
std::vector<std::vector<float>> dot(std::vector<std::vector<float>> a, std::vector<std::vector<float>> b);

//matrix dot
std::vector<float> dot(std::vector<std::vector<float>> a, std::vector<float> b);

//scale elementwise
std::vector<float> scale(float s, std::vector<float> vec);

//Transpose
std::vector<std::vector<float>> T(std::vector<std::vector<float>> vec);

std::vector<float> elemWiseSubtraction(std::vector<float> a, std::vector<float> b);

std::vector<float> elemWiseAddition(std::vector<float> a, std::vector<float> b);

std::string vecToString(std::vector<float> toS);
#endif //!_matrixOps_H