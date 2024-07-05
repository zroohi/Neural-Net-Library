#include "src/network.h"
#include <iostream>

int main()
{
    // Create a neural network
    std::vector<int> layers = {2};
    NeuralNetwork n(layers, ActivationFunctions::sigmoid);

    // Initialize it with our data
    std::vector<float> inputs = {2, 3};
    int numOutputs = 1;
    n.initialize(inputs, numOutputs);

    // Run it once
    auto a = n.forward();
    for (auto b : a) { std::cout << b << std::endl; }
    
    return 0;
}