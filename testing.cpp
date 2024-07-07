#include "src/network.h"
#include <iostream>

int main()
{
    // Create a neural network
    std::vector<int> layers = {2};
    int epochs = 1000000;
    double learningRate = 0.01;
    double cutoff = 0.00001;
    NeuralNetwork n(layers, ActivationFunctions::sigmoid, LossFunctions::mse, epochs, learningRate, cutoff);

    // Initialize it with our data
    std::vector<std::vector<double>> data_input = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> data_output = {{0}, {1}, {1}, {0}};

    n.Initialize(data_input, data_output);

    // Train it
    n.Train();

    return 0;
}

// TODO
// Save and open neural net weights
// Import an entire excel spreadsheet and run it through