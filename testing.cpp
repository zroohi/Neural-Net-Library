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
    std::vector<std::vector<double>> data_input = {{-2, -1}, {25, 6}, {17, 4}, {-15, -6}};
    std::vector<std::vector<double>> data_output = {{1}, {0}, {0}, {1}};

    int numOutputs = 1;
    n.Initialize(data_input, data_output);

    // Run it once
    n.Run();

    return 0;
}

// TODO
// Figure out how to correctly deal with the single output layer weight updating, it should be for multiple outputs etc.
// Save and open neural net weights
// Import an entire excel spreadsheet and run it through