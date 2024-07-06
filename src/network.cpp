#include "network.h"


NeuralNetwork::NeuralNetwork(std::vector<int> neuronsPerLayer,
                             std::function<float(float)> inputFunction,
                             std::function<float(std::vector<float>, std::vector<float>)> inputErrorFunction)
                             :
                             activationFunction(inputFunction),
                             errorFunction(inputErrorFunction),
                             initialized(false)
{
    // Resize the layers vector so that it can hold all of the input layers, and resize the outputs vector to hold all calculations
    // Note that the +2 is to account for both an input and output layer
    layers.resize(neuronsPerLayer.size() + 2);
    outputs.resize(neuronsPerLayer.size() + 2);

    // Resize each of the individual layers
    for (int i = 0 ; i < neuronsPerLayer.size() ; i++)
    {
        layers[i + 1].resize(neuronsPerLayer[i]);
        outputs[i + 1].resize(neuronsPerLayer[i]);
    }

    // Get the error function derivative
    errorFunctionDerivative = LossFunctions::getDerivativeFunctionName(errorFunction);
}

void NeuralNetwork::initialize(std::vector<float> inputs, int numOutputs)
{
    initialize(inputs, numOutputs, activationFunction);
}

void NeuralNetwork::initialize(std::vector<float> inputs, int numOutputs, std::function<float(float)> outputFunction)
{
    SetupInputLayer(inputs);
    SetupHiddenLayers();
    SetupOutputLayer(numOutputs, outputFunction);
    this->initialized = true;
}

void NeuralNetwork::SetupInputLayer(std::vector<float> inputs)
{
    Neuron unweightedNeuron({1}, 0);
    std::vector<Neuron> inputLayer(inputs.size(), unweightedNeuron);
    layers[0] = inputLayer;
    outputs[0] = inputs;
}

void NeuralNetwork::SetupHiddenLayers()
{
    std::vector<float> weights;
    int layerSize;
    float bias;

    for (int i = 1 ; i < layers.size() - 1 ; i++)
    {
        layerSize = layers[i].size();
        std::vector<Neuron> currentLayer(layerSize);
        std::vector<float> currentOutputs(layerSize, 0);
        layers[i] = currentLayer;
        outputs[i] = currentOutputs;

        for (int j = 0 ; j < layerSize ; j++)
        {
            // @TODO - Weights and bias should be generated randomly for every neuron with a size equal to the previous number of inputs
            weights = {1, 1};
            bias = 0;
            layers[i][j] = Neuron(weights, bias, activationFunction);
        }
    }
}

void NeuralNetwork::SetupOutputLayer(int numOutputs, std::function<float(float)> outputFunction)
{
    std::vector<float> weights;
    float bias;
    std::vector<Neuron> outputLayer(numOutputs);

    for (int i = 0 ; i < numOutputs ; i++)
    {
        // @TODO - Weights and bias should be generated randomly for every neuron with a size equal to the previous number of inputs
        weights = {1, 1};
        bias = 0;
        outputLayer[i] = Neuron(weights, bias, outputFunction);
    }
    layers.back() = outputLayer;
    outputs.back() = std::vector<float>(numOutputs, 0);
}

std::vector<float> NeuralNetwork::forward()
{
    if (!initialized)
    {
        throw(std::logic_error("Neural net is not initialized."));
    }

    // Run through the neural network, calculating the output for each layer
    for (int i = 1 ; i < layers.size() ; i++)
    {
        for (int j = 0 ; j < layers[i].size() ; j++)
        {
            outputs[i][j] = layers[i][j].forward(outputs[i - 1]);
        }
    }

    return outputs.back();
}

void NeuralNetwork::backPropogate()
{
    float dLdW, dLdY, dYdH, dHdW;

    // Calculate the loss derivative]
    float correct = 1; //@TODO need to be using the correct value here
    dLdY = errorFunctionDerivative(outputs.back()[0], correct); // @TODO - need to paramaterize this for multiple outputs

    
    // dy_pred / dh_x = the next neuron's weight for the value coming from the current neuron, multiplied by f'(the inputs and weights to this next neuron)
    // dh_x / dw_x    = the input x_i times the derivative of the activation function, fed the inputs dotproduct + bias i.e. x_i * f'(x_i * w_i + ... + b)

    dLdW = dLdY * dYdH * dHdW;
    
}