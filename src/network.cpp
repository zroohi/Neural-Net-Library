#include "network.h"

NeuralNetwork::NeuralNetwork(std::vector<int> neuronsPerLayer,
                             std::function<double(double)> inputFunction,
                             std::function<double(std::vector<double>, std::vector<double>)> inputErrorFunction,
                             int inputEpochs,
                             double inputLearningRate)
                             :
                             activationFunction(inputFunction),
                             errorFunction(inputErrorFunction),
                             learningRate(inputLearningRate),
                             epochs(inputEpochs),
                             initialized(false)
{
    // Resize the layers vector so that it can hold all of the input layers, and resize the outputs vector to hold all calculations
    // Note that the +2 is to account for both an input and output layer
    layers.resize(neuronsPerLayer.size() + 2);
    outputs.resize(neuronsPerLayer.size() + 2);
    numLayers = layers.size();

    // Resize each of the individual layers
    for (int i = 0 ; i < neuronsPerLayer.size() ; i++)
    {
        layers[i + 1].resize(neuronsPerLayer[i]);
        outputs[i + 1].resize(neuronsPerLayer[i]);
    }

    // Get the function derivatives
    errorFunctionDerivative = LossFunctions::GetDerivativeFunctionName(errorFunction);
}

void NeuralNetwork::Initialize(std::vector<std::vector<double>> xDataInput, std::vector<std::vector<double>> yDataInput)
{
    if (!xDataInput.size() || !yDataInput.size())
    {
        throw(std::invalid_argument("Dataset must not be empty"));
        
    }
    xData = xDataInput;
    yData = yDataInput;
    numInputs = xData[0].size();
    numOutputs = yData[0].size();
    SetupInputLayer();
    SetupHiddenLayers();
    SetupOutputLayer();
    this->initialized = true;
}

void NeuralNetwork::SetupInputLayer()
{
    Neuron unweightedNeuron({1}, 0);
    std::vector<Neuron> inputLayer(numInputs, unweightedNeuron);
    layers[0] = inputLayer;
}

void NeuralNetwork::SetupHiddenLayers()
{
    int layerSize;
    double bias;

    for (int i = 1 ; i < layers.size() - 1 ; i++)
    {
        layerSize = layers[i].size();
        std::vector<Neuron> currentLayer(layerSize);
        std::vector<double> currentOutputs(layerSize, 0);
        layers[i] = currentLayer;
        outputs[i] = currentOutputs;

        for (int j = 0 ; j < layerSize ; j++)
        {
            std::vector<double> weights(layers[i - 1].size());
            for (auto& w : weights) { w =  GenerateRandomNumber(); }
            bias = GenerateRandomNumber();
            layers[i][j] = Neuron(weights, bias, activationFunction);
        }
    }
}

void NeuralNetwork::SetupOutputLayer()
{
    double bias;
    std::vector<Neuron> outputLayer(numOutputs);

    for (int i = 0 ; i < numOutputs ; i++)
    {
        std::vector<double> weights(layers[numLayers - 2].size());
        for (auto& w : weights) { w =  GenerateRandomNumber(); }
        bias = GenerateRandomNumber();
        outputLayer[i] = Neuron(weights, bias, activationFunction);
    }
    layers.back() = outputLayer;
    outputs.back() = std::vector<double>(numOutputs, 0);
}

double NeuralNetwork::GenerateRandomNumber()
{
    // Setup and seed the random number generator
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_int_distribution<int> distribution(-100, 100);
    return distribution(generator);
}

void NeuralNetwork::Forward(int currentIndex)
{
    // Set the inputs
    outputs[0] = xData[currentIndex];

    // Run through the neural network, calculating the output for each layer
    for (int i = 1 ; i < layers.size() ; i++)
    {
        for (int j = 0 ; j < layers[i].size() ; j++)
        {
            outputs[i][j] = layers[i][j].Forward(outputs[i - 1]);
        }
    }
}

void NeuralNetwork::BackPropogate(int currentIndex)
{
        // Update the weights and bias for the output neuron
        double dLdY, dYdW, dYdB;
        dLdY = errorFunctionDerivative(outputs.back(), yData[currentIndex]);
        for (Neuron& neuron : layers[numLayers - 1])
        {
            dYdB = neuron.Backward(outputs[numLayers - 2]);
            neuron.UpdateBias(learningRate * dLdY * dYdB);
            for (int i = 0 ; i < neuron.GetNumWeights() ; i++)
            {
                dYdW = outputs[numLayers - 2][i] * dYdB;
                neuron.UpdateOneWeight(learningRate * dLdY * dYdW, i);
            }
        }
        
        // Update the weights and biases for the hidden layer neurons
        double dYdH, dHdB, dHdW;
        for (int i = numLayers - 2 ; i > 0 ; i--)
        {
            for (int j = 0 ; j < layers[i].size() ; j++)
            {
                // Update this neuron's bias
                dHdB = layers[i][j].Backward(outputs[i - 1]);
                dYdH = layers[i + 1][0].GetWeights()[j] * layers[i + 1][0].Backward(outputs[i]);
                // @TODO - what if there's more than one neuron in the next layer? aka the "0" above, what should it be?
                //(weight from this neuron 0 to the next neuron 2) * Next Layers Neuron deriv_sigmoid(outputs[this layer])
                layers[i][j].UpdateBias(learningRate * dLdY * dYdH * dHdB);

                // Update this neuron's weights
                for (int k = 0 ; k < layers[i][j].GetNumWeights() ; k++)
                {
                    dHdW = outputs[i - 1][k] * layers[i][j].Backward(outputs[i - 1]);
                    layers[i][j].UpdateOneWeight(learningRate * dLdY * dYdH * dHdW, k);
                }
            }
        }
}

void NeuralNetwork::Run()
{
    if (!initialized)
    {
        throw(std::logic_error("Neural net is not initialized."));
    }

    double err;
    for (int epoch = 1 ; epoch <= epochs ; epoch++)
    {
        err = 0;
        for (int currentIndex = 0 ; currentIndex < xData.size() ; currentIndex++)
        {
            Forward(currentIndex);
            BackPropogate(currentIndex);
            err += errorFunction(outputs.back(), yData[currentIndex]);
        }
        std::cout << "Epoch " << epoch << " Loss : " << err / xData.size() << std::endl;
    }


}