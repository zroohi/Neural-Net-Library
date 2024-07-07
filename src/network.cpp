#include "network.h"

NeuralNetwork::NeuralNetwork(std::vector<int> neuronsPerLayer,
                             std::function<double(double)> inputFunction,
                             std::function<double(std::vector<double>, std::vector<double>)> inputErrorFunction,
                             int inputEpochs,
                             double inputLearningRate,
                             double inputCutoff)
                             :
                             activationFunction(inputFunction),
                             errorFunction(inputErrorFunction),
                             learningRate(inputLearningRate),
                             epochs(inputEpochs),
                             cutoff(inputCutoff),
                             initialized(false)
{
    // Resize the layers vector so that it can hold all of the input layers
    // Note that the +2 is to account for both an input and output layer
    layers.resize(neuronsPerLayer.size() + 2);
    numLayers = layers.size();

    // Resize each of the individual layers
    for (int i = 0 ; i < neuronsPerLayer.size() ; i++)
    {
        layers[i + 1].resize(neuronsPerLayer[i]);
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
        layers[i] = currentLayer;

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
}

double NeuralNetwork::GenerateRandomNumber()
{
    // Setup and seed the random number generator
    std::random_device seed;
    std::mt19937 generator(seed());
    std::uniform_int_distribution<int> distribution(-5, 5);
    return distribution(generator);
}

void NeuralNetwork::Forward(int currentIndex)
{
    // Set the inputs
    for (int i = 0 ; i < layers[0].size() ; i++)
    {
        layers[0][i].SetOutput(yData[currentIndex][i]);
    }

    // Run through the neural network, calculating the output for each layer
    for (int i = 1 ; i < layers.size() ; i++)
    {
        for (int j = 0 ; j < layers[i].size() ; j++)
        {
            layers[i][j].Forward(layers[i - 1]);
        }
    }
}

void NeuralNetwork::BackPropogate(int currentIndex)
{
    // Update the weights and bias for all layers
    double dEdO, dOdN, dNdW, outputErrors;
    for (int i = numLayers - 1 ; i > 0 ; i--)
    {
        for (int j = 0 ; j < layers[i].size() ; j++)
        {

            // Calculate the output derivative with respect to the total net input (the derivative of the activation function)
            dOdN = layers[i][j].Backward(layers[i - 1]);

            // If this is an output neuron, calculate the loss derivative directly, otherwise use the sum of output neuron's loss
            if (i == numLayers - 1)
            {
                dEdO = errorFunctionDerivative(layers[i][j].GetLastOutput(), yData[currentIndex][j]);
                outputErrors += dEdO * dOdN;
            }
            else
            {
                dEdO = outputErrors * layers[i][j].Backward(layers[i - 1]);
            }
            
            // Update the bias
            layers[i][j].UpdateBias(learningRate * dEdO * dOdN);

            // Calculate the weight derivative for each weight and then update that specific weight
            for (int k = 0 ; k < layers[i][j].GetNumWeights() ; k++)
            {  
                dNdW = layers[i - 1][k].GetLastOutput();
                layers[i][j].UpdateOneWeight(learningRate * dEdO * dOdN * dNdW, k);
            }
        }
    }

    // Calculate the final mean loss
    epochErr = outputErrors / yData[currentIndex].size();
}

void NeuralNetwork::Train()
{
    if (!initialized)
    {
        throw(std::logic_error("Neural net is not initialized."));
    }

    for (int epoch = 1 ; epoch <= epochs ; epoch++)
    {
        for (int currentIndex = 0 ; currentIndex < xData.size() ; currentIndex++)
        {
            Forward(currentIndex);
            BackPropogate(currentIndex);
        }

        std::cout << "Epoch " << epoch << " Loss: " << epochErr << std::endl;

        if (epochErr <= cutoff)
        {
            std::cout << "Loss below cutoff level. Exiting early at epoch " << epoch << " with loss " << epochErr << "" << std::endl;
            break;
        }
    }
}