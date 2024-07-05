#include "neuron.h"

Neuron::Neuron()
               :
               weights(nullWeights),
               bias(nullBias),
               f(ActivationFunctions::nullActivationFunction), 
               state(false)
{
}

Neuron::Neuron(std::function<float(float)> inputFunction)
               :
               weights(nullWeights),
               bias(nullBias),
               f(inputFunction), 
               state(false)
{
}

Neuron::Neuron(std::vector<float> inputWeights, float inputBias, std::function<float(float)> inputFunction)
               :
               weights(inputWeights),
               bias(inputBias),
               f(inputFunction),
               state(true)
{
    numInputs = weights.size();
}

float Neuron::forward(std::vector<float> inputs)
{
    // Verify the neuron is valid and has been initialized
    if (!state)
    {
        throw std::logic_error("Neuron is not initialized!");
    }

    // Verify the size of the input values is the same as the weights of this neuron
    if (inputs.size() != numInputs)
    {
        throw std::invalid_argument("Neuron input vector length must match weights vector length.");
    }

    // Calculate the dot product, add the bias, and call the activation function
    float prod = dotProduct(inputs, weights);
    float output = f(prod + bias);
    return output;
}

float Neuron::dotProduct(std::vector<float> left, std::vector<float> right)
{
    float sum {0};

    for (int i = 0 ; i < numInputs ; i++)
    {
        sum += left[i] * right[i];
    }

    return sum;
}

void Neuron::setWeights(std::vector<float> inputWeights)
{
    this->weights = inputWeights;
    this->numInputs = weights.size();
    isInitialized();
}

void Neuron::setBias(float inputBias)
{
    this->bias = inputBias;
    isInitialized();
}

void Neuron::setActivationFunction(std::function<float(float)> inputFunction)
{
    this->f = inputFunction;
    isInitialized();
}

bool Neuron::isInitialized()
{
    if (state)
    {
        return true;
    }
    else if (weights == nullWeights || bias == nullBias || *f.target<float(float)>() == ActivationFunctions::nullActivationFunction)
    {
        this->state = false;
    }
    else
    {
        this->state = true;
    }

    return state;

}