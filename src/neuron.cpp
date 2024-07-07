#include "neuron.h"

Neuron::Neuron()
               :
               weights(nullWeights),
               bias(nullBias),
               state(false)
{
}

Neuron::Neuron(std::function<double(double)> inputFunction)
               :
               weights(nullWeights),
               bias(nullBias),
               f(inputFunction), 
               state(false)
{
    this->df = ActivationFunctions::GetDerivativeFunctionName(f);
}

Neuron::Neuron(std::vector<double> inputWeights, double inputBias, std::function<double(double)> inputFunction)
               :
               weights(inputWeights),
               bias(inputBias),
               f(inputFunction),
               state(true)
{
    this->numInputs = weights.size();
    this->df = ActivationFunctions::GetDerivativeFunctionName(f);
}

double Neuron::Forward(std::vector<Neuron> inputs)
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
    double prod = DotProduct(inputs, weights);
    this->lastOutput = GetActivationFunctionValue(prod + bias);
    return this->lastOutput;
}

double Neuron::Backward(std::vector<Neuron> inputs)
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
    double prod = DotProduct(inputs, weights);
    double output = GetActivationFunctionDerivativeValue(prod + bias);
    return output;
}

double Neuron::DotProduct(std::vector<Neuron> left, std::vector<double> right)
{
    double sum {0};

    for (int i = 0 ; i < numInputs ; i++)
    {
        sum += left[i].GetLastOutput() * right[i];
    }

    return sum;
}

void Neuron::SetWeights(std::vector<double> inputWeights)
{
    this->weights = inputWeights;
    this->numInputs = weights.size();
    IsInitialized();
}

void Neuron::UpdateOneWeight(double delta, int indexOfWeight)
{
    if (indexOfWeight >= weights.size())
    {
        throw(std::invalid_argument("Weight index is out of bounds"));
    }

    this->weights[indexOfWeight] -= delta;
}

std::vector<double> Neuron::GetWeights()
{
    return this->weights;
}

int Neuron::GetNumWeights()
{
    return this->numInputs;
}

double Neuron::GetBias()
{
    return this->bias;
}

void Neuron::SetBias(double inputBias)
{
    this->bias = inputBias;
    IsInitialized();
}

void Neuron::UpdateBias(double delta)
{
    this->bias -= delta;
}

void Neuron::SetActivationFunction(std::function<double(double)> inputFunction)
{
    this->f = inputFunction;
    this->df = ActivationFunctions::GetDerivativeFunctionName(f);
    IsInitialized();
}

double Neuron::GetActivationFunctionValue(double input)
{
    return this->f(input);
}

double Neuron::GetActivationFunctionDerivativeValue(double input)
{
    return this->df(input);
}  

bool Neuron::IsInitialized()
{
    if (state)
    {
        return true;
    }
    else if (weights == nullWeights || bias == nullBias || !f)
    {
        this->state = false;
    }
    else
    {
        this->state = true;
    }

    return state;

}

double Neuron::GetLastOutput()
{
    return this->lastOutput;
}

void Neuron::SetOutput(double input)
{
    this->lastOutput = input;
}