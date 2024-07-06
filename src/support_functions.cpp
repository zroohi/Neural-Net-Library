#include "support_functions.h"

float ActivationFunctions::binary(float x)
{
    return (x < 0) ? 0 : 1;
}

float ActivationFunctions::d_binary(float x)
{
    return 0;
}

float ActivationFunctions::linear(float x)
{
    return x;
}

float ActivationFunctions::d_linear(float x)
{
    return 1;
}

float ActivationFunctions::sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float ActivationFunctions::d_sigmoid(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

float ActivationFunctions::tanh(float x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float ActivationFunctions::d_tanh(float x)
{
    return 1 - tanh(x) * tanh(x);
}

float ActivationFunctions::relu(float x)
{
    return (x > 0) ? x : 0;
}

float ActivationFunctions::d_relu(float x)
{
    return (x > 0) ? 1 : 0;
}

float ActivationFunctions::lrelu(float x)
{
    return (x > 0) ? x : 0.1 * x;
}

float ActivationFunctions::d_lrelu(float x)
{
    return (x > 0) ? 1 : 0.1;
}

float ActivationFunctions::elu(float x)
{
    return (x > 0) ? x : exp(-x) - 1;
}

float ActivationFunctions::d_elu(float x)
{
    return (x > 0) ? x : - exp(-x);
}

std::function<float(float)> ActivationFunctions::getDerivativeFunctionName(std::function<float(float)> f)
{
    auto functionName = *f.target<float(float)>();
    std::function<float(float)> df;
    
    if      (functionName == ActivationFunctions::binary)  { df = ActivationFunctions::d_binary; }
    else if (functionName == ActivationFunctions::linear)  { df = ActivationFunctions::d_linear; }
    else if (functionName == ActivationFunctions::sigmoid) { df = ActivationFunctions::d_sigmoid; }
    else if (functionName == ActivationFunctions::tanh)    { df = ActivationFunctions::d_tanh; }
    else if (functionName == ActivationFunctions::relu)    { df = ActivationFunctions::d_relu; }
    else if (functionName == ActivationFunctions::lrelu)   { df = ActivationFunctions::d_lrelu; }
    else if (functionName == ActivationFunctions::elu)     { df = ActivationFunctions::d_elu; }
    
    return df;
}


float LossFunctions::mse(std::vector<float> predicted, std::vector<float> actual)
{
    if (predicted.size() != actual.size())
    {
        throw(std::invalid_argument("Size of the predicted and true value vectors must be the same."));
    }

    // Find the sum of the squared difference
    float sum = 0;
    for (int i = 0 ; i < actual.size() ; i++)
    {
        sum += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
    }

    // Find the mean
    sum = sum / actual.size();
    
    return sum;
}

float LossFunctions::d_mse(float actual, float predicted)
{
    return -2 * (actual - predicted);
}

float LossFunctions::mae(std::vector<float> predicted, std::vector<float> actual)
{
    if (predicted.size() != actual.size())
    {
        throw(std::invalid_argument("Size of the predicted and true value vectors must be the same."));
    }

     // Find the sum of the squared difference
    float sum = 0;
    for (int i = 0 ; i < actual.size() ; i++)
    {
        sum += std::abs((actual[i] - predicted[i]));
    }

    // Find the mean
    sum = sum / actual.size();
    
    return sum;

}

float LossFunctions::d_mae(float actual, float predicted)
{
    return 0;
}

std::function<float(float, float)> LossFunctions::getDerivativeFunctionName(std::function<float(std::vector<float>, std::vector<float>)> f)
{
    auto functionName = *f.target<float(std::vector<float>, std::vector<float>)>();
    std::function<float(float, float)> df;
    
    if      (functionName == LossFunctions::mse)  { df = LossFunctions::d_mse; }
    else if (functionName == LossFunctions::mae)  { df = LossFunctions::d_mae; }
    
    return df;
}