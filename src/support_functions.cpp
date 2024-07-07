#include "support_functions.h"

double ActivationFunctions::binary(double x)
{
    return (x < 0) ? 0 : 1;
}

double ActivationFunctions::d_binary(double x)
{
    return 0;
}

double ActivationFunctions::linear(double x)
{
    return x;
}

double ActivationFunctions::d_linear(double x)
{
    return 1;
}

double ActivationFunctions::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double ActivationFunctions::d_sigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

double ActivationFunctions::tanh(double x)
{
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

double ActivationFunctions::d_tanh(double x)
{
    return 1 - tanh(x) * tanh(x);
}

double ActivationFunctions::relu(double x)
{
    return (x > 0) ? x : 0;
}

double ActivationFunctions::d_relu(double x)
{
    return (x > 0) ? 1 : 0;
}

double ActivationFunctions::lrelu(double x)
{
    return (x > 0) ? x : 0.1 * x;
}

double ActivationFunctions::d_lrelu(double x)
{
    return (x > 0) ? 1 : 0.1;
}

double ActivationFunctions::elu(double x)
{
    return (x > 0) ? x : exp(-x) - 1;
}

double ActivationFunctions::d_elu(double x)
{
    return (x > 0) ? x : - exp(-x);
}

std::function<double(double)> ActivationFunctions::GetDerivativeFunctionName(std::function<double(double)> f)
{
    auto functionName = *f.target<double(*)(double)>();
    std::function<double(double)> df;
    
    if      (functionName == ActivationFunctions::binary)  { df = ActivationFunctions::d_binary; }
    else if (functionName == ActivationFunctions::linear)  { df = ActivationFunctions::d_linear; }
    else if (functionName == ActivationFunctions::sigmoid) { df = ActivationFunctions::d_sigmoid; }
    else if (functionName == ActivationFunctions::tanh)    { df = ActivationFunctions::d_tanh; }
    else if (functionName == ActivationFunctions::relu)    { df = ActivationFunctions::d_relu; }
    else if (functionName == ActivationFunctions::lrelu)   { df = ActivationFunctions::d_lrelu; }
    else if (functionName == ActivationFunctions::elu)     { df = ActivationFunctions::d_elu; }
    
    return df;
}


double LossFunctions::mse(std::vector<double> predicted, std::vector<double> actual)
{
    if (predicted.size() != actual.size())
    {
        throw(std::invalid_argument("Size of the predicted and true value vectors must be the same."));
    }

    // Find the sum of the squared difference
    double sum = 0;
    for (int i = 0 ; i < actual.size() ; i++)
    {
        sum += (actual[i] - predicted[i]) * (actual[i] - predicted[i]);
    }

    // Find the mean
    sum = sum / actual.size();
    return sum;
}

double LossFunctions::d_mse(std::vector<double> predicted, std::vector<double> actual)
{
    if (predicted.size() != actual.size())
    {
        throw(std::invalid_argument("Size of the predicted and true value vectors must be the same."));
    }

    double sum = 0;
    for (int i = 0 ; i < actual.size() ; i++)
    {
        sum += 2 * (predicted[i] - actual[i]);
    }

    // Find the mean
    sum = sum / actual.size();
    return sum;
}

double LossFunctions::mae(std::vector<double> predicted, std::vector<double> actual)
{
    if (predicted.size() != actual.size())
    {
        throw(std::invalid_argument("Size of the predicted and true value vectors must be the same."));
    }

     // Find the sum of the absolute difference
    double sum = 0;
    for (int i = 0 ; i < actual.size() ; i++)
    {
        sum += std::abs((actual[i] - predicted[i]));
    }

    // Find the mean
    sum = sum / actual.size();
    return sum;

}

double LossFunctions::d_mae(std::vector<double> predicted, std::vector<double> actual)
{
    if (predicted.size() != actual.size())
    {
        throw(std::invalid_argument("Size of the predicted and true value vectors must be the same."));
    }

     // Find the sum of the absolute difference
    double sum = 0;
    for (int i = 0 ; i < actual.size() ; i++)
    {
        if (predicted[i] > actual[i]) { sum++; }
        else if (predicted[i] < actual[i]) { sum--; }
        else { /* Do nothing **/ }
    }

    // Find the mean
    sum = sum / actual.size();
    return sum;
}

std::function<double(std::vector<double>, std::vector<double>)> LossFunctions::GetDerivativeFunctionName(std::function<double(std::vector<double>, std::vector<double>)> f)
{
    auto functionName = *f.target<double(*)(std::vector<double>, std::vector<double>)>();
    std::function<double(std::vector<double>, std::vector<double>)> df;
    
    if      (functionName == LossFunctions::mse)  { df = LossFunctions::d_mse; }
    else if (functionName == LossFunctions::mae)  { df = LossFunctions::d_mae; }
    
    return df;
}