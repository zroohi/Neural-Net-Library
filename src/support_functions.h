#ifndef SUPPORT_FUNCTIONS_H
#define SUPPORT_FUNCTIONS_H

#include <cmath>
#include <vector>
#include <unordered_map>

namespace ActivationFunctions
{
    /// @brief   Namespace to hold common types of activaction functions that a user can input when creating a neuron.
    ///          Only simple activation functions requiring a single input are included. The function getDerivativeFunctionName()
    ///          returns the derivative for each function type.
    /// @param x The input value to the activation function
    /// @return  The calculated function value

    float binary(float x);
    float linear(float x);
    float sigmoid(float x);
    float tanh(float x);
    float relu(float x);
    float lrelu(float x);
    float elu(float x);

    float d_binary(float x);
    float d_linear(float x);
    float d_sigmoid(float x);
    float d_tanh(float x);
    float d_relu(float x);
    float d_lrelu(float x);
    float d_elu(float x);

    std::function<float(float)> getDerivativeFunctionName(std::function<float(float)> f);
};

namespace LossFunctions
{
    float mse(std::vector<float> predicted, std::vector<float> actual);
    float mae(std::vector<float> predicted, std::vector<float> actual);

    float d_mse(float actual, float predicted);
    float d_mae(float actual, float predicted);

    std::function<float(float, float)> getDerivativeFunctionName(std::function<float(std::vector<float>, std::vector<float>)> f);
}

#endif // SUPPORT_FUNCTIONS_H