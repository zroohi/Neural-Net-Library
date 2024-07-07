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

    double binary(double x);
    double linear(double x);
    double sigmoid(double x);
    double tanh(double x);
    double relu(double x);
    double lrelu(double x);
    double elu(double x);

    double d_binary(double x);
    double d_linear(double x);
    double d_sigmoid(double x);
    double d_tanh(double x);
    double d_relu(double x);
    double d_lrelu(double x);
    double d_elu(double x);

    std::function<double(double)> GetDerivativeFunctionName(std::function<double(double)> f);
};

namespace LossFunctions
{
    double mse(std::vector<double> predicted, std::vector<double> actual);
    double mae(std::vector<double> predicted, std::vector<double> actual);

    double d_mse(std::vector<double> predicted, std::vector<double> actual);
    double d_mae(std::vector<double> predicted, std::vector<double> actual);

    std::function<double(std::vector<double>, std::vector<double>)> GetDerivativeFunctionName(std::function<double(std::vector<double>, std::vector<double>)> f);
}

#endif // SUPPORT_FUNCTIONS_H