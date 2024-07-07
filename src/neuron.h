#ifndef NEURON_H
#define NEURON_H

#include <vector>
#include <numeric>
#include <stdexcept>
#include <functional>
#include "support_functions.h"

class Neuron
{
    public:
        /// @brief Create a simple, uninitialized neuron, which must be setup later to be used
        Neuron();

        /// @brief               Create a neuron with no bias or weights, they must be set later on before the neuron is used
        /// @param inputFunction 
        Neuron(std::function<double(double)> inputFunction);

        /// @class               Simple implementation of a neuron that allows a variable number of inputs
        /// @param inputWeights  Vector of the weights that should be used for each input
        /// @param inputBias     Bias that should be added to the final calculation
        /// @param inputFunction The activation function that should be used by this neuron
        Neuron(std::vector<double> inputWeights, double inputBias, std::function<double(double)> inputFunction = ActivationFunctions::linear);

        /// @brief        Calculates the output for a given set of inputs to a neuron
        /// @param inputs Vector containing the inputs that should be fed to the neuron when doing a forward calculation
        /// @return       Output of this neuron, which is f(dotProduct + bias)
        double Forward(std::vector<double> inputs);

        /// @brief        Calculates the output to the derivative for the activation function, which is needed for backwards propogation
        /// @param inputs Vector containing the inputs that should be fed to the neuron when doing a backwards calculation
        /// @return       Output of this neuron, which is df(dotProduct + bias)
        double Backward(std::vector<double> inputs);

        /// @brief       Calculates the dot product for two given vectors
        /// @param left  The left vector used to calculate the dot product
        /// @param right The right vector used to calculate the dot product 
        /// @return      The dot product
        double DotProduct(std::vector<double> left, std::vector<double> right);

        /// @brief         Initialize or set all of the weights on this neuron
        /// @param weights Vector of the weights that should be used for each input
        void SetWeights(std::vector<double> inputWeights);

        /// @brief               Update the value of a specific weight on this neuron by the negative value of delta (i.e. weight - delta)
        /// @param delta         Positive or negative amount that the weight should be updated by
        /// @param indexOfWeight Index of the weight to be updated in the weights vector
        void UpdateOneWeight(double delta, int indexOfWeight);

        /// @brief  Returns a vector containing all of weights for this neuron
        /// @return Vector of weights 
        std::vector<double> GetWeights();

        /// @brief  Get the number of weights, which is equal to the number of inputs for this neuron
        /// @return The number of weights / inputs
        int GetNumWeights();

        /// @brief  Get the bias on this neuron
        /// @return The bias
        double GetBias();

        /// @brief      Initialize or change the bias on this neuron
        /// @param bias Bias that should be added to the final calculation
        void SetBias(double inputBias);

        /// @brief      Update the value of the bias on this neuron by the negative value of delta (i.e. weight - delta)
        /// @param bias Bias that should be added to the final calculation
        void UpdateBias(double delta);

        /// @brief          Initialize or change the activation function on this neuron
        /// @param function The activation function that should be used by this neuron
        void SetActivationFunction(std::function<double(double)> inputFunction);

        /// @brief  Returns the ouput of the activation function for a given value
        /// @return Output of the activation function
        double GetActivationFunctionValue(double input);

        /// @brief  Returns the ouput of the activation function derivative for a given value
        /// @return Output of the activation function derivative
        double GetActivationFunctionDerivativeValue(double input);

        /// @brief  Checks if the neuron is fully setup, with weights, bias, and an activation function
        /// @return Boolean value representing if the neuron has been initalized or not
        bool IsInitialized();
        

    private:
        /// These are used to indicate the neuron isn't fully setup
        std::vector<double> nullWeights = {};
        double nullBias = INT_MIN;

        /// Basic neuron attributes
        double bias;
        bool state;
        int numInputs;
        std::function<double(double)> f;
        std::function<double(double)> df;
        std::vector<double> weights;
        
        
        
};

#endif // NEURON_H