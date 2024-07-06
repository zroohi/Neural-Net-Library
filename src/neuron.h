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
        Neuron(std::function<float(float)> inputFunction);

        /// @class               Simple implementation of a neuron that allows a variable number of inputs
        /// @param inputWeights  Vector of the weights that should be used for each input
        /// @param inputBias     Bias that should be added to the final calculation
        /// @param inputFunction The activation function that should be used by this neuron
        Neuron(std::vector<float> inputWeights, float inputBias, std::function<float(float)> inputFunction = ActivationFunctions::linear);

        /// @brief        Calculates the output for a given set of inputs to a neuron
        /// @param inputs Vector containing the inputs that should be fed to the neuron when doing a forward calculation
        /// @return       Output of this neuron, which is f(dotProduct + bias)
        float forward(std::vector<float> inputs);

        /// @brief       Calculates the dot product for two given vectors
        /// @param left  The left vector used to calculate the dot product
        /// @param right The right vector used to calculate the dot product 
        /// @return      The dot product
        float dotProduct(std::vector<float> left, std::vector<float> right);

        /// @brief         Initialize or change the weights on this neuron
        /// @param weights Vector of the weights that should be used for each input
        void setWeights(std::vector<float> inputWeights);

        /// @brief      Initialize or change the bias on this neuron
        /// @param bias Bias that should be added to the final calculation
        void setBias(float inputBias);

        /// @brief          Initialize or change the activation function on this neuron
        /// @param function The activation function that should be used by this neuron
        void setActivationFunction(std::function<float(float)> inputFunction);

        /// @brief  Checks if the neuron is fully setup, with weights, bias, and an activation function
        /// @return Boolean value representing if the neuron has been initalized or not
        bool isInitialized();
        

    private:
        /// These are used to indicate the neuron isn't fully setup
        std::vector<float> nullWeights = {};
        float nullBias = INT_MIN;

        /// Basic neuron attributes
        float bias;
        bool state;
        int numInputs;
        std::function<float(float)> f;
        std::function<float(float)> df;
        std::vector<float> weights;
        
        
        
};

#endif // NEURON_H