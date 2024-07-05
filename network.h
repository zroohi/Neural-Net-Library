#ifndef NETWORK_H
#define NETWORK_H

#include "neuron.h"

class NeuralNetwork
{
    public:
        /// @brief                 A neutal network that uses the neuron class for simple machine learning
        /// @param neuronsPerLayer Vector containing the number of neurons that should be in each layer of the network
        /// @param inputFunction   The activation function that should be used by all neurons
        NeuralNetwork(std::vector<int> neuronsPerLayer, std::function<float(float)> inputFunction);

        /// @brief            Initializes the neural network to for a specific dataset, ensuring layers are correctly setup
        /// @param inputs     Vector containing the input dataset
        /// @param numOutputs Number of outputs that should be calculated from this input dataset
        void initialize(std::vector<float> inputs, int numOutputs);

        /// @brief                Initializes the neural network, but allows the users to specify a different activation
        ///                       function for the output layer of neurons   
        /// @param inputs         Vector containing the input dataset
        /// @param numOutputs     Number of outputs that should be calculated from this input dataset
        /// @param outputFunction The activation function that should be used by all neurons
        void initialize(std::vector<float> inputs, int numOutputs, std::function<float(float)> outputFunction);

        /// @brief  A single run through of the neural network
        /// @return Vector containing the data produced by the output layer
        std::vector<float> forward();

    private:
        /// @brief        Creates the input layer, which has no bias and doesn't alter data
        /// @param inputs Vector containing the input data
        void SetupInputLayer(std::vector<float> inputs);

        /// @brief Creates all of the hidden layers and hidden neurons, and sets up the output vector to track
        ///        data produced by these neurons during each run. Note that weights and biases are randomly
        ///        generated during this initialization process.
        void SetupHiddenLayers();

        /// @brief                Creates the output layer of the neural net
        /// @param numOutputs     Number of output neurons in this layer
        /// @param outputFunction Activation function that should be used for the output layer
        void SetupOutputLayer(int numOutputs, std::function<float(float)> outputFunction);

        /// Attributes of the neural network
        std::vector<std::vector<Neuron>> layers;
        std::vector<std::vector<float>> outputs;
        std::function<float(float)> activationFunction;
        bool initialized;
};

#endif // NETWORK_H