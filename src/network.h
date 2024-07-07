#ifndef NETWORK_H
#define NETWORK_H

#include <iostream>
#include <random>

#include "neuron.h"

class NeuralNetwork
{
    public:
        /// @brief                   A neutal network that uses the neuron class for simple machine learning
        /// @param neuronsPerLayer   Vector containing the number of neurons that should be in each layer of the network
        /// @param inputFunction     The activation function that should be used by all neurons
        /// @param errorFunction     The function that should be used to calculate loss by all neurons
        /// @param inputEpochs       The number of epochs to train for
        /// @param inputLearningRate The factor that should be used when calculating gradient decent
        /// @param inputCutoff       When loss falls below this point, the model will exit early. Defaults to never exiting early
        NeuralNetwork(std::vector<int> neuronsPerLayer,
                      std::function<double(double)> inputFunction,
                      std::function<double(std::vector<double>, std::vector<double>)> inputErrorFunction,
                      int epochs = 1000,
                      double inputLearningRate = 0.01,
                      double inputCutoff = 0.0);

        /// @brief       Initializes the neural network to for a specific dataset, ensuring layers are correctly setup
        /// @param xData Vector containing vectors with all of the input data
        /// @param yData Vector containing vectors with all of the output data that this model should predict
        void Initialize(std::vector<std::vector<double>> xData, std::vector<std::vector<double>> yData);

        /// @brief Runs through the neural network for all data in the set, back-propogates and then updates weights
        void Train();

    private:
        /// @brief        Creates the input layer, which has no bias and doesn't alter data
        void SetupInputLayer();

        /// @brief Creates all of the hidden layers and hidden neurons, and sets up the output vector to track
        ///        data produced by these neurons during each run. Note that weights and biases are randomly
        ///        generated during this initialization process.
        void SetupHiddenLayers();

        /// @brief                Creates the output layer of the neural net
        void SetupOutputLayer();

        /// @brief              A single run through of the neural network
        /// @param currentIndex Row index of the data being trained on
        void Forward(int currentIndex);

        /// @brief  Propogates backward through the neural networks outputs, calculating the derivative at each point
        /// @param currentIndex Row index of the yData that results should be compared with
        void BackPropogate(int currentIndex);

        /// @brief  Generates a random number to be used when initializing biases and weights
        /// @return Randomly generated number
        double GenerateRandomNumber();

        /// Attributes of the neural network
        int epochs;
        int numInputs;
        int numOutputs;
        int numLayers;
        double cutoff;
        double epochErr;
        double learningRate;
        std::vector<std::vector<Neuron>> layers;
        std::function<double(double)> activationFunction;
        std::vector<std::vector<double>> xData;
        std::vector<std::vector<double>> yData;
        std::function<double(double, double)> errorFunctionDerivative;
        std::function<double(std::vector<double>, std::vector<double>)> errorFunction;
        bool initialized;
};

#endif // NETWORK_H