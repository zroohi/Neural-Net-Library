#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

#include <cmath>

namespace ActivationFunctions
{
    /// @brief   Namespace to hold common types of activaction functions that a user can input when creating a neuron.
    ///          Only simple activation functions requiring a single input are included, with the exception of the
    ///          exponention linear unit (elu0), which defaults to an alpha value of 1.

    inline float binary(float x)
    {
        float val {0};
        val = (x < 0) ? 0 : 1;
        return val;
    }

    inline float linear(float x)
    {
        return x;
    }

    inline float sigmoid(float x)
    {
        float val {0};
        val = 1 / (1 + exp(-x));
        return val;
    }

    inline float tanh(float x)
    {
        float val {0};
        val = (exp(x) - exp(-x)) / (exp(x) + exp(-x));
        return val;
    }

    inline float relu(float x)
    {
        float val {0};
        val = (x > 0) ? x : 0;
        return val;
    }

    inline float lrelu(float x)
    {
        float val {0};
        val = (x > 0.1 * x) ? x : 0.1 * x;
        return val;
    }

    inline float elu(float x, float a = 1)
    {
        float val {0};
        val = (x > 0) ? x : a * (exp(-x) - 1);
        return val;
    }

    inline float nullActivationFunction(float x) // This is used for neurons that haven't been setup yet
    {
        return x;
    }
};

#endif // ACTIVATION_FUNCTIONS_H