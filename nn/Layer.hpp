#pragma once

#include "nn/Module.hpp"
#include "nn/Neuron.hpp"

class Layer : public Module
{
private:
    int inputs;
    int outputs;
    bool nonlin;
    bool relu;
    std::vector<Neuron> neurons;

public:
    Layer(int inputs, int outputs, bool nonlin, bool relu = true) : inputs(inputs), outputs(outputs), nonlin(nonlin), relu(relu)
    {
        for (int i = 0; i < outputs; i++)
        {
            Neuron n(inputs, nonlin, relu);
            neurons.push_back(n);
        }
    }

    std::vector<Value::ValuePtr> parameters() const override
    {
        std::vector<Value::ValuePtr> params;
        for (int i = 0; i < neurons.size(); i++)
        {
            auto n_p = neurons[i].parameters();
            params.insert(params.end(), n_p.begin(), n_p.end());
        }
        return params;
    }

    void setParameters(std::vector<Value::ValuePtr> params)
    {
        if (params.size() != neurons.size() * (inputs + 1))
        {
            throw std::invalid_argument("Parameter size mismatch");
        }

        for (int i = 0; i < neurons.size(); i++)
        {
            std::vector<Value::ValuePtr> n_params(params.begin() + i * (inputs + 1), params.begin() + (i + 1) * (inputs + 1));
            neurons[i].setParameters(n_params);
        }
    }

    std::vector<Value::ValuePtr> operator()(std::vector<double> x)
    {
        std::vector<Value::ValuePtr> out;
        for (Neuron &n : neurons)
        {
            out.push_back(n(x));
        }
        return out;
    }

    std::vector<Value::ValuePtr> operator()(std::vector<Value::ValuePtr> x)
    {
        std::vector<Value::ValuePtr> out;
        for (Neuron &n : neurons)
        {
            out.push_back(n(x));
        }
        return out;
    }

    friend std::ostream &operator<<(std::ostream &os, const Layer &layer)
    {
        os << "Layer of [";
        for (size_t i = 0; i < layer.neurons.size(); ++i)
        {
            os << layer.neurons[i];
            if (i != layer.neurons.size() - 1)
            {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
};