#pragma once
#include "nn/Module.hpp"
#include <vector>
#include <random>
#include <iostream>
#include <string>

class Neuron : public Module
{
private:
    std::vector<Value::ValuePtr> w;
    Value::ValuePtr b;
    bool nonlin;
    bool relu;
    int inputs;
    void initialize_weights()
    {
        // Create a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

        for (int i = 0; i < inputs; i++)
        {
            auto v = Value::create(dis(gen), "w" + std::to_string(i));
            w.push_back(v);
        }
    }

public:
    Neuron(int inputs) : b(Value::create(0, "b")), nonlin(true), inputs(inputs), relu(true)
    {
        initialize_weights();
    };
    Neuron(int inputs, bool nonlin, bool relu = true) : b(Value::create(0, "b")), nonlin(nonlin), inputs(inputs), relu(relu)
    {
        initialize_weights();
    };
    std::vector<Value::ValuePtr> parameters() const override
    {
        std::vector<Value::ValuePtr> p{b};
        p.insert(p.end(), w.begin(), w.end());
        return p;
    };

    void setParameters(std::vector<Value::ValuePtr> params)
    {
        if (params.size() != w.size() + 1)
        {
            throw std::invalid_argument("Parameter size mismatch");
        }

        b = params[0];
        w.clear();
        for (size_t i = 1; i < params.size(); i++)
        {
            w.push_back(params[i]);
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const Neuron &neuron)
    {
        os << (neuron.nonlin ? "ReLU" : "Linear") << "Neuron(" << neuron.w.size() << ")";
        return os;
    }

    Value::ValuePtr operator()(std::vector<Value::ValuePtr> x) // must return pointer to output for backprop to work
    {
        if (x.size() != inputs)
        {
            throw std::invalid_argument("Input size mismatch");
        }

        auto out = Value::create(b); // Start with bias

        for (size_t i = 0; i < inputs; i++)
        {
            out = out + x[i] * w[i];
        }

        if (nonlin)
        {
            out = relu ? out->relu() : out->tanh();
        }

        return out;
    }

    Value::ValuePtr operator()(std::vector<double> x)
    {
        if (x.size() != inputs)
        {
            throw std::invalid_argument("Input size mismatch");
        }

        std::vector<Value::ValuePtr> x_vals;
        for (size_t i = 0; i < inputs; i++)
        {
            x_vals.push_back(Value::create(x[i]));
        }

        return (*this)(x_vals);
    }
};