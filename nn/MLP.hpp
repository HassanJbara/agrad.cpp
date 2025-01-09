#pragma once

#include "agrad/Value.hpp"
#include "nn/Layer.hpp"
#include "nn/Module.hpp"

#include <vector>

class MLP : public Module
{
private:
    int inputs;
    bool relu;
    std::vector<int> outputs;
    std::vector<Layer> layers;

public:
    MLP(int inputs, std::vector<int> outputs, bool relu = true) : inputs(inputs), outputs(outputs), relu(relu)
    {
        std::vector<int> dims = {inputs};
        dims.insert(dims.end(), outputs.begin(), outputs.end());
        for (int i = 0; i < outputs.size(); i++)
        {
            Layer l(dims[i], dims[i + 1], i != outputs.size() - 1, relu);
            layers.push_back(l);
        }
    }

    std::vector<Value::ValuePtr> parameters() const override
    {
        std::vector<Value::ValuePtr> params;
        for (int i = 0; i < layers.size(); i++)
        {
            auto l_p = layers[i].parameters();
            params.insert(params.end(), l_p.begin(), l_p.end());
        }

        return params;
    }

    void setParameters(std::vector<Value::ValuePtr> params)
    {
        int offset = 0;
        for (int i = 0; i < layers.size(); i++)
        {
            std::vector<Value::ValuePtr> l_params(params.begin() + offset, params.begin() + offset + layers[i].parameters().size());
            layers[i].setParameters(l_params);
            offset += layers[i].parameters().size();
        }
    }

    std::vector<Value::ValuePtr> operator()(std::vector<Value::ValuePtr> x)
    {
        auto current = layers[0](x);

        for (int i = 1; i < layers.size(); i++)
        {
            current = layers[i](current);
        }

        return current;
    }

    std::vector<Value::ValuePtr> operator()(std::vector<double> x)
    {
        auto current = layers[0](x);

        for (int i = 1; i < layers.size(); i++)
        {
            current = layers[i](current);
        }

        return current;
    }

    std::vector<Value::ValuePtr> operator()(std::vector<std::vector<double>> x)
    {
        std::vector<Value::ValuePtr> output;
        for (int i = 0; i < x.size(); i++)
        {
            auto current = operator()(x[i]);
            output.insert(output.end(), current.begin(), current.end());
        }

        return output;
    }

    std::vector<Value::ValuePtr> operator()(std::vector<std::vector<Value::ValuePtr>> x)
    {
        std::vector<Value::ValuePtr> output;
        for (int i = 0; i < x.size(); i++)
        {
            auto current = operator()(x[i]);
            output.insert(output.end(), current.begin(), current.end());
        }

        return output;
    }

    friend std::ostream &operator<<(std::ostream &os, const MLP &mlp)
    {
        os << "MLP of [";
        for (size_t i = 0; i < mlp.layers.size(); ++i)
        {
            os << mlp.layers[i];
            if (i != mlp.layers.size() - 1)
            {
                os << ", ";
            }
        }
        os << "]";
        return os;
    }
};