#pragma once

#include <vector>
#include "agrad/Value.hpp"

class Module
{
public:
    virtual std::vector<Value::ValuePtr> parameters() const
    {
        std::vector<Value::ValuePtr> p;
        return p;
    }
    void zero_grad()
    {
        for (auto v : parameters())
        {
            v->setGrad(0);
        }
    }
};