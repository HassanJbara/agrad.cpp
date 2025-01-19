#pragma once
#include <iostream>
#include <cmath>
#include <vector>
#include <memory>
#include <functional>
#include <set>

class Value : public std::enable_shared_from_this<Value>
{
public:
    using ValuePtr = std::shared_ptr<Value>;

private:
    double data;
    double grad;
    std::vector<ValuePtr> children;
    std::function<void()> _backward;
    std::string label;
    std::string _op;
    void appendChild(ValuePtr a) { children.push_back(a); }
    void build_topo(ValuePtr v, std::set<ValuePtr> &visited, std::vector<ValuePtr> &topo)
    {
        if (visited.find(v) == visited.end())
        {
            visited.insert(v);
            for (ValuePtr child : v->children)
            {
                build_topo(child, visited, topo);
            }
            topo.push_back(v);
        }
    }

public:
    Value() : data(0.0), grad(0.0), label(""), _op("") {}
    Value(double value) : data(value), grad(0.0), label(""), _op("") {}
    Value(double value, std::string label) : data(value), grad(0.0), label(label), _op("") {}
    Value(double value, std::string label, std::vector<ValuePtr> children) : data(value), grad(0.0), label(label), _backward(nullptr), _op(""), children(children) {}
    Value(double value, std::vector<ValuePtr> childs) : data(value), grad(0.0), label(""), _op(""), children(childs) {}
    // Copy constructor
    Value(const Value &other) : data(other.data), grad(other.grad), label(other.label), _backward(other._backward), _op(other._op), children(other.children) {}

    // copy factory
    static ValuePtr create(ValuePtr other)
    {
        return std::make_shared<Value>(*other);
    }

    static ValuePtr create(Value &other)
    {
        return std::make_shared<Value>(other);
    }

    static ValuePtr create(double value)
    {
        return std::make_shared<Value>(value);
    }

    static ValuePtr create(double value, std::string label)
    {
        return std::make_shared<Value>(value, label);
    }

    static ValuePtr create(double value, std::vector<ValuePtr> children)
    {
        return std::make_shared<Value>(value, children);
    }

    static ValuePtr create(double value, std::string label, std::vector<ValuePtr> children)
    {
        return std::make_shared<Value>(value, label, children);
    }

    void backward();
    ValuePtr relu();
    ValuePtr sigmoid();
    ValuePtr tanh();
    ValuePtr pow(double exponent);

    std::vector<ValuePtr> AllChildren();

    std::vector<ValuePtr> getChildren() const { return children; }
    void setChildren(std::vector<ValuePtr> new_children) { children = new_children; }
    double getData() const { return data; }
    void setData(double new_data) { data = new_data; }
    double getGrad() const { return grad; }
    void setGrad(double new_grad) { grad = new_grad; }
    std::string getLabel() const { return label; }
    void setLabel(std::string new_label) { label = new_label; }
    std::string getOp() { return _op; }
    void setOp(std::string new_op) { _op = new_op; }
    std::function<void()> getBackward() const { return _backward; }

    Value &operator=(const Value &other);
    ValuePtr operator=(const ValuePtr other);

    Value operator+() const
    {
        return *this;
    }
    ValuePtr operator+(ValuePtr other);
    ValuePtr operator+(double other);
    Value &operator+=(double other);
    Value &operator+=(ValuePtr other);

    Value operator-() const
    {
        Value a(-data, children);
        return a;
    }
    ValuePtr operator-(ValuePtr other);
    ValuePtr operator-(double other);
    Value &operator-=(double other);
    Value &operator-=(Value &other);

    ValuePtr operator*(ValuePtr other);
    ValuePtr operator*(double other);
    Value &operator*=(double other);
    Value &operator*=(Value &other);

    ValuePtr operator/(ValuePtr other);
    ValuePtr operator/(double other);
    Value &operator/=(double other);
    Value &operator/=(Value &other);

    friend std::ostream &operator<<(std::ostream &os, const Value &value);
    void printChildren();
    void printChildrenRecursively(int depth = 0);
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator+(const std::shared_ptr<Value> &lhs, double rhs);
std::shared_ptr<Value> operator+(double lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator+=(const std::shared_ptr<Value> &lhs, double rhs);
std::shared_ptr<Value> operator+=(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);

std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value> &lhs, double rhs);
std::shared_ptr<Value> operator-(double lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator-=(const std::shared_ptr<Value> &lhs, double rhs);
std::shared_ptr<Value> operator-=(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);

std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value> &lhs, double rhs);
std::shared_ptr<Value> operator*(double lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator*=(const std::shared_ptr<Value> &lhs, double rhs);
std::shared_ptr<Value> operator*=(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);

std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value> &lhs, double rhs);
std::shared_ptr<Value> operator/(double lhs, const std::shared_ptr<Value> &rhs);
std::shared_ptr<Value> operator/=(const std::shared_ptr<Value> &lhs, double rhs);
std::shared_ptr<Value> operator/=(const std::shared_ptr<Value> &lhs, const std::shared_ptr<Value> &rhs);
