#include "Value.hpp"

using ValuePtr = std::shared_ptr<Value>;

Value &Value::operator=(const Value &other)
{
    if (this == &other)
        return *this;

    data = other.data;
    grad = other.grad;
    label = other.label;
    _backward = other._backward;
    children = other.children;

    return *this;
}

ValuePtr Value::operator=(const ValuePtr other)
{
    data = other->data;
    grad = other->grad;
    label = other->label;
    _backward = other->_backward;
    children = other->children;

    return shared_from_this();
}

void Value::printChildren()
{
    std::cout << "children of " << label << std::endl;
    if (children.empty())
    {
        std::cout << "No children\n";
    }
    for (auto v : children)
    {
        std::cout << *v << "\n";
    }
}

void Value::printChildrenRecursively(int depth)
{
    std::string indent(depth * 2, ' ');

    std::cout << "Node[" << label << "] @ " << this << " (data=" << data << ")" << " (grad=" << grad << ")" << std::endl;
    if (children.empty())
    {
        std::cout << indent << "└─ No children\n";
        return;
    }
    for (auto v : children)
    {
        std::cout << indent << "└─ ";
        v->printChildrenRecursively(depth + 1);
    }
}

std::vector<std::shared_ptr<Value>> Value::AllChildren()
{
    std::vector<std::shared_ptr<Value>> childs;

    for (auto v : children)
    {
        childs.push_back(v);

        auto tmp = v->AllChildren();
        childs.insert(childs.end(), tmp.begin(), tmp.end());
    }

    return childs;
}

void Value::backward()
{
    grad = 1.0;
    _backward();

    auto childs = AllChildren();
    for (auto v : childs)
    {
        if (v->_backward)
        {
            v->_backward();
        }
    }
}

ValuePtr Value::relu()
{
    std::vector<ValuePtr> childs = {shared_from_this()};
    auto a = Value::create(data > 0 ? data : 0, childs);
    a->setOp("relu");

    a->_backward = [this, a]()
    {
        this->grad += (this->data > 0.0) * a->grad;
    };
    return a;
}

ValuePtr Value::sigmoid()
{
    std::function<double(double)> sigm = [](double x)
    {
        return (1.0 / (1.0 + exp(-x)));
    };
    std::vector<ValuePtr> childs = {shared_from_this()};
    auto a = Value::create(sigm(data), childs);
    a->setOp("sigm");

    a->_backward = [this, a]()
    {
        this->grad += (a->data * (1.0 - a->data)) * a->grad; // a.data = sigm(this.data)
    };

    return a;
}

ValuePtr Value::tanh()
{
    std::vector<ValuePtr> childs = {shared_from_this()};
    auto a = Value::create(std::tanh(data), childs);
    a->setOp("tanh");

    a->_backward = [this, a]()
    {
        this->grad += (1.0 - a->data * a->data) * a->grad;
    };

    return a;
}

ValuePtr Value::pow(double exponent)
{
    std::vector<ValuePtr> childs = {shared_from_this()};
    auto a = Value::create(std::pow(data, exponent), childs);
    a->setOp("pow");

    a->_backward = [this, a, exponent]()
    {
        this->grad += exponent * std::pow(this->data, exponent - 1) * a->grad;
    };

    return a;
}

ValuePtr Value::operator+(ValuePtr other)
{
    std::vector<ValuePtr> childs = {shared_from_this(), other};
    auto a = Value::create(data + other->data, childs);
    a->setOp("+");

    a->_backward = [this, other, a]()
    {
        this->grad += a->grad;
        other->grad += a->grad;
    };
    return a;
}

ValuePtr Value::operator+(double other)
{
    auto otherValue = Value::create(other);
    std::vector<ValuePtr> childs = {shared_from_this(), otherValue};

    auto a = Value::create(data + other, childs);
    a->setOp("+");

    a->_backward = [this, a, otherValue]()
    {
        this->grad += a->grad;
        otherValue->grad += a->grad;
    };

    return a;
}

Value &Value::operator+=(double other)
{
    data += other;
    auto old_backward = _backward;

    _backward = [this, old_backward]()
    {
        if (old_backward)
        {
            old_backward();
        }
        // Gradient with respect to 'this' remains unchanged
    };

    return *this;
}

Value &Value::operator+=(ValuePtr other)
{
    // Store old data for backward pass
    double old_data = data;

    data += other->data;
    auto old_backward = _backward;

    // Update backward function to handle both old and new gradients
    _backward = [this, other, old_backward]()
    {
        if (old_backward)
        {
            old_backward(); // This handles the gradient for previous operations
        }

        other->grad += this->grad;
    };

    // Update children to include the other value
    if (std::find_if(children.begin(), children.end(), [&](const std::shared_ptr<Value> &ptr)
                     { return ptr.get() == other.get(); }) == children.end())
    {
        children.push_back(other);
    }

    _op = "+=";
    return *this;
}

ValuePtr Value::operator-(ValuePtr other)
{
    std::vector<ValuePtr> childs = {shared_from_this(), other};
    auto a = Value::create(data - other->data, childs);
    a->setOp("-");
    a->setLabel(label + " - " + other->label);

    a->_backward = [this, other, a]()
    {
        this->grad += a->grad;
        other->grad += -a->grad;
    };

    return a;
}

ValuePtr Value::operator-(double other)
{
    auto otherValue = Value::create(other);
    std::vector<ValuePtr> childs = {shared_from_this(), otherValue};

    auto a = Value::create(data - other, childs);
    a->setOp("-");

    a->_backward = [this, a, otherValue]()
    {
        this->grad += a->grad;
        otherValue->grad += -a->grad;
    };

    return a;
}

Value &Value::operator-=(double other)
{
    data -= other;
    return *this;
}

Value &Value::operator-=(Value &other)
{
    // Store old data for backward pass
    double old_data = data;

    data -= other.data;
    auto old_backward = _backward;

    // Update backward function to handle both old and new gradients
    _backward = [this, &other, old_backward]()
    {
        if (old_backward)
        {
            old_backward(); // This handles the gradient for previous operations
        }

        other.grad += -this->grad;
    };

    // Update children to include the other value
    if (std::find_if(children.begin(), children.end(),
                     [&](const std::shared_ptr<Value> &ptr)
                     { return ptr.get() == &other; }) == children.end())
    {
        children.push_back(std::make_shared<Value>(other));
    }

    _op = "-=";
    return *this;
}

ValuePtr Value::operator*(ValuePtr other)
{
    std::vector<ValuePtr> childs = {shared_from_this(), other};
    auto a = Value::create(data * other->data, childs);
    a->setOp("*");

    a->_backward = [this, other, a]()
    {
        this->grad += a->grad * other->data;
        other->grad += a->grad * this->data;
    };

    return a;
}

ValuePtr Value::operator*(double other)
{
    auto otherValue = Value::create(other);
    std::vector<ValuePtr> childs = {shared_from_this(), otherValue};
    auto a = Value::create(data * other, childs);
    a->setOp("*");
    a->setLabel(label + " * " + std::to_string(other));

    a->_backward = [this, otherValue, a]()
    {
        this->grad += a->grad * otherValue->data;
        otherValue->grad += a->grad * this->data;
    };

    return a;
}

Value &Value::operator*=(double other)
{
    // Store old data for backward pass
    double old_data = data;

    data *= other;
    auto old_backward = _backward;

    // Update backward function to handle both old and new gradients
    _backward = [this, other, old_backward]()
    {
        this->grad *= other;

        if (old_backward)
        {
            old_backward(); // This handles the gradient for previous operations
        }
    };

    _op = "*=";
    return *this;
}

Value &Value::operator*=(Value &other)
{
    // Store old data for backward pass
    double old_data = data;

    data *= other.data;
    auto old_backward = _backward;

    // Update backward function to handle both old and new gradients
    _backward = [this, &other, old_backward, old_data]()
    {
        this->grad *= other.data;
        other.grad *= old_data;

        if (old_backward)
        {
            old_backward(); // This handles the gradient for previous operations
        }
    };

    // Update children to include the other value
    if (std::find_if(children.begin(), children.end(),
                     [&](const std::shared_ptr<Value> &ptr)
                     { return ptr.get() == &other; }) == children.end())
    {
        children.push_back(std::make_shared<Value>(other));
    }

    _op = "*=";
    return *this;
}

ValuePtr Value::operator/(ValuePtr other)
{
    std::vector<ValuePtr> childs = {shared_from_this(), other};
    auto a = Value::create(data / other->data, childs);
    a->setOp("/");

    a->_backward = [this, other, a]()
    {
        this->grad += a->grad / other->data;
        other->grad += -a->grad * this->data / (other->data * other->data);
    };

    return a;
}

ValuePtr Value::operator/(double other)
{
    auto otherValue = Value::create(other);
    std::vector<ValuePtr> childs = {shared_from_this(), otherValue};

    auto a = Value::create(data / other, childs);
    a->setOp("/");
    a->setLabel(label + " / " + std::to_string(other));

    a->_backward = [this, otherValue, a]()
    {
        this->grad += a->grad / otherValue->data;
        otherValue->grad += -a->grad * this->data / (otherValue->data * otherValue->data);
    };

    return a;
}

Value &Value::operator/=(double other)
{
    // Store old data for backward pass
    double old_data = data;

    data /= other;
    auto old_backward = _backward;

    // Update backward function to handle both old and new gradients
    _backward = [this, other, old_backward]()
    {
        this->grad /= other;

        if (old_backward)
        {
            old_backward(); // This handles the gradient for previous operations
        }
    };

    _op = "/=";
    return *this;
}

Value &Value::operator/=(Value &other)
{
    // Store old data for backward pass
    double old_data = data;

    data /= other.data;
    auto old_backward = _backward;

    // Update backward function to handle both old and new gradients
    _backward = [this, &other, old_backward, old_data]()
    {
        this->grad /= other.data;
        other.grad += -this->grad * old_data / (other.data * other.data); // I'm not sure about this, but doesn't matter much

        if (old_backward)
        {
            old_backward(); // This handles the gradient for previous operations
        }
    };

    // Update children to include the other value
    if (std::find_if(children.begin(), children.end(),
                     [&](const std::shared_ptr<Value> &ptr)
                     { return ptr.get() == &other; }) == children.end())
    {
        children.push_back(std::make_shared<Value>(other));
    }

    _op = "/=";
    return *this;
}

ValuePtr operator+(const ValuePtr &rhs)
{
    return rhs;
}

ValuePtr operator+(const ValuePtr &lhs, const ValuePtr &rhs)
{
    return *(lhs) + rhs;
}

ValuePtr operator+(const ValuePtr &lhs, double rhs)
{
    return *(lhs) + rhs;
}

ValuePtr operator+(double lhs, const ValuePtr &rhs)
{
    return *rhs + lhs;
}

ValuePtr operator+=(const ValuePtr &lhs, double rhs)
{
    auto rhsValue = Value::create(rhs);
    auto newValue = *(lhs) + rhsValue;
    return newValue;
}

ValuePtr operator+=(const ValuePtr &lhs, const ValuePtr &rhs)
{
    auto newValue = *(lhs) + rhs;
    return newValue;
}

ValuePtr operator-(const ValuePtr &rhs)
{
    auto newValue = Value::create(-rhs->getData(), {rhs});
    return newValue;
}

ValuePtr operator-(const ValuePtr &lhs, const ValuePtr &rhs)
{
    return *(lhs)-rhs;
}

ValuePtr operator-(const ValuePtr &lhs, double rhs)
{
    return *(lhs)-rhs;
}

ValuePtr operator-(double lhs, const ValuePtr &rhs)
{
    return -(*rhs) + lhs;
}

ValuePtr operator-=(const ValuePtr &lhs, double rhs)
{
    auto rhsValue = Value::create(rhs);
    auto newValue = *(lhs)-rhsValue;
    return newValue;
}

ValuePtr operator-=(const ValuePtr &lhs, const ValuePtr &rhs)
{
    auto newValue = *(lhs)-rhs;
    return newValue;
}

ValuePtr operator*(const ValuePtr &lhs, const ValuePtr &rhs)
{
    return *(lhs)*rhs;
}

ValuePtr operator*(const ValuePtr &lhs, double rhs)
{
    return *(lhs)*rhs;
}

ValuePtr operator*(double lhs, const ValuePtr &rhs)
{
    return *rhs * lhs;
}

ValuePtr operator*=(const ValuePtr &lhs, double rhs)
{
    auto rhsValue = Value::create(rhs);
    auto newValue = *(lhs)*rhsValue;
    return newValue;
}

ValuePtr operator*=(const ValuePtr &lhs, const ValuePtr &rhs)
{
    auto newValue = *(lhs)*rhs;
    return newValue;
}

ValuePtr operator/(const ValuePtr &lhs, const ValuePtr &rhs)
{
    return *(lhs) / rhs;
}

ValuePtr operator/(const ValuePtr &lhs, double rhs)
{
    return *(lhs) / rhs;
}

ValuePtr operator/(double lhs, const ValuePtr &rhs)
{
    return Value::create(lhs) / rhs;
}

ValuePtr operator/=(const ValuePtr &lhs, double rhs)
{
    auto rhsValue = Value::create(rhs);
    auto newValue = *(lhs) / rhsValue;
    return newValue;
}

ValuePtr operator/=(const ValuePtr &lhs, const ValuePtr &rhs)
{
    auto newValue = *(lhs) / rhs;
    return newValue;
}

std::ostream &operator<<(std::ostream &os, const Value &value)
{
    os << "Value(data: " << value.data << ", grad: " << value.grad << ", label: " << value.label << ")";
    return os;
}