#include "agrad/Value.hpp"
#include "agrad/ValueGraph.hpp"

// int main(int argc, char *argv[])
// {
//     auto a = Value::create(1.0, "a");
//     auto b = Value::create(2.0, "b");
//     auto c = Value::create(3.0, "c");

//     auto g = a * b;
//     auto f = g * c;

//     g->setLabel("g");
//     f->setLabel("f");
//     f->backward();

//     ValueGraph::visualize(f.get(), "graph");

//     return 0;
// }
int main(int argc, char *argv[])
{
    auto x1 = Value::create(2.0, "x1");
    auto x2 = Value::create(0.0, "x2");

    // Create a simple neural network computation
    auto w1 = Value::create(-3.0, "w1");
    auto w2 = Value::create(1.0, "w2");
    auto b = Value::create(6.8813735870195432, "b");

    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto sum = x1w1 + x2w2 + b;
    auto o = sum->tanh();

    o->setLabel("output");
    o->backward();

    ValueGraph::visualize(o.get(), "graph");

    return 0;
}