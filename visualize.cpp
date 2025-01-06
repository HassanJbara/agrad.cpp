#include "agrad/Value.hpp"
#include "agrad/ValueGraph.hpp"

int main(int argc, char *argv[])
{
    auto a = Value::create(1.0, "a");
    auto b = Value::create(2.0, "b");
    auto c = Value::create(3.0, "c");

    auto g = a * b;
    auto f = g * c;

    g->setLabel("g");
    f->setLabel("f");
    f->backward();

    ValueGraph::visualize(f.get(), "graph");

    return 0;
}