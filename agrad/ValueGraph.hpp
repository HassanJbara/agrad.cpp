#pragma once
#include <fstream>
#include <sstream>
#include <set>
#include <vector>
#include <sys/stat.h>
#include "agrad/Value.hpp"

class ValueGraph
{
private:
    static void trace(Value *v, std::set<Value *> &nodes, std::set<Value *> &edges)
    {
        if (!v || nodes.find(v) != nodes.end())
        {
            return;
        }

        nodes.insert(v);
        const auto children = v->getChildren();
        for (auto child : children)
        {
            if (child)
            {
                edges.insert(child.get());
                trace(child.get(), nodes, edges);
            }
        }
    }

public:
    static void visualize(Value *root, const std::string &filename, const std::string &rankdir = "LR", const std::string &output_dir = "../graphs/")
    {
        if (!root)
            return;

        // check if the output directory exists. If not, create it
        struct stat info;
        if (stat(output_dir.c_str(), &info) != 0)
        {
            std::string command = "mkdir -p " + output_dir;
            system(command.c_str());
        }

        std::set<Value *> nodes;
        std::set<Value *> edges;
        trace(root, nodes, edges);

        std::stringstream dot;
        dot << "digraph G {\n";
        dot << "rankdir=" << rankdir << ";\n";
        dot << "node [fontsize=12];\n";

        for (Value *n : nodes)
        {
            std::string node_id = "n" + std::to_string(reinterpret_cast<std::uintptr_t>(n));

            // Create three rows: label, data, grad
            std::string label = n->getLabel();
            std::stringstream node_label;
            node_label << "{";
            if (!label.empty())
            {
                node_label << label << "|";
            }
            node_label << "data " << std::fixed << std::setprecision(4) << n->getData() << "|"
                       << "grad " << std::fixed << std::setprecision(4) << n->getGrad()
                       << "}";

            dot << node_id << " [shape=record,label=\"" << node_label.str() << "\"];\n";

            const auto children = n->getChildren();
            if (!children.empty() && !n->getOp().empty())
            {
                std::string op_id = "op" + std::to_string(reinterpret_cast<std::uintptr_t>(n));
                dot << op_id << " [label=\"" << n->getOp() << "\",shape=circle];\n";

                // Connect all children to the operation node
                for (auto child : children)
                {
                    if (child)
                    {
                        std::string child_id = "n" + std::to_string(reinterpret_cast<std::uintptr_t>(child.get()));
                        dot << child_id << " -> " << op_id << ";\n";
                    }
                }
                // Connect operation to result
                dot << op_id << " -> " << node_id << ";\n";
            }
        }

        dot << "}\n";

        std::ofstream out(output_dir + filename);
        out << dot.str();
        out.close();

        system(("dot -Tpng " + output_dir + filename + " -o " + output_dir + filename + ".png").c_str());
        // remove the .dot file
        remove((output_dir + filename).c_str());
    }
};