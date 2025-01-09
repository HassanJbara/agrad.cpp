#pragma once
#include <vector>
#include "agrad/Value.hpp"
#include <matplot/matplot.h>
#include "nn/MLP.hpp"
#include "data/DataLoader.hpp"

class Visualization
{
private:
    std::vector<std::vector<double>> X; // Input features
    std::vector<double> y;              // Labels
    MLP &model;                         // Reference to the model

    // Helper function to create mesh grid
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
    meshgrid(const std::vector<double> &x, const std::vector<double> &y)
    {
        size_t nx = x.size();
        size_t ny = y.size();
        std::vector<std::vector<double>> X(ny, std::vector<double>(nx));
        std::vector<std::vector<double>> Y(ny, std::vector<double>(nx));

        for (size_t i = 0; i < ny; i++)
        {
            for (size_t j = 0; j < nx; j++)
            {
                X[i][j] = x[j];
                Y[i][j] = y[i];
            }
        }
        return {X, Y};
    }

    // Helper function to get min/max of a column
    std::pair<double, double> get_column_range(const std::vector<std::vector<double>> &data, int col)
    {
        double min_val = std::numeric_limits<double>::max();
        double max_val = std::numeric_limits<double>::lowest();

        for (const auto &row : data)
        {
            min_val = std::min(min_val, row[col]);
            max_val = std::max(max_val, row[col]);
        }
        return {min_val, max_val};
    }

    // Create range with step size
    std::vector<double> arange(double start, double end, double step)
    {
        std::vector<double> range;
        for (double i = start; i < end; i += step)
        {
            range.push_back(i);
        }
        return range;
    }

public:
    Visualization(std::vector<std::vector<double>> &features,
                  std::vector<double> &labels,
                  MLP &mlp_model)
        : X(features), y(labels), model(mlp_model) {}

    void plot_decision_boundary(double h = 0.25)
    {
        // Get ranges for x and y
        auto [x_min, x_max] = get_column_range(X, 0);
        auto [y_min, y_max] = get_column_range(X, 1);

        x_min -= 1.0;
        x_max += 1.0;
        y_min -= 1.0;
        y_max += 1.0;

        // Create mesh grid
        auto x_range = arange(x_min, x_max, h);
        auto y_range = arange(y_min, y_max, h);
        auto [xx, yy] = meshgrid(x_range, y_range);

        // Create input points from mesh
        std::vector<std::vector<double>> X_mesh;
        for (size_t i = 0; i < xx.size(); i++)
        {
            for (size_t j = 0; j < xx[0].size(); j++)
            {
                X_mesh.push_back({xx[i][j], yy[i][j]});
            }
        }

        // Evaluate model on mesh points
        std::vector<bool> Z;
        for (const auto &x_point : X_mesh)
        {
            Value score = *model(x_point)[0];
            Z.push_back(score.getData() > 0);
        }

        // Reshape Z to match xx shape
        std::vector<std::vector<bool>> Z_reshaped(xx.size(),
                                                  std::vector<bool>(xx[0].size()));
        size_t k = 0;
        for (size_t i = 0; i < xx.size(); i++)
        {
            for (size_t j = 0; j < xx[0].size(); j++)
            {
                Z_reshaped[i][j] = Z[k++];
            }
        }

        // Plot using matplot++
        auto f = matplot::figure(true);

        // Plot decision boundary
        auto ax = f->add_axes();
        ax->contourf(xx, yy, Z_reshaped);

        // Plot data points
        std::vector<double> x_pos, y_pos, x_neg, y_neg;
        for (size_t i = 0; i < X.size(); i++)
        {
            if (y[i] == 1)
            {
                x_pos.push_back(X[i][0]);
                y_pos.push_back(X[i][1]);
            }
            else
            {
                x_neg.push_back(X[i][0]);
                y_neg.push_back(X[i][1]);
            }
        }

        // Plot positive and negative points with different colors
        ax->scatter(x_pos, y_pos)->marker_face_color("r"); // Red for positive class
        ax->scatter(x_neg, y_neg)->marker_face_color("b"); // Blue for negative class

        // Set plot limits
        ax->xlim({x_min, x_max});
        ax->ylim({y_min, y_max});

        // Show the plot
        matplot::show();
    }
};

class DatasetVisualizer
{
public:
    static void visualize_dataset(const Dataset &dataset, const std::string &title = "Dataset Visualization")
    {
        std::vector<double> x1_pos, x2_pos; // Features for positive class
        std::vector<double> x1_neg, x2_neg; // Features for negative class

        // Separate data points by class
        for (size_t i = 0; i < dataset.X.size(); i++)
        {
            if (dataset.y[i] == 1.0)
            {
                x1_pos.push_back(dataset.X[i][0]);
                x2_pos.push_back(dataset.X[i][1]);
            }
            else
            {
                x1_neg.push_back(dataset.X[i][0]);
                x2_neg.push_back(dataset.X[i][1]);
            }
        }

        // Create figure
        auto f = matplot::figure(true);
        f->size(1200, 800);

        // Create scatter plots
        auto s1 = matplot::scatter(x1_pos, x2_pos);
        s1->marker_color({1, 0, 0}); // Red for positive class
        s1->marker_size(10);
        s1->display_name("Class +1");

        matplot::hold(matplot::on);

        auto s2 = matplot::scatter(x1_neg, x2_neg);
        s2->marker_color({0, 0, 1}); // Blue for negative class
        s2->marker_size(10);
        s2->display_name("Class -1");

        // Customize the plot
        matplot::title(title);
        matplot::xlabel("Feature X1");
        matplot::ylabel("Feature X2");
        matplot::legend();
        matplot::grid(matplot::on);

        // Show the plot
        matplot::show();
    }

    static void visualize_with_decision_boundary(
        const Dataset &dataset,
        const std::function<int(double, double)> &classifier,
        const std::string &title = "Decision Boundary Visualization")
    {
        // Create figure
        auto f = matplot::figure(true);
        f->size(1200, 800);

        // Plot decision boundary contour
        auto c = matplot::fcontour(classifier, "b")->filled(true).line_width(0.5);
        matplot::colormap({{1.0, 0.8, 0.8}, {0.9, 1.0, 0.9}});

        matplot::hold(matplot::on);

        // Plot dataset points
        std::vector<double> x1_pos, x2_pos;
        std::vector<double> x1_neg, x2_neg;

        for (size_t i = 0; i < dataset.X.size(); i++)
        {
            if (dataset.y[i] == 1.0)
            {
                x1_pos.push_back(dataset.X[i][0]);
                x2_pos.push_back(dataset.X[i][1]);
            }
            else
            {
                x1_neg.push_back(dataset.X[i][0]);
                x2_neg.push_back(dataset.X[i][1]);
            }
        }

        matplot::scatter(x1_pos, x2_pos)->marker_color({1, 0, 0}).marker_face(true).display_name("Class +1");
        matplot::scatter(x1_neg, x2_neg)->marker_color({0, 0, 1}).marker_face(true).display_name("Class -1");

        // Customize the plot
        matplot::title(title);
        matplot::xlabel("Feature X1");
        matplot::ylabel("Feature X2");
        matplot::colorbar(matplot::off);
        matplot::grid(matplot::on);

        // Show the plot
        matplot::show();
    }
};
