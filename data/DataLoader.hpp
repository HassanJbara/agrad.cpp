#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <algorithm>

struct Dataset
{
    std::vector<std::vector<double>> X;
    std::vector<double> y;
};

struct sample
{
    std::vector<double> x;
    double y;
};
struct Batch
{
    std::vector<sample> samples;
    static constexpr size_t feature_dim = 2;
};

class DataLoader
{
private:
    DataLoader() = delete;
    static std::ifstream open_file(const std::string &filename)
    {
        // Check if file exists
        if (!std::filesystem::exists(filename))
        {
            throw std::runtime_error("File not found: " + filename);
        }

        // Check if file is empty
        if (std::filesystem::file_size(filename) == 0)
        {
            throw std::runtime_error("File is empty: " + filename);
        }

        std::ifstream file(filename);
        if (!file.is_open())
        {
            throw std::runtime_error("Unable to open file: " + filename);
        }

        return file;
    }

public:
    static Dataset load_dataset(const std::string &filename)
    {
        std::ifstream file = open_file(filename);
        std::string line;
        Dataset dataset;
        size_t line_number = 0;

        try
        {
            // Skip header
            std::getline(file, line);

            // Read and validate header
            if (line != "x1,x2,label")
            {
                throw std::runtime_error("Invalid file format: Expected header 'x1,x2,label'");
            }

            // Read data
            while (std::getline(file, line))
            {
                line_number++;
                std::stringstream ss(line);
                std::string value;
                std::vector<double> row;

                // Read x1, x2
                for (int i = 0; i < 2; i++)
                {
                    if (!std::getline(ss, value, ','))
                    {
                        throw std::runtime_error("Missing value in line " +
                                                 std::to_string(line_number));
                    }
                    try
                    {
                        row.push_back(std::stod(value));
                    }
                    catch (const std::exception &e)
                    {
                        throw std::runtime_error("Invalid number format in line " +
                                                 std::to_string(line_number) + ": " + value);
                    }
                }
                dataset.X.push_back(row);

                // Read label
                if (!std::getline(ss, value, ','))
                {
                    throw std::runtime_error("Missing label in line " +
                                             std::to_string(line_number));
                }
                try
                {
                    double label = std::stod(value);
                    if (label != -1.0 && label != 1.0)
                    {
                        throw std::runtime_error("Invalid label in line " +
                                                 std::to_string(line_number) +
                                                 ": Expected -1 or 1, got " + value);
                    }
                    dataset.y.push_back(label);
                }
                catch (const std::exception &e)
                {
                    throw std::runtime_error("Invalid label format in line " +
                                             std::to_string(line_number) + ": " + value);
                }
            }

            // Check if dataset is empty
            if (dataset.X.empty())
            {
                throw std::runtime_error("No data found in file");
            }
        }
        catch (const std::exception &e)
        {
            file.close();
            throw; // Re-throw the exception
        }

        file.close();
        return dataset;
    }

    static std::tuple<Dataset, Dataset> train_test_split(const Dataset &dataset, double split, size_t max_size = 0)
    {
        if (split < 0.0 || split > 1.0)
        {
            throw std::invalid_argument("split must be in the range [0, 1]");
        }

        Dataset train_set, test_set;
        double max_samples = max_size > 0 ? std::min(max_size, dataset.X.size()) : dataset.X.size();
        size_t train_samples = static_cast<size_t>(split * max_samples);

        for (size_t i = 0; i < max_samples; i++)
        {
            if (i < train_samples)
            {
                train_set.X.push_back(dataset.X[i]);
                train_set.y.push_back(dataset.y[i]);
            }
            else
            {
                test_set.X.push_back(dataset.X[i]);
                test_set.y.push_back(dataset.y[i]);
            }
        }

        return {train_set, test_set};
    }

    // Utility function to print dataset info
    static void print_dataset_info(const Dataset &dataset)
    {
        std::cout << "Dataset Information:" << std::endl;
        std::cout << "Number of samples: " << dataset.X.size() << std::endl;
        std::cout << "Number of features: " << (dataset.X.empty() ? 0 : dataset.X[0].size()) << std::endl;

        // Count classes
        int class_neg = 0, class_pos = 0;
        for (const auto &label : dataset.y)
        {
            if (label == -1)
                class_neg++;
            else if (label == 1)
                class_pos++;
        }
        std::cout << "Class distribution:" << std::endl;
        std::cout << "  Class -1: " << class_neg << std::endl;
        std::cout << "  Class  1: " << class_pos << std::endl;
    }
};
