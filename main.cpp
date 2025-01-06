#include "agrad/Value.hpp"
#include "data/DataLoader.hpp"
#include "nn/MLP.hpp"
#include "nn/Visualization.hpp"

int main()
{
    // Load the dataset
    auto dataset = DataLoader::load_dataset("../data/moon_dataset.csv");

    // Split the dataset into training and validation sets
    auto [train_dataset, val_dataset] = DataLoader::train_test_split(dataset, 1.0, 0);

    std::cout << "Train dataset size: " << train_dataset.X.size() << std::endl;
    std::cout << "Validation dataset size: " << val_dataset.X.size() << std::endl;

    MLP model(2, {16, 16, 1}, false);

    int EPOCHS = 500;
    double LEARNING_RATE = 0.001;
    int BATCH_SIZE = 1;
    double epoch_loss = 0.0;
    double epoch_val_loss = 0.0;
    double accuracy = 0.0;
    int training_batches = train_dataset.X.size() / BATCH_SIZE;
    int validation_batches = val_dataset.X.size() / BATCH_SIZE;

    for (int i = 0; i < EPOCHS; i++)
    {
        epoch_loss = 0.0;
        epoch_val_loss = 0.0;
        accuracy = 0.0;

        for (int j = 0; j < training_batches; j++)
        {
            std::vector<sample> batch_samples;
            for (int z = 0; z < BATCH_SIZE; z++)
            {
                batch_samples.push_back({train_dataset.X[j * BATCH_SIZE + z], train_dataset.y[j * BATCH_SIZE + z]});
            }

            // Forward pass
            std::vector<Value::ValuePtr> y_pred;
            for (auto &sample : batch_samples)
            {
                y_pred.push_back(model(sample.x)[0]);
            }

            // Compute loss and accuracy
            Value::ValuePtr loss = Value::create(0.0);

            for (size_t z = 0; z < BATCH_SIZE; z++)
            {
                loss = loss + (y_pred[z] - Value::create(batch_samples[z].y))->pow(2);
                accuracy += (y_pred[z]->getData() > 0.5) == (batch_samples[z].y == 1);
            }

            // Backward pass
            model.zero_grad();
            loss->backward();
            epoch_loss += loss->getData();

            // Update parameters
            for (auto &param : model.parameters())
            {
                param->setData(param->getData() - LEARNING_RATE * param->getGrad());
            }
        }

        // Validation
        for (int j = 0; j < validation_batches; j++)
        {
            std::vector<sample> batch_samples;
            for (int z = 0; z < BATCH_SIZE; z++)
            {
                batch_samples.push_back({val_dataset.X[j * BATCH_SIZE + z], val_dataset.y[j * BATCH_SIZE + z]});
            }

            // Forward pass
            std::vector<Value::ValuePtr> y_pred;
            for (auto &sample : batch_samples)
            {
                y_pred.push_back(model(sample.x)[0]);
            }

            // Compute loss
            Value::ValuePtr loss = Value::create(0.0);

            for (size_t z = 0; z < BATCH_SIZE; z++)
            {
                loss = loss + (y_pred[z] - Value::create(batch_samples[z].y))->pow(2);
                accuracy += (y_pred[z]->getData() > 0.5) == (batch_samples[z].y == 1);
            }

            epoch_val_loss += loss->getData();
        }

        epoch_loss /= training_batches;
        epoch_val_loss /= validation_batches;
        accuracy = (accuracy / dataset.X.size()) * 100.0;

        std::cout << "Epoch[" << i << "]: " << epoch_loss << ", Val: " << epoch_val_loss << ", Acc: " << accuracy << "%" << std::endl;
    }

    // Visualize the decision boundary
    auto predict_fn = [&model](const std::vector<double> &x)
    {
        return model(x)[0]->getData() > 0;
    };

    DatasetVisualizer::visualize_with_decision_boundary(dataset, predict_fn);

    return 0;
}