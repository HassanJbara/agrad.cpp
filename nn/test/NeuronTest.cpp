
#include <gtest/gtest.h>
#include "nn/Neuron.hpp"

class NeuronTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        neuron = new Neuron(2, false); // Linear neuron with 2 inputs
        std::vector<Value::ValuePtr> params = {
            Value::create(1.0, "b"),  // bias
            Value::create(0.5, "w0"), // weight 1
            Value::create(0.5, "w1")  // weight 2
        };
        neuron->setParameters(params);
    }

    void TearDown() override
    {
        delete neuron;
    }

    Neuron *neuron;
};

TEST_F(NeuronTest, Forward)
{
    std::vector<double> input = {1.0, 1.0};
    Value::ValuePtr output = (*neuron)(input);
    EXPECT_DOUBLE_EQ(output->getData(), 2.0); // 1.0 + 0.5*1.0 + 0.5*1.0 = 2.0
}

TEST_F(NeuronTest, Parameters)
{
    auto params = neuron->parameters();
    EXPECT_EQ(params.size(), 3);
    EXPECT_DOUBLE_EQ(params[0]->getData(), 1.0); // bias
    EXPECT_DOUBLE_EQ(params[1]->getData(), 0.5); // w0
    EXPECT_DOUBLE_EQ(params[2]->getData(), 0.5); // w1
}

TEST_F(NeuronTest, InvalidInput)
{
    std::vector<double> invalid_input = {1.0}; // Only one input
    EXPECT_THROW((*neuron)(invalid_input), std::invalid_argument);
}