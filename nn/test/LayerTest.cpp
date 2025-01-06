
#include <gtest/gtest.h>
#include "nn/Layer.hpp"

class LayerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        layer = new Layer(2, 3, false); // 2 inputs -> 3 outputs
        // Set deterministic parameters for testing
        std::vector<Value::ValuePtr> params;
        for (int i = 0; i < layer->parameters().size(); i++)
        {
            params.push_back(Value::create(0.1));
        }
        layer->setParameters(params);
    }

    void TearDown() override
    {
        delete layer;
    }

    Layer *layer;
};

TEST_F(LayerTest, Forward)
{
    std::vector<double> input = {1.0, 1.0};
    auto output = (*layer)(input); // 1.0 * 0.1 + 1.0 * 0.1 + 0.1 * 1.0 = 0.3
    EXPECT_EQ(output.size(), 3);
    EXPECT_DOUBLE_EQ(output[0]->getData(), 0.3);
}

TEST_F(LayerTest, Architecture)
{
    EXPECT_NO_THROW(Layer(2, 3, true));
}

TEST_F(LayerTest, Parameters)
{
    auto params = layer->parameters();
    EXPECT_GT(params.size(), 0);
}