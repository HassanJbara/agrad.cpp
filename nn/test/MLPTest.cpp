
#include <gtest/gtest.h>
#include "nn/MLP.hpp"

class MLPTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mlp = new MLP(2, {3, 1}); // 2 inputs -> 3 hidden -> 1 output
        // Set deterministic parameters for testing
        std::vector<Value::ValuePtr> params;
        for (int i = 0; i < mlp->parameters().size(); i++)
        {
            params.push_back(Value::create(0.1));
        }
        mlp->setParameters(params);
    }

    void TearDown() override
    {
        delete mlp;
    }

    MLP *mlp;
};

TEST_F(MLPTest, Forward)
{
    std::vector<double> input = {1.0, 1.0};
    auto output = (*mlp)(input);
    EXPECT_EQ(output.size(), 1);
    EXPECT_NO_THROW(output[0]->getData());
}

TEST_F(MLPTest, Architecture)
{
    EXPECT_NO_THROW(MLP(2, {3, 1}));
    EXPECT_NO_THROW(MLP(2, {3, 2, 1}));
}

TEST_F(MLPTest, Parameters)
{
    auto params = mlp->parameters();
    EXPECT_GT(params.size(), 0);
}