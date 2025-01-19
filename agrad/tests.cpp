
#include <gtest/gtest.h>
#include "agrad/Value.hpp"

using ValuePtr = std::shared_ptr<Value>;

class ValueTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        v0 = Value::create(0.0);
        v1 = Value::create(1.0);
        v2 = Value::create(2.0);
        minusV1 = Value::create(-1.0);
        minusV2 = Value::create(-2.0);
    }

    ValuePtr v0;
    ValuePtr v1;
    ValuePtr v2;
    ValuePtr minusV1;
    ValuePtr minusV2;
};

TEST_F(ValueTest, Addition)
{
    ValuePtr result = v1 + v2;
    result->backward();
    EXPECT_DOUBLE_EQ(result->getData(), 3.0);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 1.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 1.0);
}

TEST_F(ValueTest, Multiplication)
{
    ValuePtr result = v1 * v2;
    result->backward();
    EXPECT_DOUBLE_EQ(result->getData(), 2.0);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 2.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 1.0);
}

TEST_F(ValueTest, ReLU)
{
    ValuePtr a = Value::create(1.0);
    ValuePtr b = Value::create(-1.0);
    ValuePtr pos = a->relu();
    ValuePtr neg = b->relu();
    pos->backward();
    neg->backward();
    EXPECT_DOUBLE_EQ(pos->getData(), 1.0);
    EXPECT_DOUBLE_EQ(neg->getData(), 0.0);
    EXPECT_DOUBLE_EQ(a->getGrad(), 1.0);
    EXPECT_DOUBLE_EQ(b->getGrad(), 0.0);
}

TEST_F(ValueTest, Backward)
{
    ValuePtr c = v1 * v2;
    c->backward();
    EXPECT_DOUBLE_EQ(v1->getGrad(), 2.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 1.0);
}

TEST_F(ValueTest, Sigmoid)
{
    ValuePtr a = v0->sigmoid();
    ValuePtr b = v1->sigmoid();
    a->backward();
    b->backward();
    EXPECT_DOUBLE_EQ(a->getData(), 0.5);
    EXPECT_DOUBLE_EQ(b->getData(), 0.7310585786300049);
    EXPECT_DOUBLE_EQ(v0->getGrad(), 0.25);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 0.19661193324148185);
}

TEST_F(ValueTest, Tanh)
{
    ValuePtr a = v0->tanh();
    ValuePtr b = v1->tanh();
    ValuePtr c = minusV1->tanh();
    a->backward();
    b->backward();
    c->backward();
    EXPECT_DOUBLE_EQ(a->getData(), 0);
    EXPECT_DOUBLE_EQ(b->getData(), 0.76159415595576485);
    EXPECT_DOUBLE_EQ(c->getData(), -0.76159415595576485);
    EXPECT_DOUBLE_EQ(v0->getGrad(), 1);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 0.41997434161402614);
    EXPECT_DOUBLE_EQ(minusV1->getGrad(), 0.41997434161402614);
}

TEST_F(ValueTest, Pow)
{
    ValuePtr b = v2->pow(3);
    b->backward();
    EXPECT_DOUBLE_EQ(b->getData(), 8.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 12.0);
}

TEST_F(ValueTest, Subtraction)
{
    ValuePtr result = v1 - v2;
    result->backward();
    EXPECT_DOUBLE_EQ(result->getData(), -1.0);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 1.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), -1.0);
}

TEST_F(ValueTest, Division)
{
    ValuePtr result = v1 / v2;
    result->backward();
    EXPECT_DOUBLE_EQ(result->getData(), 0.5);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 0.5);
    EXPECT_DOUBLE_EQ(v2->getGrad(), -0.25);
}

TEST_F(ValueTest, UnaryMinus)
{
    ValuePtr result = -v1;
    EXPECT_DOUBLE_EQ(result->getData(), -1.0);
}

TEST_F(ValueTest, UnaryPlus)
{
    ValuePtr result = +v1;
    EXPECT_DOUBLE_EQ(result->getData(), 1.0);
}

TEST_F(ValueTest, Assignment)
{
    v1 = v2;
    EXPECT_DOUBLE_EQ(v1->getData(), 2.0);
}

TEST_F(ValueTest, CompoundAssignment)
{
    ValuePtr c = Value::create(3.0);
    ValuePtr d = v1 * v2;
    d = d += c; // 1 * 2 + 3 = 5

    for (int i = 0; i < 2; i++)
    {
        d = d += 1.0;
    } // 5 + 2 = 7

    for (int i = 0; i < 2; i++)
    {
        ValuePtr e = Value::create(1.0);
        d = d += e;
    } // 7 + 2 = 9

    ValuePtr e = Value::create(2.0);
    ValuePtr f = d * e;

    f->backward();
    EXPECT_DOUBLE_EQ(d->getData(), 9.0);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 4.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 2.0);
    EXPECT_DOUBLE_EQ(c->getGrad(), 2.0);
}

TEST_F(ValueTest, CompoundSubtraction)
{
    ValuePtr c = Value::create(2.0);
    ValuePtr d = v1 * v2;
    d = d -= c; // 1 * 2 - 2 = 0

    for (int i = 0; i < 2; i++)
    {
        d = d -= 1.0;
    } // 0 - 2 = -2

    for (int i = 0; i < 2; i++)
    {
        ValuePtr e = Value::create(1.0);
        d = d -= e;
    } // -2 - 2 = -4

    ValuePtr e = Value::create(2.0);
    ValuePtr f = d * e;

    f->backward();
    EXPECT_DOUBLE_EQ(d->getData(), -4.0);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 4.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 2.0);
    EXPECT_DOUBLE_EQ(c->getGrad(), -2.0);
}

TEST_F(ValueTest, CompoundMultiplication)
{
    ValuePtr d = v1 * v2; // 1 + 2 = 2

    for (int i = 0; i < 2; i++)
    {
        d = d *= 2.0;
    } // 2 * 2 * 2 = 8

    for (int i = 0; i < 2; i++)
    {
        ValuePtr e = Value::create(2.0);
        d = d *= e;
    } // 8 * 2 * 2 = 32

    ValuePtr e = Value::create(2.0);
    ValuePtr f = d * e;

    f->backward();
    EXPECT_DOUBLE_EQ(d->getData(), 32.0);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 64.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 32.0);
    EXPECT_DOUBLE_EQ(e->getGrad(), 32.0);
}

TEST_F(ValueTest, CompoundDivision)
{
    v1->setData(32.0);
    v2->setData(32.0);
    ValuePtr d = v1 + v2;

    for (int i = 0; i < 2; i++)
    {
        d = d /= 2.0;
    } // 64 / 2 / 2 = 16

    for (int i = 0; i < 2; i++)
    {
        ValuePtr e = Value::create(2.0);
        d = d /= e;
    } // 16 / 2 / 2 = 4

    ValuePtr e = Value::create(32.0);
    ValuePtr f = d * e;

    f->backward();
    EXPECT_DOUBLE_EQ(d->getData(), 4.0);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 2.0);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 2.0);
    EXPECT_DOUBLE_EQ(e->getGrad(), 4.0);
}

TEST_F(ValueTest, CompoundCalculationDouble)
{
    ValuePtr f = (v2 * 3.0 + 3.0 - 2.0) / 10; // (2 * 3 + 3 - 2) / 10 = 0.7

    f->backward();
    EXPECT_DOUBLE_EQ(f->getData(), 0.7);
    EXPECT_DOUBLE_EQ(v2->getGrad(), 0.3);
}

TEST_F(ValueTest, CompoundCalculationValue)
{
    ValuePtr a = Value::create(2.0);
    ValuePtr b = Value::create(3.0);
    ValuePtr c = Value::create(3.0);
    ValuePtr d = Value::create(2.0);
    ValuePtr e = Value::create(10.0);
    ValuePtr f = (a * b + c - d) / e; // (2 * 3 + 3 - 2) / 10 = 0.7

    f->backward();
    EXPECT_DOUBLE_EQ(f->getData(), 0.7);
    EXPECT_DOUBLE_EQ(a->getGrad(), 0.3);
    EXPECT_DOUBLE_EQ(b->getGrad(), 0.2);
    EXPECT_DOUBLE_EQ(c->getGrad(), 0.1);
    EXPECT_DOUBLE_EQ(d->getGrad(), -0.1);
    EXPECT_DOUBLE_EQ(e->getGrad(), -0.07);
}

TEST_F(ValueTest, Children)
{
    ValuePtr c = v1 * v2;
    std::vector<ValuePtr> children = c->AllChildren();
    EXPECT_EQ(children.size(), 2);
    EXPECT_DOUBLE_EQ(children[0]->getData(), 1.0);
    EXPECT_DOUBLE_EQ(children[1]->getData(), 2.0);
}

TEST_F(ValueTest, TopologicalGradient)
{
    ValuePtr a = v0 * v1;
    ValuePtr b = a * v1;
    ValuePtr result = a + b;

    result->backward();
    EXPECT_DOUBLE_EQ(v0->getGrad(), 2.0);
    EXPECT_DOUBLE_EQ(v1->getGrad(), 0.0);
    EXPECT_DOUBLE_EQ(a->getGrad(), 2.0);
    EXPECT_DOUBLE_EQ(b->getGrad(), 1.0);
    EXPECT_DOUBLE_EQ(result->getData(), 0.0);
}