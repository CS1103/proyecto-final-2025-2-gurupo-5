#include "../src/utec/nn/neural_network.h"
#include "../src/utec/nn/nn_dense.h"
#include "../src/utec/nn/nn_activation.h"
#include "../src/utec/nn/nn_loss.h"
#include "../src/utec/nn/nn_optimizer.h"
#include "../src/utec/nn/tensor.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <memory>

using namespace utec::neural_network;

class NeuralNetworkTestSuite {
private:
    size_t tests_passed = 0;
    size_t tests_total = 0;

    void assert_test(bool condition, const std::string& test_name) {
        tests_total++;
        if (condition) {
            tests_passed++;
            std::cout << "  [PASS] " << test_name << "\n";
        } else {
            std::cout << "  [FAIL] " << test_name << "\n";
        }
    }

public:
    void run_all_tests() {
        std::cout << "=== NEURAL NETWORK TEST SUITE ===\n\n";

        test_relu_activation();
        test_sigmoid_activation();
        test_dense_layer();
        test_mse_loss();
        test_softmax_cross_entropy_loss();
        test_sgd_optimizer();
        test_xor_training();

        print_summary();
    }

private:
    void test_relu_activation() {
        std::cout << "Test 1: ReLU Activation\n";

        Tensor<float, 2> input(2, 2);
        input(0, 0) = -1.0f;
        input(0, 1) = 2.0f;
        input(1, 0) = 0.0f;
        input(1, 1) = -3.0f;

        ReLU<float> relu;
        auto output = relu.forward(input);

        assert_test(output(0, 0) == 0.0f, "ReLU negative to zero");
        assert_test(output(0, 1) == 2.0f, "ReLU positive unchanged");
        assert_test(output(1, 0) == 0.0f, "ReLU zero unchanged");

        // Test backward
        Tensor<float, 2> grad(2, 2);
        grad.fill(1.0f);
        auto input_grad = relu.backward(grad);

        assert_test(input_grad(0, 0) == 0.0f, "ReLU gradient for negative input");
        assert_test(input_grad(0, 1) == 1.0f, "ReLU gradient for positive input");

        std::cout << "\n";
    }

    void test_sigmoid_activation() {
        std::cout << "Test 2: Sigmoid Activation\n";

        Tensor<float, 2> input(1, 2);
        input(0, 0) = 0.0f;
        input(0, 1) = 2.0f;

        Sigmoid<float> sigmoid;
        auto output = sigmoid.forward(input);

        assert_test(std::abs(output(0, 0) - 0.5f) < 0.01f, "Sigmoid at zero");
        assert_test(output(0, 1) > 0.8f && output(0, 1) < 0.9f, "Sigmoid at 2");

        std::cout << "\n";
    }

    void test_dense_layer() {
        std::cout << "Test 3: Dense Layer\n";

        Dense<float> dense(3, 2);

        Tensor<float, 2> input(1, 3);
        input(0, 0) = 1.0f;
        input(0, 1) = 2.0f;
        input(0, 2) = 3.0f;

        auto output = dense.forward(input);

        assert_test(output.shape()[0] == 1 && output.shape()[1] == 2, "Dense output shape");

        // Test backward
        Tensor<float, 2> grad(1, 2);
        grad.fill(1.0f);
        auto input_grad = dense.backward(grad);

        assert_test(input_grad.shape()[0] == 1 && input_grad.shape()[1] == 3, "Dense backward shape");

        std::cout << "\n";
    }

    void test_mse_loss() {
        std::cout << "Test 4: MSE Loss\n";

        Tensor<float, 2> prediction(2, 1);
        Tensor<float, 2> target(2, 1);

        prediction(0, 0) = 1.0f;
        prediction(1, 0) = 2.0f;

        target(0, 0) = 0.0f;
        target(1, 0) = 4.0f;

        MSELoss<float> loss(prediction, target);
        float loss_value = loss.loss();

        // Expected: ((1-0)^2 + (2-4)^2) / 2 = (1 + 4) / 2 = 2.5
        assert_test(std::abs(loss_value - 2.5f) < 0.01f, "MSE loss calculation");

        auto grad = loss.loss_gradient();
        assert_test(grad.shape()[0] == 2 && grad.shape()[1] == 1, "MSE gradient shape");

        std::cout << "\n";
    }

    void test_softmax_cross_entropy_loss() {
        std::cout << "Test 5: Softmax Cross Entropy Loss\n";

        Tensor<float, 2> logits(1, 3);
        Tensor<float, 2> labels(1, 3);

        logits(0, 0) = 1.0f;
        logits(0, 1) = 2.0f;
        logits(0, 2) = 0.5f;

        labels.fill(0.0f);
        labels(0, 1) = 1.0f; // One-hot: class 1

        SoftmaxCrossEntropyLoss<float> loss(logits, labels);
        float loss_value = loss.loss();

        assert_test(loss_value > 0.0f, "Softmax CE loss positive");

        auto grad = loss.loss_gradient();
        assert_test(grad.shape()[0] == 1 && grad.shape()[1] == 3, "Softmax CE gradient shape");

        std::cout << "\n";
    }

    void test_sgd_optimizer() {
        std::cout << "Test 6: SGD Optimizer\n";

        Tensor<float, 2> weights(2, 2);
        Tensor<float, 2> gradients(2, 2);

        weights.fill(1.0f);
        gradients.fill(0.1f);

        SGD<float> optimizer(0.1f);
        optimizer.update(weights, gradients);

        // Expected: 1.0 - 0.1 * 0.1 = 0.99
        assert_test(std::abs(weights(0, 0) - 0.99f) < 0.001f, "SGD weight update");

        std::cout << "\n";
    }

    void test_xor_training() {
        std::cout << "Test 7: XOR Training (Convergence Test)\n";

        // XOR dataset
        Tensor<float, 2> X(4, 2);
        X(0, 0) = 0.0f; X(0, 1) = 0.0f;
        X(1, 0) = 0.0f; X(1, 1) = 1.0f;
        X(2, 0) = 1.0f; X(2, 1) = 0.0f;
        X(3, 0) = 1.0f; X(3, 1) = 1.0f;

        Tensor<float, 2> Y(4, 1);
        Y(0, 0) = 0.0f;
        Y(1, 0) = 1.0f;
        Y(2, 0) = 1.0f;
        Y(3, 0) = 0.0f;

        // Simple network: 2 -> 4 -> 1
        NeuralNetwork<float> network;
        network.add_layer(std::make_unique<Dense<float>>(2, 4));
        network.add_layer(std::make_unique<ReLU<float>>());
        network.add_layer(std::make_unique<Dense<float>>(4, 1));
        network.add_layer(std::make_unique<Sigmoid<float>>());

        // Train for 1000 epochs with higher learning rate
        network.train<MSELoss, SGD>(X, Y, 1000, 4, 0.5f, "", 999); // High patience

        // Test predictions
        auto predictions = network.predict(X);

        // Check approximate correctness (threshold 0.5)
        bool xor_correct = true;
        for (size_t i = 0; i < 4; ++i) {
            float pred = predictions(i, 0);
            float target = Y(i, 0);
            bool correct = (pred < 0.5f && target < 0.5f) || (pred >= 0.5f && target >= 0.5f);
            if (!correct) {
                xor_correct = false;
                break;
            }
        }

        assert_test(xor_correct, "XOR problem solved (approximate)");

        std::cout << "\n";
    }

    void print_summary() {
        std::cout << "\n=== TEST SUMMARY ===\n";
        std::cout << "Tests passed: " << tests_passed << "/" << tests_total;
        double percentage = (static_cast<double>(tests_passed) / tests_total) * 100.0;
        std::cout << " (" << std::fixed << std::setprecision(1) << percentage << "%)\n";

        if (tests_passed == tests_total) {
            std::cout << "ALL TESTS PASSED!\n";
        } else {
            std::cout << "Some tests failed.\n";
        }
        std::cout << "====================\n";
    }
};

int main() {
    NeuralNetworkTestSuite suite;
    suite.run_all_tests();
    return 0;
}
