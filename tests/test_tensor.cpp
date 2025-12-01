#include "../src/utec/nn/tensor.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace utec::algebra;

class TensorTestSuite {
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
        std::cout << "=== TENSOR TEST SUITE ===\n\n";

        test_creation_and_fill();
        test_arithmetic_operations();
        test_matrix_multiplication();
        test_transpose();
        test_shape_operations();
        test_broadcasting();
        test_edge_cases();

        print_summary();
    }

private:
    void test_creation_and_fill() {
        std::cout << "Test 1: Creación y fill\n";

        Tensor<float, 2> t(3, 4);
        t.fill(5.0f);

        assert_test(t(0, 0) == 5.0f, "Fill value correct");
        assert_test(t(2, 3) == 5.0f, "Fill all elements");
        assert_test(t.shape()[0] == 3 && t.shape()[1] == 4, "Shape correct");

        std::cout << "\n";
    }

    void test_arithmetic_operations() {
        std::cout << "Test 2: Operaciones aritméticas\n";

        Tensor<double, 2> a(2, 2);
        Tensor<double, 2> b(2, 2);

        a.fill(3.0);
        b.fill(2.0);

        auto sum = a + b;
        auto diff = a - b;
        auto prod = a * b;

        assert_test(sum(0, 0) == 5.0, "Addition");
        assert_test(diff(1, 1) == 1.0, "Subtraction");
        assert_test(prod(0, 1) == 6.0, "Element-wise multiplication");

        // Scalar operations
        auto scaled = a * 2.0;
        assert_test(scaled(0, 0) == 6.0, "Scalar multiplication");

        std::cout << "\n";
    }

    void test_matrix_multiplication() {
        std::cout << "Test 3: Multiplicación de matrices\n";

        Tensor<float, 2> m1(2, 3);
        Tensor<float, 2> m2(3, 2);

        // m1 = [[1, 2, 3], [4, 5, 6]]
        for (size_t i = 0; i < 2; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                m1(i, j) = i * 3 + j + 1;
            }
        }

        m2.fill(1.0f);

        auto result = matrix_product(m1, m2);

        assert_test(result.shape()[0] == 2 && result.shape()[1] == 2, "Result shape correct");
        assert_test(result(0, 0) == 6.0f, "Matrix product value [0,0]");
        assert_test(result(1, 0) == 15.0f, "Matrix product value [1,0]");

        std::cout << "\n";
    }

    void test_transpose() {
        std::cout << "Test 4: Transpose 2D\n";

        Tensor<int, 2> m(2, 3);

        // m = [[1, 2, 3], [4, 5, 6]]
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                m(i, j) = i * 3 + j + 1;
            }
        }

        auto mt = m.transpose_2d();

        assert_test(mt.shape()[0] == 3 && mt.shape()[1] == 2, "Transpose shape");
        assert_test(mt(0, 0) == 1 && mt(0, 1) == 4, "Transpose values row 0");
        assert_test(mt(2, 0) == 3 && mt(2, 1) == 6, "Transpose values row 2");

        std::cout << "\n";
    }

    void test_shape_operations() {
        std::cout << "Test 5: Operaciones de shape\n";

        Tensor<float, 2> t(2, 6);
        for (size_t i = 0; i < 12; ++i) {
            t[i] = i;
        }

        t.reshape(3, 4);
        assert_test(t.shape()[0] == 3 && t.shape()[1] == 4, "Reshape successful");
        assert_test(t[11] == 11, "Data preserved after reshape");

        // Test reshape to smaller size
        t.reshape(2, 5);
        assert_test(t.shape()[0] == 2 && t.shape()[1] == 5, "Reshape to smaller size");

        // Test reshape to larger size (should resize)
        t.reshape(4, 4);
        assert_test(t.shape()[0] == 4 && t.shape()[1] == 4, "Reshape to larger size");

        std::cout << "\n";
    }

    void test_broadcasting() {
        std::cout << "Test 6: Broadcasting\n";

        Tensor<float, 2> a(2, 1);
        Tensor<float, 2> b(2, 3);

        a(0, 0) = 2.0f;
        a(1, 0) = 3.0f;
        b.fill(5.0f);

        auto result = a * b;

        assert_test(result(0, 0) == 10.0f, "Broadcast multiply [0,0]");
        assert_test(result(0, 2) == 10.0f, "Broadcast multiply [0,2]");
        assert_test(result(1, 1) == 15.0f, "Broadcast multiply [1,1]");

        std::cout << "\n";
    }

    void test_edge_cases() {
        std::cout << "Test 7: Casos extremos\n";

        // Single element tensor
        Tensor<double, 1> single(1);
        single(0) = 42.0;
        auto scaled = single * 2.0;
        assert_test(scaled(0) == 84.0, "Single element tensor");

        // Zero tensor
        Tensor<float, 2> zeros(3, 3);
        zeros.fill(0.0f);
        auto sum = zeros + zeros;
        assert_test(sum(1, 1) == 0.0f, "Zero tensor operations");

        // Large tensor
        Tensor<int, 2> large(100, 100);
        large.fill(1);
        assert_test(large(99, 99) == 1, "Large tensor creation");

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
    TensorTestSuite suite;
    suite.run_all_tests();
    return 0;
}
