#include "../include/utec/data/medical_mnist_loader.h"
#include "../src/utec/nn/tensor.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <fstream>

using namespace utec::data;
using namespace utec::algebra;

class DataLoaderTestSuite {
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
        std::cout << "=== DATA LOADER TEST SUITE ===\n\n";

        test_class_name_mapping();
        test_csv_parsing();
        test_normalization();
        test_one_hot_encoding();

        print_summary();
    }

private:
    void test_class_name_mapping() {
        std::cout << "Test 1: Class Name Mapping\n";

        MedicalMNISTLoader loader("dummy_train.csv", "dummy_test.csv");

        assert_test(loader.get_class_name(0) == "AbdomenCT", "Class 0 name");
        assert_test(loader.get_class_name(1) == "BreastMRI", "Class 1 name");
        assert_test(loader.get_class_name(2) == "ChestCT", "Class 2 name");
        assert_test(loader.get_class_name(3) == "CXR", "Class 3 name");
        assert_test(loader.get_class_name(4) == "Hand", "Class 4 name");
        assert_test(loader.get_class_name(5) == "HeadCT", "Class 5 name");

        std::cout << "\n";
    }

    void test_csv_parsing() {
        std::cout << "Test 2: CSV Parsing (Mock Data)\n";

        // Create a small mock CSV file
        std::string test_csv = "test_small.csv";
        std::ofstream out(test_csv);

        // Write 3 samples: label, 4096 pixel values
        out << "0";
        for (int i = 0; i < 4096; ++i) out << ",100";
        out << "\n";

        out << "1";
        for (int i = 0; i < 4096; ++i) out << ",200";
        out << "\n";

        out << "2";
        for (int i = 0; i < 4096; ++i) out << ",50";
        out << "\n";

        out.close();

        // Load the mock CSV
        Tensor<float, 2> images;
        Tensor<float, 2> labels;

        MedicalMNISTLoader loader("dummy.csv", test_csv);
        size_t count = loader.load_test(images, labels, true, 3);

        assert_test(count == 3, "Loaded 3 samples");
        assert_test(images.shape()[0] == 3 && images.shape()[1] == 4096, "Images shape correct");
        assert_test(labels.shape()[0] == 3 && labels.shape()[1] == 6, "Labels shape correct");

        // Remove test file
        std::remove(test_csv.c_str());

        std::cout << "\n";
    }

    void test_normalization() {
        std::cout << "Test 3: Pixel Normalization\n";

        std::string test_csv = "test_norm.csv";
        std::ofstream out(test_csv);

        // Sample with known pixel values
        out << "0";
        for (int i = 0; i < 4096; ++i) {
            out << "," << (i % 256); // Values from 0 to 255
        }
        out << "\n";

        out.close();

        Tensor<float, 2> images;
        Tensor<float, 2> labels;

        MedicalMNISTLoader loader("dummy.csv", test_csv);
        loader.load_test(images, labels, true, 1); // Normalize = true

        // Check normalization: pixel 0 should be 0.0, pixel 255 should be ~1.0
        assert_test(std::abs(images(0, 0) - 0.0f) < 0.01f, "Pixel 0 normalized to 0.0");
        assert_test(std::abs(images(0, 255) - 1.0f) < 0.01f, "Pixel 255 normalized to 1.0");

        std::remove(test_csv.c_str());

        std::cout << "\n";
    }

    void test_one_hot_encoding() {
        std::cout << "Test 4: One-Hot Encoding\n";

        std::string test_csv = "test_onehot.csv";
        std::ofstream out(test_csv);

        // 3 samples with classes 0, 3, 5
        for (int cls : {0, 3, 5}) {
            out << cls;
            for (int i = 0; i < 4096; ++i) out << ",128";
            out << "\n";
        }

        out.close();

        Tensor<float, 2> images;
        Tensor<float, 2> labels;

        MedicalMNISTLoader loader("dummy.csv", test_csv);
        loader.load_test(images, labels, true, 3);

        // Check one-hot encoding
        assert_test(labels(0, 0) == 1.0f && labels(0, 1) == 0.0f, "Sample 0 class 0 one-hot");
        assert_test(labels(1, 3) == 1.0f && labels(1, 0) == 0.0f, "Sample 1 class 3 one-hot");
        assert_test(labels(2, 5) == 1.0f && labels(2, 2) == 0.0f, "Sample 2 class 5 one-hot");

        std::remove(test_csv.c_str());

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
    DataLoaderTestSuite suite;
    suite.run_all_tests();
    return 0;
}
