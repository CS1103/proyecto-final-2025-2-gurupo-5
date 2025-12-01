/**
 * EXAMPLE 4: Predict Single Image
 * 
 * This utility loads a pre-trained model and predicts the class of a single
 * medical image provided by the user as a command-line argument.
 * 
 * Usage: ./build/predict_single <path_to_image>
 */

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "../scripts/stb_image.h" // Ensure this path is correct relative to where you build
// You might need to copy stb_image.h to include/ or adjust the path if scripts/ is not in include path

#include "../src/utec/nn/neural_network.h"
#include "../src/utec/nn/nn_dense.h"
#include "../src/utec/nn/nn_activation.h"
#include "../include/utec/data/medical_mnist_loader.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace utec::neural_network;
using namespace utec::data;

// Define image dimensions expected by the model
const int IMG_WIDTH = 64;
const int IMG_HEIGHT = 64;
const int IMG_CHANNELS = 1; // Grayscale

// Function to load and preprocess a single image
// Returns a Tensor<float, 2> of shape (1, 4096)
bool load_single_image(const string& filepath, Tensor<float, 2>& output_tensor) {
    int width, height, channels;
    
    // Force load as 1 channel (grayscale)
    unsigned char* img_data = stbi_load(filepath.c_str(), &width, &height, &channels, IMG_CHANNELS);
    
    if (!img_data) {
        cerr << "Error: Could not load image '" << filepath << "'" << endl;
        return false;
    }

    // Check dimensions
    if (width != IMG_WIDTH || height != IMG_HEIGHT) {
        // Simple resizing logic (nearest neighbor) or error
        // Ideally, use stbi_resize or similar. For this example, we will error out
        // if not exactly 64x64 to keep it simple, OR we could implement a basic resize.
        // Let's just warn and try to read as much as possible or fail.
        
        // Better approach for robustness: Resize manually if needed.
        // But let's assume the user provides 64x64 for now or warn.
        if (width != 64 || height != 64) {
            cerr << "Error: Image must be 64x64 pixels. Provided: " << width << "x" << height << endl;
            stbi_image_free(img_data);
            return false; 
        }
    }

    // Prepare tensor (1 sample, 4096 features)
    output_tensor = Tensor<float, 2>(1, IMG_WIDTH * IMG_HEIGHT);

    // Normalize pixel values to [0, 1]
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; ++i) {
        output_tensor(0, i) = static_cast<float>(img_data[i]) / 255.0f;
    }

    stbi_image_free(img_data);
    return true;
}

Tensor<float, 2> apply_softmax(const Tensor<float, 2>& logits) {
    size_t batch_size = logits.shape()[0];
    size_t num_classes = logits.shape()[1];
    Tensor<float, 2> probabilities(batch_size, num_classes);

    for (size_t i = 0; i < batch_size; ++i) {
        float max_logit = logits(i, 0);
        for (size_t j = 1; j < num_classes; ++j) {
            max_logit = max(max_logit, logits(i, j));
        }

        float sum_exp = 0.0f;
        for (size_t j = 0; j < num_classes; ++j) {
            probabilities(i, j) = exp(logits(i, j) - max_logit);
            sum_exp += probabilities(i, j);
        }

        for (size_t j = 0; j < num_classes; ++j) {
            probabilities(i, j) /= sum_exp;
        }
    }
    return probabilities;
}

int main() {
    cout << "=== MEDICAL MNIST SINGLE IMAGE PREDICTOR ===\n\n";

    // --- MODIFICA ESTA LÍNEA PARA ELEGIR TU IMAGEN ---
    string image_path = "../data/medical_mnist/CXR/000018.jpeg";

    cout << "Loading image: " << image_path << "...\n";

    Tensor<float, 2> input_tensor;
    if (!load_single_image(image_path, input_tensor)) {
        return 1;
    }
    cout << "Image loaded and preprocessed successfully.\n\n";

    // 2. CREATE NETWORK ARCHITECTURE (Must match training)
    NeuralNetwork<float> network;
    network.add_layer(make_unique<Dense<float>>(4096, 256));
    network.add_layer(make_unique<ReLU<float>>());
    network.add_layer(make_unique<Dense<float>>(256, 128));
    network.add_layer(make_unique<ReLU<float>>());
    network.add_layer(make_unique<Dense<float>>(128, 6));

    // 3. LOAD TRAINED MODEL
    cout << "Loading trained model...\n";
    // Try to find the model in a few likely locations relative to build dir
    vector<string> model_paths = {
        "../models/medical_mnist_model",
        "models/medical_mnist_model",
        "../../models/medical_mnist_model"
    };

    bool loaded = false;
    for (const auto& path : model_paths) {
        if (network.load_model(path)) {
            cout << "Model loaded from: " << path << "\n";
            loaded = true;
            break;
        }
    }

    if (!loaded) {
        cerr << "Error: Could not find trained model directory 'medical_mnist_model'.\n";
        cerr << "Please ensure you have trained the model first using 'train_medical_mnist'.\n";
        return 1;
    }

    // 4. PREDICT
    cout << "\nAnalyzing image...\n";
    auto logits = network.predict(input_tensor);
    auto probabilities = apply_softmax(logits);

    // 5. DISPLAY RESULTS
    // Helper to get class names (using Loader purely for the static map if available, or hardcoded)
    // Since Loader requires CSV paths to init, we'll just manually map or use a dummy loader if needed.
    // Actually, let's just hardcode the classes as per the dataset definition to avoid CSV dependency here.
    const vector<string> class_names = {
        "AbdomenCT", "BreastMRI", "ChestCT", "CXR", "Hand", "HeadCT"
    };

    int best_class = -1;
    float best_prob = -1.0f;

    cout << "\nPrediction Results:\n";
    cout << "-------------------\n";
    for (size_t i = 0; i < 6; ++i) {
        float prob = probabilities(0, i);
        cout << setw(12) << class_names[i] << ": " 
             << fixed << setprecision(2) << (prob * 100.0f) << "%\n";
        
        if (prob > best_prob) {
            best_prob = prob;
            best_class = i;
        }
    }
    cout << "-------------------\n";
    cout << "\nFINAL DIAGNOSIS: " << class_names[best_class] 
         << " (" << (best_prob * 100.0f) << "% confidence)\n";

    return 0;
}
