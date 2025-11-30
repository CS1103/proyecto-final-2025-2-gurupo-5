#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

using namespace std;

// Simple image reading using STB (header-only library)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = filesystem;

struct ImageData {
    vector<int> pixels;     // 64x64 grayscale pixels (0-255)
    int label;              // Class label (0-5)
    string class_name;      // Nombre de la clase

    ImageData() : label(-1) {}

    explicit ImageData(int label_, const string& name) : label(label_), class_name(name), pixels(64 * 64, 0) {}
};

// Class mapping
const vector<string> CLASS_NAMES = {
    "AbdomenCT",   // 0
    "BreastMRI",   // 1
    "ChestCT",     // 2
    "CXR",         // 3 (Chest X-Ray)
    "Hand",        // 4
    "HeadCT"       // 5
};

// Load single image (64x64 grayscale)
ImageData load_image(const string& filepath, int label, const string& class_name) {
    ImageData img(label, class_name);

    int width, height, channels;
    // Cargar imagen forzando 1 canal (grayscale)
    unsigned char* data = stbi_load(filepath.c_str(), &width, &height, &channels, 1);

    if (!data) {
        cerr << "Failed to load image: " << filepath << endl;
        return img;  // Retorna imagen vacía
    }

    // Verificar dimensiones (todas las imágenes Medical MNIST son 64x64)
    if (width != 64 || height != 64) {
        cerr << "Warning: Image " << filepath << " is " << width << "x" << height
             << " (expected 64x64)" << endl;
        stbi_image_free(data);
        return img;
    }

    // Convertir de unsigned char[4096] a vector<int> con valores 0-255
    for (int i = 0; i < 64 * 64; ++i) {
        img.pixels[i] = static_cast<int>(data[i]);
    }

    stbi_image_free(data);
    return img;
}

// Write images to CSV format: label,p0,p1,...,p4095
void write_csv(const string& output_path, const vector<ImageData>& images) {
    ofstream file(output_path);
    if (!file.is_open()) {
        cerr << "Failed to open output file: " << output_path << endl;
        return;
    }

    cout << "Writing " << images.size() << " images to " << output_path << "..." << endl;

    for (const auto& img : images) {
        file << img.label;
        for (int pixel : img.pixels) {
            file << "," << pixel;  // Ya son int (0-255)
        }
        file << "\n";
    }

    file.close();
    cout << "Done!" << endl;
}

int main() {
    cout << "=== MEDICAL MNIST: JPEG to CSV Converter ===\n\n";

    const string input_dir = "../data/medical_mnist";
    const string output_dir = "../data/processed";

    // Create output directory if it doesn't exist
    fs::create_directories(output_dir);

    vector<ImageData> all_images;

    // Load images from each class
    for (size_t class_idx = 0; class_idx < CLASS_NAMES.size(); ++class_idx) {
        const string& class_name = CLASS_NAMES[class_idx];
        string class_dir = input_dir + "/" + class_name;

        if (!fs::exists(class_dir)) {
            cout << "Warning: Directory not found: " << class_dir << endl;
            continue;
        }

        cout << "Loading class " << class_idx << ": " << class_name << "..." << endl;

        int count = 0;
        for (const auto& entry : fs::directory_iterator(class_dir)) {
            if (entry.is_regular_file()) {
                string ext = entry.path().extension().string();
                transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                    ImageData img = load_image(entry.path().string(), class_idx, class_name);
                    if (!img.pixels.empty()) {
                        all_images.push_back(img);
                        count++;
                    }
                }
            }
        }

        cout << "  Loaded " << count << " images from " << class_name << endl;
    }

    cout << "\nTotal images loaded: " << all_images.size() << endl;

    if (all_images.empty()) {
        cerr << "Error: No images were loaded. Check the directory structure." << endl;
        return 1;
    }

    // Shuffle images
    random_device rd;
    mt19937 gen(42); // Fixed seed for reproducibility
    shuffle(all_images.begin(), all_images.end(), gen);

    // Split 80% train, 20% test
    size_t train_size = static_cast<size_t>(all_images.size() * 0.8);

    vector<ImageData> train_images(all_images.begin(), all_images.begin() + train_size);
    vector<ImageData> test_images(all_images.begin() + train_size, all_images.end());

    cout << "\nTrain images: " << train_images.size() << endl;
    cout << "Test images: " << test_images.size() << endl;

    // Write CSV files
    write_csv(output_dir + "/train.csv", train_images);
    write_csv(output_dir + "/test.csv", test_images);

    // Write metadata
    ofstream meta(output_dir + "/metadata.txt");
    meta << "Medical MNIST Dataset\n";
    meta << "=====================\n\n";
    meta << "Total images: " << all_images.size() << "\n";
    meta << "Train images: " << train_images.size() << "\n";
    meta << "Test images: " << test_images.size() << "\n\n";
    meta << "Classes:\n";
    for (size_t i = 0; i < CLASS_NAMES.size(); ++i) {
        meta << "  " << i << ": " << CLASS_NAMES[i] << "\n";
    }
    meta.close();

    cout << "\n Conversion complete!" << endl;
    cout << "  - train.csv: " << output_dir << "/train.csv" << endl;
    cout << "  - test.csv: " << output_dir << "/test.csv" << endl;

    return 0;
}
