//
// Created by rudri on 10/11/2020.
//

#ifndef PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
#define PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H

#include <cstddef>
#include <utility>
#include <array>
#include <sstream>
#include <stdexcept>
#include <initializer_list>
#include <type_traits>
#include <vector>
#include <iostream>
#include <functional>
#include <numeric>
using namespace std;

namespace utec::algebra {
    // Clase Tensor para arrays multidimensionales
    template<typename T, size_t Rank>
    class Tensor {
    private:
        array<size_t, Rank> shape_;  // Dimensiones del tensor
        vector<T> data_;             // Datos almacenados en forma lineal

        // Valida que el número de dimensiones introducidas en el constructor coincida con Rank
        void check_rank(const vector<size_t>& dimensions) {
            if (dimensions.size() != Rank) {
                ostringstream oss;
                oss << "Number of dimensions do not match with " << Rank;
                throw invalid_argument(oss.str());
            }
        }

        // Calcula el número total de elementos del tensor, se usa para hacer resize al vector data_ luego de crear el tensor
        size_t count_elements() const {
            if (shape_.empty())
                return 0;
            // calculamos el total= dim_1 * dim_2 * ... * dim_n
            size_t total = 1;
            for (const auto& dimension : shape_) {
                total *= dimension;
            }
            return total;
        }

        // Convierte índices multidimensionales a índice lineal
        size_t linearize(const array<size_t, Rank>& indices) const {
            size_t flat_idx = 0;
            size_t step = 1;
            for (int i = Rank - 1; i >= 0; i--) {
                flat_idx += indices[i] * step;
                step *= shape_[i];
            }
            return flat_idx;
        }

        // Imprime recursivamente el tensor con formato anidado
        template <typename OStream>
        void print_recursive(OStream& os, array<size_t, Rank> indices, size_t depth) const {
            if (depth == Rank - 1) {
                for (size_t i = 0; i < shape_[depth]; i++) {
                    indices[depth] = i;
                    os << data_[linearize(indices)];
                    if (i < shape_[depth] - 1) {
                        os << " ";
                    }
                }
                return;
            }
            os << "{" << "\n";
            for (size_t i = 0; i < shape_[depth]; i++) {
                indices[depth] = i;
                print_recursive(os, indices, depth + 1);
                if (i < shape_[depth] - 1) {
                    os << "\n";
                }
            }
            os << "\n" << "}";
        }

        // Verifica si dos shapes son compatibles para broadcasting
        bool can_broadcast(const array<size_t, Rank>& s1,
                              const array<size_t, Rank>& s2) const {
            for (int i = Rank - 1; i >= 0; i--) {
                if (s1[i] != s2[i] && s1[i] != 1 && s2[i] != 1) {
                    return false;
                }
            }
            return true;
        }

        // Ajusta índice para broadcasting (dimensiones de tamaño 1 se quedan en 0)
        array<size_t, Rank> broadcast_idx(
            const array<size_t, Rank>& target,
            const array<size_t, Rank>& tensor_shape) const {
            array<size_t, Rank> adjusted;
            for (size_t i = 0; i < Rank; i++) {
                if (tensor_shape[i] == 1) {
                    adjusted[i] = 0;
                } else {
                    adjusted[i] = target[i];
                }
            }
            return adjusted;
        }

        // Calcula el shape resultante después de broadcasting
        array<size_t, Rank> merged_shape(
            const array<size_t, Rank>& s1,
            const array<size_t, Rank>& s2) const {
            array<size_t, Rank> output;
            for (size_t i = 0; i < Rank; i++) {
                output[i] = max(s1[i], s2[i]);
            }
            return output;
        }

    public:
        using iterator = typename vector<T>::iterator;
        using const_iterator = typename vector<T>::const_iterator;
        // Constructor por defecto
        Tensor() = default;

        // Constructor con dimensiones variádicas (solo tipos aritméticos)
        template<typename... Dims>
        Tensor(Dims&&... dims)
            requires (sizeof...(dims) > 0 && (is_arithmetic_v<decay_t<Dims>> && ...))
        {
            vector<size_t> dimensions = { static_cast<size_t>(forward<Dims>(dims))... };
            check_rank(dimensions);
            copy(dimensions.begin(), dimensions.begin() + Rank, shape_.begin());
            data_.resize(count_elements());
        }

        // Constructor con initializer_list de dimensiones
        Tensor(initializer_list<size_t> dims) {
            vector<size_t> dimension_vector(dims.begin(), dims.end());
            check_rank(dimension_vector);
            copy(dimension_vector.begin(), dimension_vector.begin() + Rank, shape_.begin());
            data_.resize(count_elements());
        }

        // Constructor explícito con array de dimensiones
        explicit Tensor(const array<size_t, Rank>& dims) {
            shape_ = dims;
            data_.resize(count_elements());
        }

        // Asigna datos desde initializer_list
        Tensor& operator=(initializer_list<T> values) {
            if (values.size() != count_elements()) {
                throw invalid_argument("Data size does not match tensor size");
            }
            data_.assign(values.begin(), values.end());
            return *this;
        }

        // Acceso a elementos con índices multidimensionales
        template<typename... Idxs>
        T& operator()(Idxs... indices) {
            static_assert(sizeof...(indices) == Rank, "Wrong number of arguments passed");
            array<size_t, Rank> idx_array = { static_cast<size_t>(indices)... };
            for (size_t i = 0; i < Rank; i++) {
                if (idx_array[i] >= shape_[i]) {
                    ostringstream oss;
                    oss << "Index out of bounds for dimension " << i
                        << ": " << idx_array[i] << " >= " << shape_[i];
                    throw out_of_range(oss.str());
                }
            }
            return data_[linearize(idx_array)];
        }

        // the same dddddddddd
        template<typename... Args>
        const T& operator()(Args... indices) const {

            array<size_t, Rank> idx_array = { static_cast<size_t>(indices)... };
            return data_[linearize(idx_array)];
        }

        // Cambia las dimensiones del tensor
        template<typename... Dims>
        void reshape(Dims&&... dims) {
            vector<size_t> updated_dims = { static_cast<size_t>(forward<Dims>(dims))... };
            check_rank(updated_dims);
            size_t total_size = 1;
            for (const auto& d : updated_dims) {
                total_size *= d;
            }
            if (total_size > data_.size()) {
                data_.resize(total_size, T());
            }
            copy(updated_dims.begin(), updated_dims.begin() + Rank, shape_.begin());
        }

        // Retorna el número total de elementos
        size_t size() const {
            return data_.size();
        }


        // Acceso directo al vector de datos por índice lineal
        T& operator[](size_t idx) {
            return data_[idx];
        }

        const T& operator[](size_t idx) const {
            return data_[idx];
        }

        // Retorna el shape del tensor
        array<size_t, Rank> shape() const {
            return shape_;
        }

        // Retorna una referencia al vector de datos
        vector<T>& get_data() {
            return data_;
        }

        // Retorna una referencia constante al vector de datos
        const vector<T>& get_data() const {
            return data_;
        }

        // Llena el tensor con un valor
        void fill(const T& value) {
            std::fill(data_.begin(), data_.end(), value);
        }

        // Iteradores
        iterator begin() { return data_.begin(); }
        iterator end() { return data_.end(); }
        const_iterator begin() const { return data_.begin(); }
        const_iterator end() const { return data_.end(); }
        const_iterator cbegin() const { return data_.cbegin(); }
        const_iterator cend() const { return data_.cend(); }


























        // Operador de salida para imprimir el tensor
        friend ostream& operator<<(ostream& os, const Tensor<T, Rank>& tensor) {
            if (tensor.data_.empty()) {
                os << "{}";
                return os;
            }
            array<size_t, Rank> indices{};
            tensor.print_recursive(os, indices, 0);
            return os;
        }


        // Operadores de comparación
        bool operator==(const Tensor<T, Rank>& other) const {
            if (shape_ != other.shape_) {
                return false;
            }
            return data_ == other.data_;
        }

        bool operator!=(const Tensor<T, Rank>& other) const {
            return !(*this == other);
        }

        // Operaciones elemento a elemento entre tensores (con broadcasting)

        // Suma tensor-tensor
        Tensor<T, Rank> operator+(const Tensor<T, Rank>& other) const {
            if (!can_broadcast(shape_, other.shape_)) {
                throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }
            auto output_shape = merged_shape(shape_, other.shape_);
            Tensor<T, Rank> result(output_shape);
            array<size_t, Rank> position{};
            const size_t total = accumulate(output_shape.begin(), output_shape.end(), (size_t)1, multiplies<size_t>());
            for (size_t pos = 0; pos < total; pos++) {
                size_t temp = pos;
                for (int dimension = Rank - 1; dimension >= 0; dimension--) {
                    position[dimension] = temp % output_shape[dimension];
                    temp /= output_shape[dimension];
                }
                auto idx1 = broadcast_idx(position, shape_);
                auto idx2 = broadcast_idx(position, other.shape_);
                result.data_[pos] = data_[linearize(idx1)] + other.data_[other.linearize(idx2)];
            }
            return result;
        }

        // Resta tensor-tensor
        Tensor<T, Rank> operator-(const Tensor<T, Rank>& other) const {
            if (!can_broadcast(shape_, other.shape_)) {
                throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }
            auto output_shape = merged_shape(shape_, other.shape_);
            Tensor<T, Rank> result(output_shape);
            array<size_t, Rank> position{};
            const size_t total = accumulate(output_shape.begin(), output_shape.end(), (size_t)1, multiplies<size_t>());
            for (size_t pos = 0; pos < total; pos++) {
                size_t temp = pos;
                for (int dimension = Rank - 1; dimension >= 0; dimension--) {
                    position[dimension] = temp % output_shape[dimension];
                    temp /= output_shape[dimension];
                }
                auto idx1 = broadcast_idx(position, shape_);
                auto idx2 = broadcast_idx(position, other.shape_);
                result.data_[pos] = data_[linearize(idx1)] - other.data_[other.linearize(idx2)];
            }
            return result;
        }

        // Multiplicación tensor-tensor
        Tensor<T, Rank> operator*(const Tensor<T, Rank>& other) const {
            if (!can_broadcast(shape_, other.shape_)) {
                throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }
            auto output_shape = merged_shape(shape_, other.shape_);
            Tensor<T, Rank> result(output_shape);
            array<size_t, Rank> position{};
            const size_t total = accumulate(output_shape.begin(), output_shape.end(), (size_t)1, multiplies<size_t>());
            for (size_t pos = 0; pos < total; pos++) {
                size_t temp = pos;
                for (int dimension = Rank - 1; dimension >= 0; dimension--) {
                    position[dimension] = temp % output_shape[dimension];
                    temp /= output_shape[dimension];
                }
                auto idx1 = broadcast_idx(position, shape_);
                auto idx2 = broadcast_idx(position, other.shape_);
                result.data_[pos] = data_[linearize(idx1)] * other.data_[other.linearize(idx2)];
            }
            return result;
        }


        // División tensor-tensor
        Tensor<T, Rank> operator/(const Tensor<T, Rank>& other) const {
            if (!can_broadcast(shape_, other.shape_)) {
                throw invalid_argument("Shapes do not match and they are not compatible for broadcasting");
            }
            auto output_shape = merged_shape(shape_, other.shape_);
            Tensor<T, Rank> result(output_shape);
            array<size_t, Rank> position{};
            const size_t total = accumulate(output_shape.begin(), output_shape.end(), (size_t)1, multiplies<size_t>());
            for (size_t pos = 0; pos < total; pos++) {
                size_t temp = pos;
                for (int dimension = Rank - 1; dimension >= 0; dimension--) {
                    position[dimension] = temp % output_shape[dimension];
                    temp /= output_shape[dimension];
                }
                auto idx1 = broadcast_idx(position, shape_);
                auto idx2 = broadcast_idx(position, other.shape_);
                if (other.data_[other.linearize(idx2)] == T{}) {
                    throw domain_error("Division by zero");
                }
                result.data_[pos] = data_[linearize(idx1)] / other.data_[other.linearize(idx2)];
            }
            return result;
        }

        // Operaciones tensor-escalar
        Tensor<T, Rank> operator+(const T& value) const {
            Tensor<T, Rank> result(*this);
            for (size_t i = 0; i < data_.size(); i++) {
                result.data_[i] += value;
            }
            return result;
        }

        Tensor<T, Rank> operator-(const T& value) const {
            Tensor<T, Rank> result(*this);
            for (size_t i = 0; i < data_.size(); i++) {
                result.data_[i] -= value;
            }
            return result;
        }

        Tensor<T, Rank> operator*(const T& value) const {
            Tensor<T, Rank> result(*this);
            for (size_t i = 0; i < data_.size(); i++) {
                result.data_[i] *= value;
            }
            return result;
        }

        Tensor<T, Rank> operator/(const T& value) const {
            if (value == T{}) {
                throw domain_error("Division by zero");
            }
            Tensor<T, Rank> result(*this);
            for (size_t i = 0; i < data_.size(); i++) {
                result.data_[i] /= value;
            }
            return result;
        }

        // Operaciones escalar-tensor (conmutativas)
        friend Tensor<T, Rank> operator+(const T& value, const Tensor<T, Rank>& t) {
            return t + value;
        }
        friend Tensor<T, Rank> operator*(const T& value, const Tensor<T, Rank>& t) {
            return t * value;
        }

        // Operaciones escalar-tensor (no conmutativas)
        friend Tensor<T, Rank> operator-(const T& value, const Tensor<T, Rank>& t) {
            Tensor<T, Rank> result(t.shape_);
            for (size_t i = 0; i < t.data_.size(); i++) {
                result.data_[i] = value - t.data_[i];
            }
            return result;
        }
        friend Tensor<T, Rank> operator/(const T& value, const Tensor<T, Rank>& t) {
            Tensor<T, Rank> result(t.shape_);
            for (size_t i = 0; i < t.data_.size(); i++) {
                if (t.data_[i] == T{}) {
                    throw domain_error("Division by zero");
                }
                result.data_[i] = value / t.data_[i];
            }
            return result;
        }


        // Transpone las últimas dos dimensiones del tensor
        Tensor<T, Rank> transpose_2d() const {
            if constexpr (Rank < 2) {
                throw invalid_argument("Cannot transpose 1D tensor: need at least 2 dimensions");
            }
            array<size_t, Rank> new_shape = shape_;
            swap(new_shape[Rank-2], new_shape[Rank-1]);
            Tensor<T, Rank> result(new_shape);
            array<size_t, Rank> indices{};
            const size_t total = count_elements();
            for (size_t pos = 0; pos < total; pos++) {
                size_t temp = pos;
                for (int dimension = Rank - 1; dimension >= 0; dimension--) {
                    indices[dimension] = temp % shape_[dimension];
                    temp /= shape_[dimension];
                }
                array<size_t, Rank> transposed_indices = indices;
                swap(transposed_indices[Rank-2], transposed_indices[Rank-1]);
                size_t result_pos = 0;
                size_t step = 1;
                for (int dimension = Rank - 1; dimension >= 0; dimension--) {
                    result_pos += transposed_indices[dimension] * step;
                    step *= new_shape[dimension];
                }
                result.data_[result_pos] = data_[pos];
            }
            return result;
        }


        // Acceso público al cálculo de índice lineal
        size_t get_linear_index(const array<size_t, Rank>& indices) const {
            return linearize(indices);
        }

        template<typename U, size_t M>
        friend Tensor<U, M> matrix_product(const Tensor<U, M>& tensor1, const Tensor<U, M>& tensor2);
    };


    // Función libre para transponer
    template<typename T, size_t Rank>
    Tensor<T, Rank> transpose_2d(const Tensor<T, Rank>& tensor) {
        return tensor.transpose_2d();

    }

    // Función apply: aplica una función a cada elemento del tensor
    template<typename T, size_t Rank, typename Func>
    Tensor<T, Rank> apply(const Tensor<T, Rank>& tensor, Func func) {
        Tensor<T, Rank> result(tensor.shape());
        auto& result_data = result.get_data();
        const auto& tensor_data = tensor.get_data();
        for (size_t i = 0; i < tensor_data.size(); ++i) {
            result_data[i] = func(tensor_data[i]);
        }
        return result;
    }


    // Multiplicación de matrices (opera sobre las últimas dos dimensiones)
    template<typename T, size_t Rank>
    Tensor<T, Rank> matrix_product(const Tensor<T, Rank>& tensor1, const Tensor<T, Rank>& tensor2) {
        static_assert(Rank >= 2, "Tensors must have at least 2 dimensions for matrix multiplication");

        auto shape1 = tensor1.shape();
        auto shape2 = tensor2.shape();

        // Dimensiones de las matrices (últimas dos dimensiones)
        size_t rows1 = shape1[Rank-2];
        size_t cols1 = shape1[Rank-1];
        size_t rows2 = shape2[Rank-2];
        size_t cols2 = shape2[Rank-1];
        bool matrix_compatible = (cols1 == rows2);

        // Verificar que las dimensiones batch coincidan
        bool batch_compatible = true;
        for (size_t i = 0; i < Rank - 2; i++) {
            if (shape1[i] != shape2[i]) {
                batch_compatible = false;
                break;
            }
        }


        if (!matrix_compatible && !batch_compatible) {
            throw invalid_argument("Matrix dimensions are not compatible for multiplication AND Batch dimensions do not match");
        } else if (!matrix_compatible) {
            throw invalid_argument("Matrix dimensions are incompatible for multiplication");
        } else if (!batch_compatible) {
            throw invalid_argument("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
        }


        array<size_t, Rank> result_shape = shape1;
        result_shape[Rank-2] = rows1;
        result_shape[Rank-1] = cols2;

        Tensor<T, Rank> result(result_shape);

        result.fill(T{});

        // Iterar sobre todos los batches
        array<size_t, Rank-2> batch_indices{};
        size_t total_batches = 1;
        for (size_t i = 0; i < Rank - 2; i++) {
            total_batches *= shape1[i];
        }
        for (size_t batch = 0; batch < total_batches; batch++) {
            size_t temp = batch;
            for (int dimension = static_cast<int>(Rank) - 3; dimension >= 0; dimension--) {
                batch_indices[dimension] = temp % shape1[dimension];
                temp /= shape1[dimension];
            }
            // Multiplicación de matrices estándar
            for (size_t i = 0; i < rows1; i++) {
                for (size_t j = 0; j < cols2; j++) {
                    T sum = T{};
                    for (size_t k = 0; k < cols1; k++) {
                        array<size_t, Rank> idx1;
                        for (size_t d = 0; d < Rank - 2; d++) {
                            idx1[d] = batch_indices[d];
                        }
                        idx1[Rank-2] = i;
                        idx1[Rank-1] = k;

                        array<size_t, Rank> idx2;

                        for (size_t d = 0; d < Rank - 2; d++) {

                            idx2[d] = batch_indices[d];

                        }
                        idx2[Rank-2] = k;
                        idx2[Rank-1] = j;
                        sum += tensor1.data_[tensor1.get_linear_index(idx1)] *
                               tensor2.data_[tensor2.get_linear_index(idx2)];
                    }
                    array<size_t, Rank> result_idx;
                    for (size_t d = 0; d < Rank - 2; d++) {
                        result_idx[d] = batch_indices[d];
                    }
                    result_idx[Rank-2] = i;
                    result_idx[Rank-1] = j;
                    result.data_[result.get_linear_index(result_idx)] = sum;
                }
            }
        }
        return result;
    }
}


#endif //PROG3_NN_FINAL_PROJECT_V2025_01_TENSOR_H
