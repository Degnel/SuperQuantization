#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

template<typename T>
vector<vector<T>> generate_matrix(int rows, int cols, bool normal = false) {
    random_device rd;
    mt19937 gen(rd());

    if constexpr (is_same_v<T, bool>) {
        bernoulli_distribution dist(0.5);
        vector<vector<T>> matrix(rows, vector<T>(cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = dist(gen);
        return matrix;
    } else {
        normal_distribution<T> dist(0.0, 1.0);
        vector<vector<T>> matrix(rows, vector<T>(cols));
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = dist(gen);
        return matrix;
    }
}

template<typename T1, typename T2>
vector<vector<decltype(T1() * T2())>> multiply_matrices(
    const vector<vector<T1>>& A, const vector<vector<T2>>& B) {

    int rows = A.size(), cols = B[0].size(), common_dim = B.size();
    using TResult = decltype(T1() * T2());
    vector<vector<TResult>> result(rows, vector<TResult>(cols, 0));

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            for (int k = 0; k < common_dim; k++)
                result[i][j] += A[i][k] * B[k][j];

    return result;
}

vector<vector<float>> multiply_bool_float(const vector<vector<bool>>& A, const vector<vector<float>>& B) {
    int rows = A.size(), cols = B[0].size(), common_dim = B.size();
    vector<vector<float>> result(rows, vector<float>(cols, 0));

    for (int i = 0; i < rows; i++)
        for (int k = 0; k < common_dim; k++)
            if (A[i][k])
                for (int j = 0; j < cols; j++)
                    result[i][j] += B[k][j];

    return result;
}

int main() {
    int size = 1000;

    vector<vector<float>> A = generate_matrix<float>(size, size, true);
    vector<vector<float>> B = generate_matrix<float>(size, size, true);
    vector<vector<bool>> B_bool = generate_matrix<bool>(size, size);

    auto start = high_resolution_clock::now();
    auto C1 = multiply_matrices(A, B);
    auto stop = high_resolution_clock::now();
    int fxf_time = duration_cast<milliseconds>(stop - start).count();
    cout << "float x float mm time: " << fxf_time << " ms" << endl;

    start = high_resolution_clock::now();
    auto C2 = multiply_bool_float(B_bool, B);
    stop = high_resolution_clock::now();
    int bxf_time = duration_cast<milliseconds>(stop - start).count();
    cout << "bool x float mm time: " << bxf_time << " ms" << endl;

    cout << "Speed up ratio: " << static_cast<float>(fxf_time) / bxf_time << endl;
    return 0;
}