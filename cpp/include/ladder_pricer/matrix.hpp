#pragma once

#include <cstddef>
#include <vector>

namespace ladder_pricer {

class DenseMatrix {
public:
    DenseMatrix() = default;
    DenseMatrix(std::size_t rows, std::size_t cols, double value = 0.0);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    double& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    double operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    static DenseMatrix identity(std::size_t n);

    DenseMatrix operator+(const DenseMatrix& rhs) const;
    DenseMatrix operator-(const DenseMatrix& rhs) const;
    DenseMatrix operator*(double scalar) const;
    DenseMatrix operator/(double scalar) const;
    DenseMatrix operator*(const DenseMatrix& rhs) const;

    DenseMatrix& operator+=(const DenseMatrix& rhs);
    DenseMatrix& operator-=(const DenseMatrix& rhs);

    double norm_one() const;
    std::vector<double> multiply_row(const std::vector<double>& row) const;

    std::vector<double> solve(const std::vector<double>& rhs) const;
    DenseMatrix solve(const DenseMatrix& rhs) const;
    DenseMatrix expm() const;

private:
    void require_same_shape(const DenseMatrix& rhs) const;

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};


class SparseMatrix {
public:
    SparseMatrix() = default;
    SparseMatrix(std::size_t rows, std::size_t cols);

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }

    void add(std::size_t row, std::size_t col, double value);
    std::vector<double> multiply_transpose(const std::vector<double>& x) const;
    std::vector<double> exp_action_row(const std::vector<double>& row, double time) const;

private:
    struct Entry { std::size_t col; double value; };
    std::vector<double> arnoldi_step(const std::vector<double>& x, double dt, int dimension) const;

    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<std::vector<Entry>> rows_data_;
};

inline DenseMatrix operator*(double scalar, const DenseMatrix& matrix) {
    return matrix * scalar;
}

}  // namespace ladder_pricer
