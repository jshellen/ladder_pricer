#include "ladder_pricer/matrix.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

extern "C" {
void dgesv_(const int* n, const int* nrhs, double* a, const int* lda,
            int* ipiv, double* b, const int* ldb, int* info);
}

namespace ladder_pricer {

DenseMatrix::DenseMatrix(std::size_t rows, std::size_t cols, double value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

DenseMatrix DenseMatrix::identity(std::size_t n) {
    DenseMatrix out(n, n);
    for (std::size_t i = 0; i < n; ++i) out(i, i) = 1.0;
    return out;
}

void DenseMatrix::require_same_shape(const DenseMatrix& rhs) const {
    if (rows_ != rhs.rows_ || cols_ != rhs.cols_) {
        throw std::invalid_argument("matrix shapes do not match");
    }
}

DenseMatrix DenseMatrix::operator+(const DenseMatrix& rhs) const {
    require_same_shape(rhs);
    DenseMatrix out(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] + rhs.data_[i];
    return out;
}

DenseMatrix DenseMatrix::operator-(const DenseMatrix& rhs) const {
    require_same_shape(rhs);
    DenseMatrix out(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] - rhs.data_[i];
    return out;
}

DenseMatrix DenseMatrix::operator*(double scalar) const {
    DenseMatrix out(rows_, cols_);
    for (std::size_t i = 0; i < data_.size(); ++i) out.data_[i] = data_[i] * scalar;
    return out;
}

DenseMatrix DenseMatrix::operator/(double scalar) const {
    if (scalar == 0.0) throw std::invalid_argument("division by zero");
    return *this * (1.0 / scalar);
}

DenseMatrix DenseMatrix::operator*(const DenseMatrix& rhs) const {
    if (cols_ != rhs.rows_) throw std::invalid_argument("matrix multiply shape mismatch");
    DenseMatrix out(rows_, rhs.cols_);
    for (std::size_t i = 0; i < rows_; ++i) {
        for (std::size_t k = 0; k < cols_; ++k) {
            const double a = (*this)(i, k);
            if (a == 0.0) continue;
            for (std::size_t j = 0; j < rhs.cols_; ++j) {
                out(i, j) += a * rhs(k, j);
            }
        }
    }
    return out;
}

DenseMatrix& DenseMatrix::operator+=(const DenseMatrix& rhs) {
    require_same_shape(rhs);
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] += rhs.data_[i];
    return *this;
}

DenseMatrix& DenseMatrix::operator-=(const DenseMatrix& rhs) {
    require_same_shape(rhs);
    for (std::size_t i = 0; i < data_.size(); ++i) data_[i] -= rhs.data_[i];
    return *this;
}

double DenseMatrix::norm_one() const {
    double best = 0.0;
    for (std::size_t j = 0; j < cols_; ++j) {
        double sum = 0.0;
        for (std::size_t i = 0; i < rows_; ++i) sum += std::abs((*this)(i, j));
        best = std::max(best, sum);
    }
    return best;
}

std::vector<double> DenseMatrix::multiply_row(const std::vector<double>& row) const {
    if (row.size() != rows_) throw std::invalid_argument("row vector size mismatch");
    std::vector<double> out(cols_, 0.0);
    for (std::size_t i = 0; i < rows_; ++i) {
        const double x = row[i];
        if (x == 0.0) continue;
        for (std::size_t j = 0; j < cols_; ++j) out[j] += x * (*this)(i, j);
    }
    return out;
}

std::vector<double> DenseMatrix::solve(const std::vector<double>& rhs) const {
    if (rows_ != cols_ || rhs.size() != rows_) throw std::invalid_argument("linear solve shape mismatch");
    DenseMatrix B(rows_, 1);
    for (std::size_t i = 0; i < rows_; ++i) B(i, 0) = rhs[i];
    const DenseMatrix X = solve(B);
    std::vector<double> out(rows_);
    for (std::size_t i = 0; i < rows_; ++i) out[i] = X(i, 0);
    return out;
}

DenseMatrix DenseMatrix::solve(const DenseMatrix& rhs) const {
    if (rows_ != cols_ || rhs.rows_ != rows_) throw std::invalid_argument("linear solve shape mismatch");
    if (rows_ > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        rhs.cols_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("linear system too large for LAPACK integer interface");
    }

    const int n = static_cast<int>(rows_);
    const int nrhs = static_cast<int>(rhs.cols_);
    const int lda = n;
    const int ldb = n;

    // LAPACK uses column-major storage; DenseMatrix is row-major.
    std::vector<double> a_col(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a_col[static_cast<std::size_t>(j) * n + i] = (*this)(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }
    std::vector<double> b_col(static_cast<std::size_t>(n) * static_cast<std::size_t>(nrhs));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nrhs; ++j) {
            b_col[static_cast<std::size_t>(j) * n + i] = rhs(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
        }
    }

    std::vector<int> piv(static_cast<std::size_t>(n));
    int info = 0;
    dgesv_(&n, &nrhs, a_col.data(), &lda, piv.data(), b_col.data(), &ldb, &info);
    if (info < 0) throw std::runtime_error("LAPACK dgesv received an invalid argument");
    if (info > 0) throw std::runtime_error("singular or numerically singular linear system");

    DenseMatrix x(rows_, rhs.cols_);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nrhs; ++j) {
            x(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) =
                b_col[static_cast<std::size_t>(j) * n + i];
        }
    }
    return x;
}

DenseMatrix DenseMatrix::expm() const {
    if (rows_ != cols_) throw std::invalid_argument("matrix exponential requires square matrix");
    if (rows_ == 0) return *this;

    constexpr double theta13 = 5.371920351148152;
    const double n1 = norm_one();
    if (n1 == 0.0) return identity(rows_);
    const int s = n1 > theta13 ? std::max(0, static_cast<int>(std::ceil(std::log2(n1 / theta13)))) : 0;
    const double scale = std::ldexp(1.0, s);
    const DenseMatrix A = *this / scale;

    static constexpr double b[] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0, 129060195264000.0, 10559470521600.0,
        670442572800.0, 33522128640.0, 1323241920.0, 40840800.0,
        960960.0, 16380.0, 182.0, 1.0
    };

    const DenseMatrix I = identity(rows_);
    const DenseMatrix A2 = A * A;
    const DenseMatrix A4 = A2 * A2;
    const DenseMatrix A6 = A4 * A2;

    const DenseMatrix U = A * (
        A6 * (A6 * b[13] + A4 * b[11] + A2 * b[9])
        + A6 * b[7] + A4 * b[5] + A2 * b[3] + I * b[1]
    );
    const DenseMatrix V =
        A6 * (A6 * b[12] + A4 * b[10] + A2 * b[8])
        + A6 * b[6] + A4 * b[4] + A2 * b[2] + I * b[0];

    DenseMatrix R = (V - U).solve(V + U);
    for (int i = 0; i < s; ++i) R = R * R;
    return R;
}

}  // namespace ladder_pricer

namespace ladder_pricer {

SparseMatrix::SparseMatrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), rows_data_(rows) {}

void SparseMatrix::add(std::size_t row, std::size_t col, double value) {
    if (row >= rows_ || col >= cols_) throw std::out_of_range("sparse matrix index out of range");
    if (value == 0.0) return;
    auto& entries = rows_data_[row];
    for (auto& entry : entries) {
        if (entry.col == col) {
            entry.value += value;
            return;
        }
    }
    entries.push_back({col, value});
}

std::vector<double> SparseMatrix::multiply_transpose(const std::vector<double>& x) const {
    if (x.size() != rows_) throw std::invalid_argument("sparse matvec size mismatch");
    std::vector<double> out(cols_, 0.0);
    for (std::size_t i = 0; i < rows_; ++i) {
        const double xi = x[i];
        if (xi == 0.0) continue;
        for (const auto& e : rows_data_[i]) out[e.col] += e.value * xi;
    }
    return out;
}

namespace {

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) out += a[i] * b[i];
    return out;
}

double norm2(const std::vector<double>& x) { return std::sqrt(std::max(0.0, dot(x, x))); }

double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double out = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) out = std::max(out, std::abs(a[i] - b[i]));
    return out;
}

double max_abs_vec(const std::vector<double>& x) {
    double out = 0.0;
    for (double v : x) out = std::max(out, std::abs(v));
    return out;
}

}  // namespace

std::vector<double> SparseMatrix::arnoldi_step(const std::vector<double>& x, double dt, int dimension) const {
    const double beta = norm2(x);
    if (beta == 0.0 || dt == 0.0) return x;
    const int m = std::min<int>(dimension, static_cast<int>(x.size()));
    std::vector<std::vector<double>> V;
    V.reserve(m + 1);
    V.push_back(x);
    for (double& v : V[0]) v /= beta;
    DenseMatrix H(static_cast<std::size_t>(m + 1), static_cast<std::size_t>(m));
    int k = 0;

    for (int j = 0; j < m; ++j) {
        auto w = multiply_transpose(V[static_cast<std::size_t>(j)]);
        // Twice-reorthogonalized modified Gram-Schmidt is still cheap at m<=40.
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i <= j; ++i) {
                const double hij = dot(V[static_cast<std::size_t>(i)], w);
                H(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) += hij;
                for (std::size_t r = 0; r < w.size(); ++r) w[r] -= hij * V[static_cast<std::size_t>(i)][r];
            }
        }
        const double hnext = norm2(w);
        H(static_cast<std::size_t>(j + 1), static_cast<std::size_t>(j)) = hnext;
        k = j + 1;
        if (hnext <= 64.0 * std::numeric_limits<double>::epsilon()) break;
        for (double& v : w) v /= hnext;
        V.push_back(std::move(w));
    }

    DenseMatrix Hk(static_cast<std::size_t>(k), static_cast<std::size_t>(k));
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) Hk(static_cast<std::size_t>(i), static_cast<std::size_t>(j)) = H(static_cast<std::size_t>(i), static_cast<std::size_t>(j));
    }
    const DenseMatrix E = (Hk * dt).expm();
    std::vector<double> out(x.size(), 0.0);
    for (int j = 0; j < k; ++j) {
        const double coeff = beta * E(static_cast<std::size_t>(j), 0);
        for (std::size_t r = 0; r < out.size(); ++r) out[r] += coeff * V[static_cast<std::size_t>(j)][r];
    }
    return out;
}

std::vector<double> SparseMatrix::exp_action_row(const std::vector<double>& row, double time) const {
    if (rows_ != cols_ || row.size() != rows_) throw std::invalid_argument("exp action shape mismatch");
    if (time < 0.0) throw std::invalid_argument("exp action time must be nonnegative");
    if (time == 0.0) return row;

    auto propagate = [&](int steps) {
        std::vector<double> x = row;
        const double dt = time / static_cast<double>(steps);
        for (int s = 0; s < steps; ++s) x = arnoldi_step(x, dt, 36);
        return x;
    };

    int steps = 1;
    std::vector<double> coarse = propagate(steps);
    constexpr int kMaxSteps = 256;
    while (steps < kMaxSteps) {
        const int refined_steps = steps * 2;
        std::vector<double> refined = propagate(refined_steps);
        const double error = max_abs_diff(coarse, refined);
        const double scale = std::max(1.0, max_abs_vec(refined));
        if (error <= 2e-9 * scale) return refined;
        coarse = std::move(refined);
        steps = refined_steps;
    }
    return coarse;
}

}  // namespace ladder_pricer
