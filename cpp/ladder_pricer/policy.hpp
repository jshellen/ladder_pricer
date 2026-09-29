#pragma once

#include "common.hpp"
#include "quote_analytics.hpp"

#include <stdexcept>
#include <vector>

namespace hjb {

struct QuotePolicy {
    std::vector<double> q_grid;
    std::vector<double> sizes;
    Matrix bid;
    Matrix ask;

    QuotePolicy() = default;

    QuotePolicy(
        std::vector<double> q_grid_,
        std::vector<double> sizes_,
        Matrix bid_,
        Matrix ask_
    )
        : q_grid(std::move(q_grid_)),
          sizes(std::move(sizes_)),
          bid(std::move(bid_)),
          ask(std::move(ask_)) {
        validate();
    }

    static void resize_matrix(Matrix& mat, std::size_t rows, std::size_t cols) {
        mat.resize(rows);
        for (auto& row : mat) {
            row.assign(cols, 0.0);
        }
    }

    void reset_shape(
        const std::vector<double>& q_grid_,
        const std::vector<double>& sizes_
    ) {
        q_grid = q_grid_;
        sizes = sizes_;
        resize_matrix(bid, q_grid.size(), sizes.size());
        resize_matrix(ask, q_grid.size(), sizes.size());
    }

    void validate() const {
        validate_strictly_increasing(q_grid, "QuotePolicy.q_grid");
        validate_positive_strictly_increasing(sizes, "QuotePolicy.sizes");

        const std::size_t nq = q_grid.size();
        const std::size_t nz = sizes.size();

        if (bid.size() != nq || ask.size() != nq) {
            throw std::invalid_argument("QuotePolicy: row count mismatch.");
        }
        for (const auto& row : bid) {
            if (row.size() != nz) {
                throw std::invalid_argument("QuotePolicy: bid column count mismatch.");
            }
        }
        for (const auto& row : ask) {
            if (row.size() != nz) {
                throw std::invalid_argument("QuotePolicy: ask column count mismatch.");
            }
        }
    }

    const Matrix& matrix(Side side) const {
        return side == Side::Bid ? bid : ask;
    }

    Matrix& matrix_mut(Side side) {
        return side == Side::Bid ? bid : ask;
    }

    double& delta_ref(std::size_t i, std::size_t j, Side side) {
        return matrix_mut(side)[i][j];
    }

    double interp_q_column(const Matrix& mat, std::size_t j, double q) const {
        if (q_grid.size() == 1) {
            return mat[0][j];
        }
        const auto [i, tq] = locate_segment_with_weight(q_grid, q);
        return mat[i][j] + tq * (mat[i + 1][j] - mat[i][j]);
    }

    double delta(double q, double z, Side side) const {
        const Matrix& mat = matrix(side);

        if (sizes.size() == 1) {
            return interp_q_column(mat, 0, q);
        }

        const auto [j, tz] = locate_segment_with_weight(sizes, z);
        const double v0 = interp_q_column(mat, j, q);
        const double v1 = interp_q_column(mat, j + 1, q);
        return v0 + tz * (v1 - v0);
    }

    double reference_delta(double q, Side side) const {
        if (sizes.empty()) {
            throw std::runtime_error("QuotePolicy::reference_delta: no sizes available.");
        }
        return delta(q, sizes.front(), side);
    }

    QuoteSummary quote_summary(
        double q,
        double z,
        Side side,
        double mid,
        double spread
    ) const {
        const double d = delta(q, z, side);
        const double d_ref = reference_delta(q, side);
        return QuoteMetrics::make_summary(d, d_ref, side, mid, spread);
    }
};

struct DarkPoolPolicy {
    std::vector<double> q_grid;
    std::vector<double> bid_size;
    std::vector<double> ask_size;
    std::vector<unsigned char> bid_active;
    std::vector<unsigned char> ask_active;

    DarkPoolPolicy() = default;

    explicit DarkPoolPolicy(std::vector<double> q_grid_)
        : q_grid(std::move(q_grid_)) {
        reset_shape(q_grid);
    }

    void reset_shape(const std::vector<double>& q_grid_) {
        q_grid = q_grid_;
        bid_size.assign(q_grid.size(), 0.0);
        ask_size.assign(q_grid.size(), 0.0);
        bid_active.assign(q_grid.size(), 0u);
        ask_active.assign(q_grid.size(), 0u);
    }

    void validate() const {
        validate_strictly_increasing(q_grid, "DarkPoolPolicy.q_grid");
        if (bid_size.size() != q_grid.size() || ask_size.size() != q_grid.size()) {
            throw std::invalid_argument("DarkPoolPolicy: size vector length mismatch.");
        }
        if (bid_active.size() != q_grid.size() || ask_active.size() != q_grid.size()) {
            throw std::invalid_argument("DarkPoolPolicy: active flag length mismatch.");
        }
    }

    const std::vector<double>& size_vector(Side side) const {
        return side == Side::Bid ? bid_size : ask_size;
    }

    std::vector<double>& size_vector_mut(Side side) {
        return side == Side::Bid ? bid_size : ask_size;
    }

    const std::vector<unsigned char>& active_vector(Side side) const {
        return side == Side::Bid ? bid_active : ask_active;
    }

    std::vector<unsigned char>& active_vector_mut(Side side) {
        return side == Side::Bid ? bid_active : ask_active;
    }

    double posted_size(double q, Side side) const {
        const auto& vals = size_vector(side);
        if (q_grid.empty()) {
            throw std::runtime_error("DarkPoolPolicy::posted_size: q_grid is empty.");
        }
        if (q_grid.size() == 1) {
            return vals.front();
        }

        const auto [i, tq] = locate_segment_with_weight(q_grid, q);
        return tq <= 0.5 ? vals[i] : vals[i + 1];
    }

    bool is_active(std::size_t q_index, Side side) const {
        const auto& flags = active_vector(side);
        return q_index < flags.size() && flags[q_index] != 0u;
    }

    void set_posted_size(std::size_t q_index, Side side, double size, bool active) {
        auto& vals = size_vector_mut(side);
        auto& flags = active_vector_mut(side);
        if (q_index >= vals.size() || q_index >= flags.size()) {
            throw std::out_of_range("DarkPoolPolicy::set_posted_size: q_index out of range.");
        }
        vals[q_index] = size;
        flags[q_index] = active ? 1u : 0u;
    }
};

} // namespace hjb
