#pragma once

#include "common.hpp"

#include <cmath>

namespace hjb {

struct QuoteSummary {
    double delta{0.0};

    double price_improvement_frac{0.0};
    double price_improvement_pct_of_spread{0.0};
    double price_improvement_pips{0.0};

    double quote_relative_to_mid{0.0};
    double quote_relative_to_mid_pips{0.0};

    double distance_to_mid{0.0};
    double distance_to_mid_pips{0.0};

    double reference_delta{0.0};
    double volume_premium_pips{0.0};

    double quote_price{0.0};
};

struct QuoteMetrics {
    static constexpr double pips_per_unit = 10000.0;

    static double price_improvement(double delta, double spread) {
        return spread * delta;
    }

    static double price_improvement_pct_of_spread(double delta) {
        return 100.0 * delta;
    }

    static double price_improvement_pips(double delta, double spread) {
        return pips_per_unit * price_improvement(delta, spread);
    }

    static double quote_relative_to_mid(double delta, Side side, double spread) {
        return side == Side::Bid
            ? spread * (delta - 0.5)
            : spread * (0.5 - delta);
    }

    static double quote_relative_to_mid_pips(double delta, Side side, double spread) {
        return pips_per_unit * quote_relative_to_mid(delta, side, spread);
    }

    static double distance_to_mid(double delta, double spread) {
        return std::abs(spread * (0.5 - delta));
    }

    static double distance_to_mid_pips(double delta, double spread) {
        return pips_per_unit * distance_to_mid(delta, spread);
    }

    static double volume_premium_pips(double delta_ref, double delta_cur, double spread) {
        return pips_per_unit * spread * (delta_ref - delta_cur);
    }

    static double quote_price(double mid, double delta, Side side, double spread) {
        return mid + quote_relative_to_mid(delta, side, spread);
    }

    static QuoteSummary make_summary(
        double delta,
        double delta_ref,
        Side side,
        double mid,
        double spread
    ) {
        QuoteSummary out;
        out.delta = delta;
        out.price_improvement_frac = price_improvement(delta, spread);
        out.price_improvement_pct_of_spread = price_improvement_pct_of_spread(delta);
        out.price_improvement_pips = price_improvement_pips(delta, spread);
        out.quote_relative_to_mid = quote_relative_to_mid(delta, side, spread);
        out.quote_relative_to_mid_pips = quote_relative_to_mid_pips(delta, side, spread);
        out.distance_to_mid = distance_to_mid(delta, spread);
        out.distance_to_mid_pips = distance_to_mid_pips(delta, spread);
        out.reference_delta = delta_ref;
        out.volume_premium_pips = volume_premium_pips(delta_ref, delta, spread);
        out.quote_price = quote_price(mid, delta, side, spread);
        return out;
    }
};

} // namespace hjb
