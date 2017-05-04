#ifndef OP_H
#define OP_H

#include "nd_array.hh"
#include <cmath>

namespace ddf {

// Deep networks are formed by stacking simple operations to construct a target
// loss function, then perform minization on that loss function. Here we
// provide some building blocks (ops) for constructing deep networds.

template <typename _numeric_type>
struct math_op {
    typedef _numeric_type numeric_type;
    math_op(const char *name = "op") {
        strncpy(_name, name, sizeof _name);
        _name[(sizeof _name) - 1] = '\0';
    }
    virtual ~math_op(void) = default;
    virtual void f_x(const vector<numeric_type> &x,
                     vector<numeric_type> &y) = 0;
    // a fallback implementation of derivative
    virtual void df_x(
        const vector<numeric_type> &x, vector<numeric_type> &y,
        int dim) {
        assert(("dimension bounds check", dim < x.size()));
        numeric_type delta = 1e-6;
        numeric_type x_dim = x[dim];
        vector<numeric_type> y0 = y.clone();
        f_x(x, y0);
        x[dim] += delta;
        f_x(x, y);
        x[dim] = x_dim;
        y -= y0;
        y *= (1 / delta);
    }
    char _name[64];
};

// matrix variable multiplied by vector
//   y = x * v
//     where shape of x is (a, b) and shape of v is (b, 1);
//     y is a vector of (a, 1)
template <typename numeric_type>
struct matrix_mult: math_op<numeric_type> {
    matrix_mult(const vector<numeric_type> &v)
        : math_op<numeric_type>("matmul"), _v(v) {
    }

    // y = x * _v
    void f_x(const vector<numeric_type> &x, vector<numeric_type> &y) {
        matrix<numeric_type> mat = matrix_view_of(x);
        mat.mult(_v, y);
    }

    void df_x(const vector<numeric_type> &x, vector<numeric_type> &y, int dim) {
        matrix<numeric_type> mat = matrix_view_of(x);
        int m = mat.shape(0), n = mat.shape(1);
        assert(("dimension bounds check", dim < m * n));
        y.resize(m);
        y.fill(0);
        y[dim / n] = _v[dim % n];
    }

    matrix<numeric_type> matrix_view_of(const vector<numeric_type> &x) const {
        int n = _v.size();
        int m = x.size() / n;
        assert(("matrix size should be multiplier of vector size",
                m * n == x.size()));
        return matrix<numeric_type>(m, n, x.raw_data());
    }

    vector<numeric_type> _v;
};

// sum cross_entropy(label, softmax(x)) where label in labels
template <typename numeric_type>
struct softmax_cross_entropy_with_logits: math_op<numeric_type> {
    softmax_cross_entropy_with_logits(const vector<numeric_type> &l)
        : math_op<numeric_type>("DS"), _l(l) {
    }

    void f_x(const vector<numeric_type> &w, vector<numeric_type> &y) {
        numeric_type multiplier = 0;
        int n = w.size();

        // TODO: move calculation of multiplier as a common procedure
        for (int k = 0; k < n; k++) multiplier += exp(w[k]);
        multiplier = 1 / multiplier;

        numeric_type sum_ce = 0;
        for (int k = 0; k < n; k++) {
            sum_ce += _l[k] * log(w[k] * multiplier);
        }

        y.resize(1);
        y[0] = sum_ce;
    }

    // void df_x(const vector<numeric_type> &w, vector<numeric_type> &y, int dim) {
    //     numeric_type multiplier = 0;
    //     int n = w.size();
    //     assert(("dimension bounds check", dim < n));

    //     // TODO: move calculation of multiplier as a common procedure
    //     for (int k = 0; k < n; k++) multiplier += exp(w[k]);
    //     multiplier = 1 / multiplier;

    //     y.resize(1);
    //     y[0] = _l[dim] - w[dim] * multiplier;
    // }

    vector<numeric_type> _l;
};

} // end namespace ddf

#endif /* OP_H */
