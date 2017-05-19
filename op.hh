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
    math_op(const char *name = "op",
        int mult_opt = 0, int mult_by_opt = 0)
        : _mult_opt(mult_opt), _mult_by_opt(mult_by_opt) {
        strncpy(_name, name, sizeof _name);
        _name[(sizeof _name) - 1] = '\0';
    }
    virtual ~math_op(void) = default;
    virtual void f_x(const vector<numeric_type> &x,
                     vector<numeric_type> &y) = 0;

    // Df_x is the Jacobian matrix of f_x
    // This is a fallback implementation of partial derivatives
    virtual void Df_x(
        const vector<numeric_type> &x, matrix<numeric_type> &y) {
        slow_Df_x(x, y);
    }

    void slow_Df_x(const vector<numeric_type> &x, matrix<numeric_type> &y) {
        numeric_type delta = 1e-6;

        // get dimension of f(x)
        ddf::vector<numeric_type> y0(0);
        f_x(x, y0);
        int x_size = x.size(), y_size = y0.size();
        y = ddf::matrix<numeric_type>(y_size, x_size);

        // calculate derivatives
        for (int dim = 0; dim < x_size; dim++) {
            numeric_type x_dim = x[dim];
            x[dim] += delta;
            ddf::vector<numeric_type> y1(y_size);
            f_x(x, y1);
            x[dim] = x_dim;
            y1 -= y0;
            y1 *= (1 / delta);
            y.set_column(dim, y1);
        }
    }

    // multiply gradient matrix at x by B
    virtual void mult_grad(
        const matrix<numeric_type> &B, const vector<numeric_type> &x,
        matrix<numeric_type> &y) {
        matrix<numeric_type> A;
        Df_x(x, A);
        A.mult(B, y);
    }

    // A multiplied by gradient matrix at x
    virtual void mult_by_grad(
        const matrix<numeric_type> &A, const vector<numeric_type> &x, 
        matrix<numeric_type> &y) {
        matrix<numeric_type> B;
        Df_x(x, B);
        A.mult(B, y);
    }

    std::string name() const { return _name; }
    int mult_opt_level(void) const { return _mult_opt; }
    int mult_by_opt_level(void) const { return _mult_by_opt; }

    char _name[64];

    // optimization level for jacobian matrix multiplications
    int _mult_opt;
    int _mult_by_opt;
};

// matrix variable multiplied by vector
//   y = x * v
//     where shape of x is (a, b) and shape of v is (b, 1);
//     y is a vector of (a, 1)
template <typename numeric_type>
struct matrix_mult: math_op<numeric_type> {
    matrix_mult(const vector<numeric_type> &v)
        : math_op<numeric_type>("matmul", 1, 2), _v(v) {
    }

    // y = x * _v
    void f_x(const vector<numeric_type> &x, vector<numeric_type> &y) {
        matrix<numeric_type> mat = matrix_view_of(x);
        mat.mult(_v, y);
    }

    void Df_x(const vector<numeric_type> &x, matrix<numeric_type> &y) {
        matrix<numeric_type> mat = matrix_view_of(x);
        int m = mat.shape(0), n = mat.shape(1);
        y.resize(m, x.size());
        y.fill(0);
        for (int dim = 0; dim < x.size(); dim++) {
            y(dim / n, dim) = _v[dim % n];
        }
    }

    matrix<numeric_type> matrix_view_of(const vector<numeric_type> &x) const {
        int n = _v.size();
        int m = x.size() / n;
        assert(("matrix size should be multiplier of vector size",
                m * n == x.size()));
        return matrix<numeric_type>(m, n, x.raw_data());
    }

    // We need to apply an optimization for matrix mult here: the Jacobian of
    // `D (W * x)` has significiant pattern, which would lead to a good
    // optimization

    void mult_grad(
        const matrix<numeric_type> &B, const vector<numeric_type> &x, 
        matrix<numeric_type> &y) {
        ddf::mult_by_strided_matrix(B, _v, y);
    }

    void mult_by_grad(
        const matrix<numeric_type> &A, const vector<numeric_type> &x, 
        matrix<numeric_type> &y) {
        ddf::mult_strided_matrix(A, _v, y);
    }

    vector<numeric_type> _v;
};

// sum cross_entropy(label, softmax(x)) where label in labels
template <typename numeric_type>
struct softmax_cross_entropy_with_logits: math_op<numeric_type> {
    softmax_cross_entropy_with_logits(const vector<numeric_type> &l)
        : math_op<numeric_type>("DS"), _l(l), _exp_w(0) {
    }

    void f_x(const vector<numeric_type> &w, vector<numeric_type> &y) {
        numeric_type multiplier = 0;
        int n = w.size();
        assert(("prediction size should match with label", n == _l.size()));

        // TODO: move calculation of multiplier as a common procedure
        _exp_w.resize(n);
        for (int k = 0; k < n; k++) {
            numeric_type exp_wk = exp(w[k]);
            _exp_w[k] = exp_wk;
            multiplier += exp_wk;
        }
        multiplier = 1 / multiplier;

        numeric_type sum_ce = 0;
        for (int k = 0; k < n; k++) {
            sum_ce -= _l[k] * log(_exp_w[k] * multiplier);
        }

        y.resize(1);
        y[0] = sum_ce;
    }

    void Df_x(const vector<numeric_type> &w, matrix<numeric_type> &y) {
        numeric_type multiplier = 0;
        int n = w.size();
        y.resize(1, n);

        // TODO: move calculation of multiplier as a common procedure
        _exp_w.resize(n);
        for (int k = 0; k < n; k++) {
            numeric_type exp_wk = exp(w[k]);
            _exp_w[k] = exp_wk;
            multiplier += exp_wk;
        }
        multiplier = 1 / multiplier;

        for (int i = 0; i < n; i++) {
            y(0, i) = (_exp_w[i] * multiplier) - _l[i];
        }
    }

    vector<numeric_type> _l;
    vector<numeric_type> _exp_w;
};

// Rectifier
template <typename numeric_type>
struct relu: math_op<numeric_type> {
    relu(): math_op<numeric_type>("relu") {
    }

    void f_x(const vector<numeric_type> &x, vector<numeric_type> &y) {
        int n = x.size();
        y.resize(n);
        for (int k = 0; k < n; k++) {
            y[k] = x[k] > 0? x[k]: 0;
        }
    }

    void Df_x(const vector<numeric_type> &x, matrix<numeric_type> &y) {
        int n = x.size();
        y.resize(n, n);
        y.fill(0);
        for (int k = 0; k < n; k++) {
            y(k, k) = x[k] > 0? 1: 0;
        }
    }

    // the jacobian of `D relu` is a diagonal matrix, which would lead to a
    // good optimization

    void mult_grad(
        const matrix<numeric_type> &B, const vector<numeric_type> &x, 
        matrix<numeric_type> &y) {

        int n = x.size();
        y.resize(n, B.shape(1));
        for (int k = 0; k < n; k++) {
            if (x[k] > 0) {
                std::copy_n(y(k,0), B(k,0), n);
            } else {
                std::fill_n(y(k,0), n, 0);
            }
        }
    }

    void mult_by_grad(
        const matrix<numeric_type> &A, const vector<numeric_type> &x, 
        matrix<numeric_type> &y) {
    }
};

} // end namespace ddf

#endif /* OP_H */
