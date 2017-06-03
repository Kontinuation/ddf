#ifndef OP_H
#define OP_H

// Implementation of Ops

#include "op_base.hh"

namespace ddf {

// matrix variable multiplied by vector
//   y = x * v
//     where shape of x is (a, b) and shape of v is (b, 1);
//     y is a vector of (a, 1)
template <typename numeric_type>
class matrix_mult: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;

    matrix_mult(void): math_op<numeric_type>("matmul", 2, {
            {3, 3},             // A * Dw, Dw * B
            {1, 1}              // A * Dx, Dx * B
        }) {
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        if (k_param == 0) _w = v;
        else _x = v;
    }

    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _w;
        else return _x;
    }

    int size_f() {
        int n = _x.size();
        int m = _w.size() / n;
        if (m * n != _w.size()) {
            throw exception(
                "matrix size should be multiplier of vector size");
        }
        return m;
    }

    // y = _w * _x
    void f(vector_type &y) {
        matrix_type mat = matrix_view_of(_w);
        mat.mult(_x, y);
    }

    void Df(int k_param, matrix_type &D) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            D_w(D);
        } else {
            D_x(D);
        }
    }

    void D_w(matrix_type &D) {
        matrix_type mat = matrix_view_of(_w);
        int m = mat.shape(0), n = mat.shape(1);
        D.resize(m, _w.size());
        D.fill(0);
        for (int k = 0; k < _w.size(); k++) {
            D(k / n, k) = _x[k % n];
        }
    }

    void D_x(matrix_type &D) {
        matrix_type mat = matrix_view_of(_w);
        D.resize(mat.shape(0), mat.shape(1));
        D.copy_from(mat.raw_data());
    }

    matrix_type matrix_view_of(const vector_type &w) const {
        int n = _x.size();
        int m = w.size() / n;
        assert(("matrix size should be multiplier of vector size",
                m * n == w.size()));
        return matrix_type(m, n, w.raw_data());
    }

    // We need to apply an optimization for matrix mult here: the Jacobian of
    // `D (W * x)` has significiant pattern, which would lead to a good
    // optimization

    void mult_grad(int k_param, const matrix_type &B, matrix_type &DB) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            opt::mult_by_strided_matrix(B, _w, DB);
        } else {
            matrix_type D = matrix_view_of(_w);
            D.mult(B, DB);
        }
    }

    void mult_by_grad(int k_param, const matrix_type &A, matrix_type &AD) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            opt::mult_strided_matrix(A, _x, AD);
        } else {
            matrix_type D = matrix_view_of(_w);
            A.mult(D, AD);
        }
    }

    virtual int cost_mult_grad(int k_param, const matrix_type &B) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            return B.shape(0) * B.shape(1);
        } else {
            matrix_type D = matrix_view_of(_w);
            return D.shape(0) * D.shape(1) * B.shape(1);
        }
    }

    virtual int cost_mult_by_grad(int k_param, const matrix_type &A) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            return A.shape(0) * A.shape(1);
        } else {
            matrix_type D = matrix_view_of(_w);
            return A.shape(0) * A.shape(1) * D.shape(1);
        }
    }

protected:
    vector_type _w;
    vector_type _x;
};

// sum cross_entropy(label, softmax(x)) where label in labels
template <typename numeric_type>
class softmax_cross_entropy_with_logits: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;
    
    softmax_cross_entropy_with_logits(void)
        : math_op<numeric_type>("DS", 2) {
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            _w = v;
            numeric_type shift = *std::max_element(
                v.raw_data(), v.raw_data() + v.size());
            
            // precalculate exp(w) and multiplier
            numeric_type divider = 0;
            int n = _w.size();            
            _exp_w.resize(n);
            for (int k = 0; k < n; k++) {
                numeric_type exp_wk = exp(_w[k] - shift);
                _exp_w[k] = exp_wk;
                divider += exp_wk;
            }
            _multiplier = 1 / divider;
        } else if (k_param == 1) {
            _l = v;
        }
    }

    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _w;
        else if (k_param == 1) return _l;
        else return vector_type();
    }

    int size_f() {
        return 1;
    }

    void f(vector_type &y) {
        int n = _w.size();
        assert(("prediction size should match with label", n == _l.size()));
        numeric_type sum_ce = 0;
        for (int k = 0; k < n; k++) {
            sum_ce -= _l[k] * log(_exp_w[k] * _multiplier);
        }

        y.resize(1);
        y[0] = sum_ce;
    }

    void Df(int k_param, matrix_type &D) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            D_w(D);
        } else {
            assert(("gradient of label is not implemented", false));
        }
    }
    
    void D_w(matrix_type &D) {
        int n = _w.size();
        assert(("prediction size should match with label", n == _l.size()));
        D.resize(1, n);
        for (int i = 0; i < n; i++) {
            D(0, i) = (_exp_w[i] * _multiplier) - _l[i];
        }
    }

protected:
    vector_type _w;
    vector_type _l;
    vector_type _exp_w;
    numeric_type _multiplier;
};

// Rectifier
template <typename numeric_type>
class relu: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;
    
    relu(): math_op<numeric_type>("relu", 1, {
            {2, 2},             // A * Dx, Dx * B
        }) {
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        _x = v;
    }
    
    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _x;
        else return vector_type();
    }

    int size_f() {
        return _x.size();
    }

    void f(vector_type &y) {
        int n = _x.size();
        y.resize(n);
        for (int k = 0; k < n; k++) {
            y[k] = _x[k] > 0? _x[k]: 0;
        }
    }

    void Df(int k_param, matrix_type &D) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            D_x(D);
        }
    }

    void D_x(matrix_type &D) {
        int n = _x.size();
        D.resize(n, n);
        D.fill(0);
        for (int k = 0; k < n; k++) {
            D(k, k) = _x[k] > 0? 1: 0;
        }
    }

    // the jacobian of `D relu` is a diagonal matrix, which would lead to a
    // good optimization

    void mult_grad(int k_param, const matrix_type &B, matrix_type &DB) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            opt::mult_by_relu_matrix(B, _x, DB);
        }
    }

    void mult_by_grad(int k_param, const matrix_type &A, matrix_type &AD) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            opt::mult_relu_matrix(A, _x, AD);
        }
    }

    virtual int cost_mult_grad(int k_param, const matrix_type &B) {
        assert_param_dim(k_param);
        return B.shape(0) * B.shape(1);
    }

    virtual int cost_mult_by_grad(int k_param, const matrix_type &A) {
        assert_param_dim(k_param);
        return A.shape(0) * A.shape(1);
    }

protected:
    vector_type _x;
};

} // end namespace ddf

#endif /* OP_H */
