#ifndef OP_H
#define OP_H

#include "nd_array.hh"
#include <cmath>

namespace ddf {

// Deep networks are formed by stacking simple operations to construct a target
// loss function, then perform minization on that loss function. Here we
// provide some building blocks (ops) for constructing deep networds.

template <typename _numeric_type>
class math_op {
public:
    typedef _numeric_type numeric_type;
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;
    
#define assert_param_dim(k_param)               \
    assert(("parameter index out of bound", (k_param) >= this->_n_params))
    
    math_op(const std::string &name, int n_params, int mult_opt = 0, 
        int mult_by_opt = 0)
        : _name(name), _n_params(n_params), _mult_opt(mult_opt),
          _mult_by_opt(mult_by_opt) {
    }
    virtual ~math_op(void) = default;

    // assign value for k-th parameter for this operator
    virtual void prepare(int k_param, const vector_type &x) = 0;
    virtual vector_type get_param(int k_param) = 0;
    virtual void ready(void) {}
    
    // evaluate on prepared value  
    virtual void f(vector_type &y) = 0;

    // Df_x is the Jacobian matrix of f_x about k_param-th parameter
    virtual void Df(int k_param, matrix_type &D) {
        slow_Df(k_param, D);
    }
    
    // This is a fallback implementation of partial derivatives, but it is
    // pretty slow
    void slow_Df(int k_param , matrix_type &D, numeric_type delta = 1e-6) {
        assert_param_dim(k_param);
        
        // get starting point and dimension of f(x)
        vector_type x = get_param(0), y0;
        f(y0);
        int x_size = x.size(), y_size = y0.size();
        D = matrix_type(y_size, x_size);

        // calculate derivatives
        vector_type y1(y_size);
        for (int dim = 0; dim < x_size; dim++) {
            // move a small step toward this dimension
            numeric_type x_dim = x[dim];
            x[dim] += delta;
            prepare(k_param, x);
            ready();
            f(y1);
            x[dim] = x_dim;

            // partial derivative on this dimension
            y1 -= y0;
            y1 *= (1 / delta);
            D.set_column(dim, y1);
        }

        // reset param
        prepare(k_param, x);
        ready();
    }

    // short-hand for evaluating value for operator with 1, 2 and 3 input
    // parameters
    void f_x(const vector_type &x, vector_type &y) {
        prepare(0, x);
        ready();
        f(y);
    }
    
    void f_x(const vector_type &x0, const vector_type &x1, 
        vector_type &y) {
        prepare(0, x0);
        prepare(1, x1);
        ready();
        f(y);
    }

    void f_x(const vector_type &x0, const vector_type &x1, 
        const vector_type &x2, vector_type &y) {
        prepare(0, x0);
        prepare(1, x1);
        prepare(2, x2);
        ready();
        f(y);
    }

    // short-hand for evaluating gradient of operator with 1, 2 and 3 input
    // parameters
    void Df_x(const vector_type &x, matrix_type &D) {
        prepare(0, x);
        ready();
        Df(0, D);
    }

    void Df_x(const vector_type &x0, const vector_type &x1, int k_param,
        matrix_type &D) {
        prepare(0, x0);
        prepare(1, x1);
        ready();
        Df(k_param, D);
    }

    void Df_x(const vector_type &x0, const vector_type &x1,
        const vector_type &x2, int k_param, matrix_type &D) {
        prepare(0, x0);
        prepare(1, x1);
        prepare(2, x2);
        ready();
        Df(k_param, D);
    }

    // multiply gradient matrix of parameter on k_param-th dimension by B
    virtual void mult_grad(int k_param, const matrix_type &B, matrix_type &DB) {
        matrix_type D;
        Df(k_param, D);
        D.mult(B, DB);
    }

    // A multiplied by gradient matrix of parameter on k_param-th dimension at x
    virtual void mult_by_grad(int k_param, const matrix_type &A, matrix_type &AD) {
        matrix_type D;
        Df(k_param, D);
        A.mult(D, AD);
    }

    std::string name() const { return _name; }
    int n_params() const { return n_params; }
    int mult_opt_level(void) const { return _mult_opt; }
    int mult_by_opt_level(void) const { return _mult_by_opt; }

protected:
    std::string _name;
    int _n_params;

    // optimization level for jacobian matrix multiplications
    int _mult_opt;
    int _mult_by_opt;
};

// matrix variable multiplied by vector
//   y = x * v
//     where shape of x is (a, b) and shape of v is (b, 1);
//     y is a vector of (a, 1)
template <typename numeric_type>
class matrix_mult: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;

    matrix_mult(void): math_op<numeric_type>("matmul", 2, 1, 2) {
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
        if (k_param == 0) {
            opt::mult_by_strided_matrix(B, _w, DB);
        } else {
            math_op<numeric_type>::mult_grad(k_param, B, DB);
        }
    }

    void mult_by_grad(int k_param, const matrix_type &A, matrix_type &AD) {
        if (k_param == 0) {
            opt::mult_strided_matrix(A, _x, AD);
        } else {
            math_op<numeric_type>::mult_by_grad(k_param, A, AD);
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
    
    softmax_cross_entropy_with_logits(const vector_type &l)
        : math_op<numeric_type>("DS", 1), _l(l) {
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            _w = v;
            
            // precalculate exp(w) and multiplier
            numeric_type divider = 0;
            int n = _w.size();
            assert(("prediction size should match with label", n == _l.size()));
            
            _exp_w.resize(n);
            for (int k = 0; k < n; k++) {
                numeric_type exp_wk = exp(_w[k]);
                _exp_w[k] = exp_wk;
                divider += exp_wk;
            }
            _multiplier = 1 / divider;
        }
    }

    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _w;
        else return vector_type();
    }

    void f(vector_type &y) {
        int n = _w.size();
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
        }
    }
    
    void D_w(matrix_type &D) {
        int n = _w.size();
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
    
    relu(): math_op<numeric_type>("relu", 1) {
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

protected:
    vector_type _x;
};

} // end namespace ddf

#endif /* OP_H */
