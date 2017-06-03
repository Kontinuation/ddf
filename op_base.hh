#ifndef OP_BASE_H
#define OP_BASE_H

// Deep networks are formed by stacking simple operations to construct a target
// loss function, then perform minization on that loss function. Here we
// provide some building blocks (ops) for constructing deep networds.

#include "nd_array.hh"
#include <vector>
#include <cmath>

namespace ddf {

template <typename _numeric_type>
class math_op {
public:
    typedef _numeric_type numeric_type;
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;
    
#define assert_param_dim(k_param)               \
    assert(("parameter index out of bound", (k_param) < this->_n_params))
    
    math_op(const std::string &name, int n_params, 
        const std::vector<std::pair<int, int> > &opt_level = {})
        : _name(name), _n_params(n_params), _opt_level(opt_level) {
    }
    virtual ~math_op(void) = default;

    // assign value for k-th parameter for this operator
    virtual void prepare(int k_param, const vector_type &x) = 0;
    virtual vector_type get_param(int k_param) = 0;
    virtual void ready(void) {}

    // get result size without actual evaluation
    virtual int size_f() = 0;
    
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

    // estimate amount of calculations required to calculate gradients
    virtual int cost_mult_grad(int k_param, const matrix_type &B) {
        assert_param_dim(k_param);
        vector_type v = get_param(k_param);
        int n_row = v.size();
        int n_col = size_f();
        return n_row * n_col * B.shape(1);
    }
    
    virtual int cost_mult_by_grad(int k_param, const matrix_type &A) {
        assert_param_dim(k_param);
        vector_type v = get_param(k_param);
        int n_col = size_f();
        return A.shape(0) * A.shape(1) * n_col;
    }

    std::string name() const { return _name; }
    int n_params() const { return n_params; }
    int mult_opt_level(int k_param) const {
        return (int) _opt_level.size() > k_param? _opt_level[k_param].first: 0;
    }
    int mult_by_opt_level(int k_param) const {
        return (int) _opt_level.size() > k_param? _opt_level[k_param].second: 0;
    }

protected:
    std::string _name;
    int _n_params;

    // optimization level for jacobian matrix multiplications, in form like: {
    //     [0]: { A * D0, D0 * B }, [1]: { A * D1, D1 * B }, ...  }
    std::vector<std::pair<int, int> > _opt_level;
};

} // end namespace ddf

#endif /* OP_BASE_H */
