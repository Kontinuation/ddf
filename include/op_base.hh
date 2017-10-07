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
    
    math_op(const std::string &name, int n_params)
        : _name(name), _n_params(n_params) {
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

    // backpropagation
    virtual void bprop(
        int k_param, const vector_type &dy, vector_type &dx) = 0;

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

    std::string name() const { return _name; }
    int n_params() const { return n_params; }

protected:
    std::string _name;
    int _n_params;
};

} // end namespace ddf

#endif /* OP_BASE_H */
