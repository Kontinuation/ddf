#ifndef EXPR_H
#define EXPR_H

#include "nd_array.hh"
#include <algorithm>
#include <cstring>

namespace ddf {

// Deep Learning requires the minimization of target loss function composed of
// various very basic building blocks, we designed a syntax tree for describing
// such functions, so that we can perform automatic differentiation in order to
// feed it to an iterative optimization algorithm (such as SGD)

template <typename _numeric_type>
struct math_expr {
    virtual math_expr *derivative(const char *var, int dim) = 0;
    virtual void eval(vector<_numeric_type> &y) = 0;
    virtual math_expr *clone(void) const = 0;
    virtual std::string to_string() const = 0;
};

// forward declarations for recursive datatypes
template <typename numeric_type> struct addition;
template <typename numeric_type> struct multiplication;
template <typename numeric_type> struct constant;
template <typename numeric_type> struct variable;
template <typename numeric_type> struct function_call;
template <typename numeric_type> struct dfunction_call;

// dc/dx = 0
template <typename numeric_type>
struct constant: math_expr<numeric_type> {
    constant(const vector<numeric_type> &v): _v(v) {
    }

    math_expr<numeric_type> *derivative(const char *, int) {
        return nullptr;
    }

    void eval(vector<numeric_type> &y) {
        y = _v.clone();
    }

    math_expr<numeric_type> *clone(void) const {
        return new constant(_v);
    }

    std::string to_string() const {
        return _v.to_string();
    }

    vector<numeric_type> _v;
};

// dx/dx = 1
template <typename numeric_type>
struct variable: math_expr<numeric_type> {
    variable(const char *var, const vector<numeric_type> &val)
        : _val(val) {
        strncpy(_var, var, sizeof _var);
        _var[(sizeof _var) - 1] = '\0';
    }

    void set_value(const vector<numeric_type> &val) {
        _val = val.clone();             // _val.copy_from(val);
    }

    math_expr<numeric_type> *derivative(const char *var, int dim) {
        if (!strcmp(var, _var)) {
            int n = _val.size();
            assert(("dimension boundary check", dim < n));
            vector<numeric_type> c(n);
            c[dim] = 1;
            return new constant<numeric_type>(c);
        } else {
            return nullptr;
        }
    }

    void eval(vector<numeric_type> &y) {
        y = _val.clone();          // y.copy_from(x);
    }

    math_expr<numeric_type> *clone(void) const {
        return new variable(_var, _val.clone());
    }

    std::string to_string() const {
        return _var;
    }

    char _var[64];
    vector<numeric_type> _val;
};

// df(arg)/dx = darg/dx * df(arg)/dx
template <typename numeric_type>
struct function_call: math_expr<numeric_type> {
    function_call(math_op<numeric_type> *op, math_expr<numeric_type> *arg)
        : _op(op), _arg(arg) {
    }

    math_expr<numeric_type> *derivative(const char *var, int dim) {
        math_expr<numeric_type> *d_arg = _arg->derivative(var, dim);
        if (d_arg) {
            return new multiplication<numeric_type>(
                new dfunction_call<numeric_type>(_op, _arg, dim),
                d_arg);
        } else {
            return nullptr;
        }
    }

    void eval(vector<numeric_type> &y) {
        vector<numeric_type> x(0);
        _arg->eval(x);
        _op->f_x(x, y);
    }

    math_expr<numeric_type> *clone(void) const {
        return new function_call(_op, _arg->clone());
    }

    std::string to_string() const {
        return std::string(_op->_name) + "(" + _arg->to_string() + ")";
    }

    math_op<numeric_type> *_op;
    math_expr<numeric_type> *_arg;
};

// ddf(arg)/dx is a invalid term
template <typename numeric_type>
struct dfunction_call: math_expr<numeric_type> {
    dfunction_call(
        math_op<numeric_type> *op, math_expr<numeric_type> *arg, int dim)
        : _op(op), _arg(arg), _dim(dim) {
    }

    math_expr<numeric_type> *derivative(const char *var, int dim) {
        assert(("second-order derivatives could not be evaluated", false));
        return nullptr;
    }

    void eval(vector<numeric_type> &y) {
        vector<numeric_type> x(0);
        _arg->eval(x);
        _op->df_x(x, y, _dim);
    }

    math_expr<numeric_type> *clone(void) const {
        return new dfunction_call(_op, _arg->clone(), _dim);
    }

    std::string to_string() const {
        return ("d " + std::string(_op->_name)) + "(" + _arg->to_string() + ")";
    }
    
    math_op<numeric_type> *_op;
    math_expr<numeric_type> *_arg;
    int _dim;
};

// d(a + b)/dx = da/dx + db/dx
template <typename numeric_type>
struct addition: math_expr<numeric_type> {
    addition(math_expr<numeric_type> *a, math_expr<numeric_type> *b)
        : _a(a), _b(b) {
    }

    math_expr<numeric_type> *derivative(const char *var, int dim) {
        auto d_a = _a->derivative(var, dim);
        auto d_b = _b->derivative(var, dim);
        if (d_a && d_b) {
            return new addition<numeric_type>(d_a, d_b);
        } else if (d_a) {
            return d_a;
        } else if (d_b) {
            return d_b;
        } else {
            return nullptr;
        }
    }

    void eval(vector<numeric_type> &y) {
        vector<numeric_type> x(0);
        _a->eval(x);
        _b->eval(y);
        y += x;
    }

    math_expr<numeric_type> *clone(void) const {
        return new addition(_a->clone(), _b->clone());
    }

    std::string to_string() const {
        return "(" + _a->to_string() + " + " + _b->to_string() + ")";
    }

    math_expr<numeric_type> *_a;
    math_expr<numeric_type> *_b;
};

// d (a * b)/dx = da/dx*b + db/dx*a
template <typename numeric_type>
struct multiplication: math_expr<numeric_type> {
    multiplication(math_expr<numeric_type> *a, math_expr<numeric_type> *b)
        : _a(a), _b(b) {
    }

    math_expr<numeric_type> *derivative(const char *var, int dim) {
        auto d_a = _a->derivative(var, dim);
        auto d_b = _b->derivative(var, dim);
        if (d_a && d_b) {
            return new addition<numeric_type>(
                new multiplication<numeric_type>(d_a, _b->clone()),
                new multiplication<numeric_type>(d_b, _a->clone()));
        } else if (d_a) {
            return new multiplication<numeric_type>(d_a, _b->clone());
        } else if (d_b) {
            return new multiplication<numeric_type>(d_b, _a->clone());
        } else {
            return nullptr;
        }
    }

    void eval(vector<numeric_type> &y) {
        vector<numeric_type> x(0);
        _a->eval(x);
        _b->eval(y);
        y *= x;
    }

    math_expr<numeric_type> *clone(void) const {
        return new multiplication(_a->clone(), _b->clone());
    }

    std::string to_string() const {
        return "(" + _a->to_string() + " * " + _b->to_string() + ")";
    }    
    
    math_expr<numeric_type> *_a;
    math_expr<numeric_type> *_b;
};

} // end namespace ddf

#endif /* EXPR_H */
