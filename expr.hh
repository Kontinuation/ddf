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

enum class expr_type {
    CONSTANT,
    VARIABLE,
    IDENTITY,
    FUNCTION_CALL,
    DFUNCTION_CALL,
    ADDITION,
    MULTIPLICATION,
};


template <typename _numeric_type>
struct math_expr {
    virtual math_expr *derivative(const char *var) = 0;
    virtual math_expr *clone(void) const = 0;
    virtual std::string to_string() const = 0;
    virtual void eval(vector<_numeric_type> &y) = 0;
    virtual void grad(matrix<_numeric_type> &m) {
        assert(("cannot calculate grad on this expr", false));
    }
    expr_type type;
};

// forward declarations for recursive datatypes
template <typename numeric_type> struct constant;
template <typename numeric_type> struct variable;
template <typename numeric_type> struct identity;
template <typename numeric_type> struct function_call;
template <typename numeric_type> struct dfunction_call;
template <typename numeric_type> struct addition;
template <typename numeric_type> struct multiplication;

// dc/dx = 0
template <typename numeric_type>
struct constant: math_expr<numeric_type> {
    constant(const vector<numeric_type> &v): _v(v) {
        this->type = expr_type::CONSTANT;
    }

    math_expr<numeric_type> *derivative(const char *) {
        return nullptr;
    }

    void eval(vector<numeric_type> &y) {
        y.copy_from(_v);
    }

    math_expr<numeric_type> *clone(void) const {
        return new constant(_v);
    }

    std::string to_string() const {
        return _v.to_string();
    }

    vector<numeric_type> _v;
};

template <typename numeric_type>
struct identity: math_expr<numeric_type> {
    identity(int size): _size(size) {
        this->type = expr_type::IDENTITY;
    }

    math_expr<numeric_type> *derivative(const char *) {
        return nullptr;
    }

    void eval(vector<numeric_type> &) {
        assert(("could not evaluate identity as vector", false));
    }

    void grad(matrix<numeric_type> &m) {
        m.resize(_size, _size);
        m.fill(0);
        for (int i = 0; i < _size; i++) {
            m(i,i) = 1;
        }
    }

    math_expr<numeric_type> *clone(void) const {
        return new identity(_size);
    }

    std::string to_string() const {
        return "I";
    }

    int _size;
};

// dx/dx = 1
template <typename numeric_type>
struct variable: math_expr<numeric_type> {
    variable(const char *var, const vector<numeric_type> &val)
        : _val(val) {
        this->type = expr_type::VARIABLE;
        strncpy(_var, var, sizeof _var);
        _var[(sizeof _var) - 1] = '\0';
    }

    void set_value(const vector<numeric_type> &val) {
        _val = val.clone();             // _val.copy_from(val);
    }

    math_expr<numeric_type> *derivative(const char *var) {
        if (!strcmp(var, _var)) {
            return new identity<numeric_type>(_val.size());
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
        this->type = expr_type::FUNCTION_CALL;
    }

    math_expr<numeric_type> *derivative(const char *var) {
        math_expr<numeric_type> *d_arg = _arg->derivative(var);
        if (d_arg) {
            return new dfunction_call<numeric_type>(_op, _arg->clone(), d_arg);
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
        math_op<numeric_type> *op, math_expr<numeric_type> *arg,
        math_expr<numeric_type> *d_arg)
        : _op(op), _arg(arg), _d_arg(d_arg) {
        this->type = expr_type::DFUNCTION_CALL;
    }

    math_expr<numeric_type> *derivative(const char *var) {
        assert(("second-order derivatives could not be evaluated", false));
        return nullptr;
    }

    void eval(vector<numeric_type> &y) {
        assert(("could not evaluate derivative as vector", false));
    }

    void grad(matrix<numeric_type> &m) {
        vector<numeric_type> x(0);
        if (_d_arg->type != expr_type::IDENTITY) {
            matrix<numeric_type> D_f(0, 0);
            matrix<numeric_type> D_g(0, 0);
            _arg->eval(x);
            _op->Df_x(x, D_f);
            _d_arg->grad(D_g);
            D_f.mult(D_g, m);       // m = D_f * D_g;
        } else {
            _arg->eval(x);
            _op->Df_x(x, m);
        }
    }

    math_expr<numeric_type> *clone(void) const {
        return new dfunction_call(_op, _arg->clone(), _d_arg);
    }

    std::string to_string() const {
        return ("D " + std::string(_op->_name)) + "(" + _arg->to_string() + ") "
            + _d_arg->to_string();
    }
    
    math_op<numeric_type> *_op;
    math_expr<numeric_type> *_arg;
    math_expr<numeric_type> *_d_arg;
};

// d(a + b)/dx = da/dx + db/dx
template <typename numeric_type>
struct addition: math_expr<numeric_type> {
    addition(math_expr<numeric_type> *a, math_expr<numeric_type> *b)
        : _a(a), _b(b) {
        this->type = expr_type::ADDITION;
    }

    math_expr<numeric_type> *derivative(const char *var) {
        auto d_a = _a->derivative(var);
        auto d_b = _b->derivative(var);

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

    void grad(matrix<numeric_type> &m) {
        matrix<numeric_type> D_b(0, 0);
        _a->grad(m);
        _b->grad(D_b);
        m += D_b;
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
        this->type = expr_type::MULTIPLICATION;
    }

    math_expr<numeric_type> *derivative(const char *var) {
        auto d_a = _a->derivative(var);
        auto d_b = _b->derivative(var);
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

    void grad(matrix<numeric_type> &m) {
        assert(("grad of f * g is not implemented yet", false));
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
