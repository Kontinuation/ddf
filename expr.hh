#ifndef EXPR_H
#define EXPR_H

#include "nd_array.hh"
#include "array_opt.hh"
#include <vector>
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

// utility function for cloning multiple expressions at once
template <typename numeric_type>
std::vector<math_expr<numeric_type> *> 
clone_exprs(const std::vector<math_expr<numeric_type> *> &exprs) {
    std::vector<math_expr<numeric_type> *> ret;
    size_t n_exprs = exprs.size();
    ret.reserve(n_exprs);
    for (size_t k = 0; k < n_exprs; k++) {
        ret.push_back(exprs[k]->clone());
    }
    return ret;
}

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
    variable(const std::string &var, const vector<numeric_type> &val)
        : _var(var), _val(val) {
        this->type = expr_type::VARIABLE;
    }

    void set_value(const vector<numeric_type> &val) {
        _val = val.clone();             // _val.copy_from(val);
    }

    math_expr<numeric_type> *derivative(const char *var) {
        if (!_var.compare(var)) {
            return new identity<numeric_type>(_val.size());
        } else {
            return nullptr;
        }
    }

    void eval(vector<numeric_type> &y) {
        y.copy_from(_val);      // y = _val.clone();
    }

    math_expr<numeric_type> *clone(void) const {
        return new variable(_var, _val);
    }

    std::string to_string() const {
        return _var;
    }

    std::string _var;
    vector<numeric_type> _val;
};

// df(arg)/dx = darg/dx * df(arg)/dx
template <typename numeric_type>
struct function_call: math_expr<numeric_type> {
    function_call(math_op<numeric_type> *op, 
        math_expr<numeric_type> *arg0 = nullptr, 
        math_expr<numeric_type> *arg1 = nullptr, 
        math_expr<numeric_type> *arg2 = nullptr) 
        : _op(op) {
        this->type = expr_type::FUNCTION_CALL;
        do {
            if (!arg0) break; else _args.push_back(arg0);
            if (!arg1) break; else _args.push_back(arg1);
            if (!arg2) break; else _args.push_back(arg2);
        } while(0);
        _xs.resize(_args.size());        
    }

    function_call(math_op<numeric_type> *op, 
        const std::vector<math_expr<numeric_type> *> &args)
        : _op(op), _args(args) {
        this->type = expr_type::FUNCTION_CALL;
        _xs.resize(args.size());
    }

    math_expr<numeric_type> *derivative(const char *var) {
        math_expr<numeric_type> *d_arg = nullptr;
        size_t n_args = _args.size();
        int k_param = -1;
        for (size_t k = 0; k < n_args; k++) {
            math_expr<numeric_type> *d_argk = _args[k]->derivative(var);
            if (d_argk != nullptr) {
                if (d_arg == nullptr) {
                    d_arg = d_argk;
                    k_param = (int) k;
                } else {
                    assert(("invalid expression", false));
                }
            }
        }        
        
        if (d_arg) {
            return new dfunction_call<numeric_type>(
                _op, clone_exprs(_args), k_param, d_arg);
        } else {
            return nullptr;
        }
    }

    void eval(vector<numeric_type> &y) {
        size_t n_args = _args.size();
        for (size_t k = 0; k < n_args; k++) {
            _args[k]->eval(_xs[k]);
            _op->prepare(k, _xs[k]);
        }
        _op->ready();
        _op->f(y);
    }

    math_expr<numeric_type> *clone(void) const {
        return new function_call(_op, clone_exprs(_args));
    }

    std::string to_string() const {
        std::string str_args = "";
        size_t n_args = _args.size();
        for (size_t k = 0; k < n_args; k++) {
            str_args += _args[k]->to_string();
            if (k != n_args - 1) str_args += ",";
        }
        return _op->name() + "(" + str_args + ")";
    }

    math_op<numeric_type> *_op;
    std::vector<math_expr<numeric_type> *> _args;
    std::vector<vector<numeric_type> > _xs;
};

// ddf(arg)/dx is a invalid term
template <typename numeric_type>
struct dfunction_call: math_expr<numeric_type> {
    dfunction_call(
        math_op<numeric_type> *op, std::vector<math_expr<numeric_type> *> args,
        int k_param, math_expr<numeric_type> *d_arg)
        : _op(op), _args(args), _k_param(k_param), _d_arg(d_arg) {
        this->type = expr_type::DFUNCTION_CALL;
        _xs.resize(args.size());
    }

    math_expr<numeric_type> *derivative(const char *var) {
        assert(("second-order derivatives could not be evaluated", false));
        return nullptr;
    }

    void eval(vector<numeric_type> &y) {
        assert(("could not evaluate derivative as vector", false));
    }

    void grad(matrix<numeric_type> &m) {
        // evaluate args for op
        size_t n_args = _args.size();
        for (size_t k = 0; k < n_args; k++) {
            _args[k]->eval(_xs[k]);
            _op->prepare(k, _xs[k]);
        }
        _op->ready();

        if (_d_arg->type != expr_type::IDENTITY) {
            // chain rule
            _op->Df(_k_param, _D_f);
            _d_arg->grad(_D_g);
            _D_f.mult(_D_g, m); // m = _D_f * _D_g;            
        } else {
            _op->Df(_k_param, m);
        }
    }

    math_expr<numeric_type> *clone(void) const {
        return new dfunction_call(_op, clone_exprs(_args), _k_param, _d_arg);
    }

    std::string to_string() const {
        std::string str_args = "";
        size_t n_args = _args.size();
        for (size_t k = 0; k < n_args; k++) {
            str_args += _args[k]->to_string();
            if (k != n_args - 1) str_args += ",";
        }
        return ("D " + std::string(_op->name())) + "(" + str_args + ") "
            + _d_arg->to_string();
    }
    
    math_op<numeric_type> *_op;
    std::vector<math_expr<numeric_type> *> _args;
    int _k_param;
    math_expr<numeric_type> *_d_arg;
    std::vector<vector<numeric_type> > _xs;
    matrix<numeric_type> _D_f;
    matrix<numeric_type> _D_g;
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
        _a->eval(_x);
        _b->eval(y);
        y += _x;
    }

    void grad(matrix<numeric_type> &m) {
        _a->grad(m);
        _b->grad(_D_b);
        m += _D_b;
    }

    math_expr<numeric_type> *clone(void) const {
        return new addition(_a->clone(), _b->clone());
    }

    std::string to_string() const {
        return "(" + _a->to_string() + " + " + _b->to_string() + ")";
    }

    math_expr<numeric_type> *_a;
    math_expr<numeric_type> *_b;
    vector<numeric_type> _x;
    matrix<numeric_type> _D_b;
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
