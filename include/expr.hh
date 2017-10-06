#ifndef EXPR_H
#define EXPR_H

// Deep Learning requires the minimization of target loss function composed of
// various very basic building blocks, we designed a syntax tree for describing
// such functions, so that we can perform automatic differentiation in order to
// feed it to an iterative optimization algorithm (such as SGD)

#include "nd_array.hh"
#include <vector>
#include <algorithm>
#include <cstring>

namespace ddf {

enum class expr_typeid {
    CONSTANT,
    VARIABLE,
    IDENTITY,
    FUNCTION_CALL,
    ADDITION,
    MULTIPLICATION,
};

// forward declarations for recursive datatypes
template <typename numeric_type> struct constant;
template <typename numeric_type> struct variable;
template <typename numeric_type> struct identity;
template <typename numeric_type> struct function_call;
template <typename numeric_type> struct addition;

// visitor class for applying operation on sub expressions
template <typename numeric_type>
class math_expr_visitor {
public:
    virtual ~math_expr_visitor(void) = default;
    virtual void apply(constant<numeric_type> *expr) = 0;
    virtual void apply(variable<numeric_type> *expr) = 0;
    virtual void apply(identity<numeric_type> *expr) = 0;
    virtual void apply(function_call<numeric_type> *expr) = 0;
    virtual void apply(addition<numeric_type> *expr) = 0;
};

template <typename _numeric_type>
class math_expr {
public:
    math_expr(expr_typeid type) : type(type) { }
    virtual ~math_expr(void) = default;
    virtual std::string to_string() const = 0;
    virtual void apply(math_expr_visitor<_numeric_type> *visitor) = 0;
    virtual void eval(vector<_numeric_type> &y) = 0;
    expr_typeid type;
    vector<_numeric_type> delta;

private:
    DISABLE_COPY_AND_ASSIGN(math_expr);
};

// dc/dx = 0
template <typename numeric_type>
struct constant: math_expr<numeric_type> {
    constant(const vector<numeric_type> &v)
        : math_expr<numeric_type>(expr_typeid::CONSTANT), _v(v) {
    }

    void eval(vector<numeric_type> &y) {
        y = _v;
    }

    std::string to_string() const {
        return _v.to_string();
    }

    void apply(math_expr_visitor<numeric_type> *visitor) {
        visitor->apply(this);
    }

    vector<numeric_type> _v;
};

template <typename numeric_type>
struct identity: math_expr<numeric_type> {
    identity(int size)
        : math_expr<numeric_type>(expr_typeid::IDENTITY), 
          _size(size) {
    }

    void eval(vector<numeric_type> &) {
        assert(("could not evaluate identity as vector", false));
    }

    std::string to_string() const {
        return "I";
    }

    void apply(math_expr_visitor<numeric_type> *visitor) {
        visitor->apply(this);
    }

    int _size;
};

// dx/dx = 1
template <typename numeric_type>
struct variable: math_expr<numeric_type> {
    variable(const std::string &var, const vector<numeric_type> &val)
        : math_expr<numeric_type>(expr_typeid::VARIABLE),
          _var(var), _val(val) {
    }

    vector<numeric_type> &value(void) {
        return _val;
    }

    void set_value(const vector<numeric_type> &val) {
        _val = val;
    }

    void eval(vector<numeric_type> &y) {
        y = _val;
    }

    std::string to_string() const {
        return _var;
    }

    void apply(math_expr_visitor<numeric_type> *visitor) {
        visitor->apply(this);
    }

    std::string _var;
    vector<numeric_type> _val;
};

// df(arg)/dx = darg/dx * df(arg)/dx
template <typename numeric_type>
struct function_call: math_expr<numeric_type> {
    typedef std::shared_ptr<math_expr<numeric_type> > shared_math_expr_ptr;
    
    function_call(math_op<numeric_type> *op, 
        math_expr<numeric_type> *arg0 = nullptr, 
        math_expr<numeric_type> *arg1 = nullptr, 
        math_expr<numeric_type> *arg2 = nullptr) 
        : math_expr<numeric_type>(expr_typeid::FUNCTION_CALL),
          _op(op) {
        do {
            if (!arg0) break; else _args.push_back(shared_math_expr_ptr(arg0));
            if (!arg1) break; else _args.push_back(shared_math_expr_ptr(arg1));
            if (!arg2) break; else _args.push_back(shared_math_expr_ptr(arg2));
        } while(0);
        _xs.resize(_args.size());
        _dxs.resize(_args.size());
    }

    function_call(math_op<numeric_type> *op, 
        const std::vector<shared_math_expr_ptr> &args)
        : math_expr<numeric_type>(expr_typeid::FUNCTION_CALL),
          _op(op), _args(args) {
        _xs.resize(args.size());
        _dxs.resize(_args.size());
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

    std::string to_string() const {
        std::string str_args = "";
        size_t n_args = _args.size();
        for (size_t k = 0; k < n_args; k++) {
            str_args += _args[k]->to_string();
            if (k != n_args - 1) str_args += ",";
        }
        return _op->name() + "(" + str_args + ")";
    }

    void apply(math_expr_visitor<numeric_type> *visitor) {
        visitor->apply(this);
    }

    std::unique_ptr<math_op<numeric_type> > _op;
    std::vector<shared_math_expr_ptr> _args;
    std::vector<vector<numeric_type> > _xs;
    std::vector<vector<numeric_type> > _dxs;
};

// d(a + b)/dx = da/dx + db/dx
template <typename numeric_type>
struct addition: math_expr<numeric_type> {
    typedef std::shared_ptr<math_expr<numeric_type> > shared_math_expr_ptr;
    
    addition(math_expr<numeric_type> *a, math_expr<numeric_type> *b)
        : math_expr<numeric_type>(expr_typeid::ADDITION),
          _a(a), _b(b) {
    }

    void eval(vector<numeric_type> &y) {
        _a->eval(_ya);
        _b->eval(_yb);
        y.copy_from(_ya);
        y += _yb;
    }

    std::string to_string() const {
        return "(" + _a->to_string() + " + " + _b->to_string() + ")";
    }

    void apply(math_expr_visitor<numeric_type> *visitor) {
        visitor->apply(this);
    }

    shared_math_expr_ptr _a;
    shared_math_expr_ptr _b;
    vector<numeric_type> _ya;
    vector<numeric_type> _yb;
};

} // end namespace ddf

#endif /* EXPR_H */
