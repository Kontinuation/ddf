#ifndef EXPR_H
#define EXPR_H

// Deep Learning requires the minimization of target loss function composed of
// various very basic building blocks, we designed a syntax tree for describing
// such functions, so that we can perform automatic differentiation in order to
// feed it to an iterative optimization algorithm (such as SGD)

#include "nd_array.hh"
#include "array_opt.hh"
#include <vector>
#include <algorithm>
#include <cstring>

namespace ddf {

enum class expr_typeid {
    CONSTANT,
    VARIABLE,
    IDENTITY,
    FUNCTION_CALL,
    DFUNCTION_CALL,
    ADDITION,
    MULTIPLICATION,
};

// forward declarations for recursive datatypes
template <typename numeric_type> struct constant;
template <typename numeric_type> struct variable;
template <typename numeric_type> struct identity;
template <typename numeric_type> struct function_call;
template <typename numeric_type> struct dfunction_call;
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
    virtual void apply(dfunction_call<numeric_type> *expr) = 0;
    virtual void apply(addition<numeric_type> *expr) = 0;
};

template <typename _numeric_type>
class math_expr {
public:
    math_expr(expr_typeid type) : type(type) { }
    virtual ~math_expr(void) = default;
    virtual math_expr *derivative(const std::string &var) = 0;
    virtual math_expr *clone(void) const = 0;
    virtual std::string to_string() const = 0;
    virtual void apply(math_expr_visitor<_numeric_type> *visitor) = 0;
    virtual void eval(vector<_numeric_type> &y) = 0;
    virtual void grad(matrix<_numeric_type> &m) {
        throw exception("cannot calculate grad on this expr");
    }
    expr_typeid type;
    vector<_numeric_type> delta;

private:
    DISABLE_COPY_AND_ASSIGN(math_expr);
};

// utility function for cloning multiple expressions at once
template <typename numeric_type>
std::vector<std::shared_ptr<math_expr<numeric_type> > >
clone_exprs(const std::vector<std::shared_ptr<math_expr<numeric_type> > > &exprs) {
    std::vector<std::shared_ptr<math_expr<numeric_type> > > ret;
    size_t n_exprs = exprs.size();
    ret.reserve(n_exprs);
    for (size_t k = 0; k < n_exprs; k++) {
        ret.push_back(std::shared_ptr<math_expr<numeric_type> >(
                exprs[k]->clone())
            );
    }
    return ret;
}

// dc/dx = 0
template <typename numeric_type>
struct constant: math_expr<numeric_type> {
    constant(const vector<numeric_type> &v)
        : math_expr<numeric_type>(expr_typeid::CONSTANT), _v(v) {
    }

    math_expr<numeric_type> *derivative(const std::string &) {
        return nullptr;
    }

    void eval(vector<numeric_type> &y) {
        y = _v;
    }

    math_expr<numeric_type> *clone(void) const {
        return new constant(_v);
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

    math_expr<numeric_type> *derivative(const std::string &) {
        return nullptr;
    }

    void eval(vector<numeric_type> &) {
        assert(("could not evaluate identity as vector", false));
    }

    math_expr<numeric_type> *clone(void) const {
        return new identity(_size);
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

    math_expr<numeric_type> *derivative(const std::string &var) {
        if (!_var.compare(var)) {
            return new identity<numeric_type>(_val.size());
        } else {
            return nullptr;
        }
    }

    void eval(vector<numeric_type> &y) {
        y = _val;
    }

    math_expr<numeric_type> *clone(void) const {
        return new variable(_var, _val);
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

    math_expr<numeric_type> *derivative(const std::string &var) {
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

    void apply(math_expr_visitor<numeric_type> *visitor) {
        visitor->apply(this);
    }

    math_op<numeric_type> *_op;
    std::vector<shared_math_expr_ptr> _args;
    std::vector<vector<numeric_type> > _xs;
    std::vector<vector<numeric_type> > _dxs;
};

// ddf(arg)/dx is a invalid term
template <typename numeric_type>
struct dfunction_call: math_expr<numeric_type> {
    typedef std::shared_ptr<math_expr<numeric_type> > shared_math_expr_ptr;
    
    dfunction_call(
        math_op<numeric_type> *op, std::vector<shared_math_expr_ptr> args,
        int k_param, math_expr<numeric_type> *d_arg)
        : math_expr<numeric_type>(expr_typeid::DFUNCTION_CALL),
          _op(op), _args(args), _k_param(k_param), _d_arg(d_arg) {
        _xs.resize(args.size());
    }

    math_expr<numeric_type> *derivative(const std::string &var) {
        assert(("second-order derivatives could not be evaluated", false));
        return nullptr;
    }

    void eval(vector<numeric_type> &y) {
        assert(("could not evaluate derivative as vector", false));
    }

    void grad(matrix<numeric_type> &m) {
        this->prepare();

        if (_d_arg->type != expr_typeid::IDENTITY) {

#ifndef DISABLE_FAST_MULT_GRAD
            // chain rule: D_f * D_g, we will look for oppotunities for
            // optimization rather than directly calculating the matrix
            // multiplication
            
            int mult_opt = _op->mult_opt_level(_k_param);
            int mult_by_opt = 0;
            auto d_arg = static_cast<dfunction_call<numeric_type> *>(_d_arg.get());
            if (_d_arg->type == expr_typeid::DFUNCTION_CALL) {
                if (d_arg->_d_arg->type == expr_typeid::IDENTITY) {
                    mult_by_opt = d_arg->_op->mult_by_opt_level(d_arg->_k_param);
                }
            }

            logging::debug("analyzing %s, mult_opt: %d, mult_by_opt: %d", 
                this->to_string().c_str(), mult_opt, mult_by_opt);

            if (mult_opt > 0 || mult_by_opt > 0) {
                // optimized case
                if (mult_opt > mult_by_opt) {
                    _d_arg->grad(_D_g);
                    _op->mult_grad(_k_param, _D_g, m);
                } else {
                    _op->Df(_k_param, _D_f);
                    d_arg->prepare();
                    d_arg->_op->mult_by_grad(d_arg->_k_param, _D_f, m);
                }
            } else {
                // fallback case
                _op->Df(_k_param, _D_f);
                _d_arg->grad(_D_g);

                logging::debug("fallback %s, Df: (%d,%d); Dg: (%d,%d)",
                    this->to_string().c_str(), 
                    _D_f.shape(0), _D_f.shape(1),
                    _D_g.shape(0), _D_g.shape(1));

                _D_f.mult(_D_g, m); // m = _D_f * _D_g;
            }
#else
            _op->Df(_k_param, _D_f);
            _d_arg->grad(_D_g);
            _D_f.mult(_D_g, m); // m = _D_f * _D_g;
#endif
        } else {
            _op->Df(_k_param, m);
        }
    }

    math_expr<numeric_type> *clone(void) const {
        return new dfunction_call(
            _op, clone_exprs(_args), _k_param, _d_arg->clone());
    }

    std::string to_string() const {
        std::string str_args = "";
        size_t n_args = _args.size();
        for (size_t k = 0; k < n_args; k++) {
            str_args += _args[k]->to_string();
            if (k != n_args - 1) str_args += ",";
        }
        return ("D_" + std::to_string(_k_param) +  " " +  
            std::string(_op->name())) + 
            "(" + str_args + ") " + _d_arg->to_string();
    }

    // evaluate args for grad
    void prepare(void) {
        size_t n_args = _args.size();
        for (size_t k = 0; k < n_args; k++) {
            _args[k]->eval(_xs[k]);
            _op->prepare(k, _xs[k]);
        }
        _op->ready();
    }

    void apply(math_expr_visitor<numeric_type> *visitor) {
        visitor->apply(this);
    }
    
    math_op<numeric_type> *_op;
    std::vector<shared_math_expr_ptr> _args;
    int _k_param;
    shared_math_expr_ptr _d_arg;
    std::vector<vector<numeric_type> > _xs;
    matrix<numeric_type> _D_f;
    matrix<numeric_type> _D_g;
};

// d(a + b)/dx = da/dx + db/dx
template <typename numeric_type>
struct addition: math_expr<numeric_type> {
    typedef std::shared_ptr<math_expr<numeric_type> > shared_math_expr_ptr;
    
    addition(math_expr<numeric_type> *a, math_expr<numeric_type> *b)
        : math_expr<numeric_type>(expr_typeid::ADDITION),
          _a(a), _b(b) {
    }

    math_expr<numeric_type> *derivative(const std::string &var) {
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
        _a->eval(_ya);
        _b->eval(_yb);
        y.copy_from(_ya);
        y += _yb;
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

    void apply(math_expr_visitor<numeric_type> *visitor) {
        visitor->apply(this);
    }

    shared_math_expr_ptr _a;
    shared_math_expr_ptr _b;
    vector<numeric_type> _ya;
    vector<numeric_type> _yb;
    matrix<numeric_type> _D_b;
};

} // end namespace ddf

#endif /* EXPR_H */