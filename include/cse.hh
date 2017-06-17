#ifndef CSE_H
#define CSE_H

// Common subexpression elimination, we reuse the result of evaluating common
// subexpressions to eliminate repreated calculation introduced by forward
// automatic differentiation

#include "expr.hh"
#include <map>

namespace ddf {

template <typename numeric_type>
class common_subexpr_elim : public math_expr_visitor<numeric_type> {
public:
    virtual void apply(constant<numeric_type> *expr) {
        _cse[expr->to_string()] = expr;
    }
    virtual void apply(identity<numeric_type> *expr) {
        _cse[expr->to_string()] = expr;
    }
    
    virtual void apply(variable<numeric_type> *expr) {
        _cse[expr->to_string()] = expr;
    }
    
    virtual void apply(function_call<numeric_type> *expr) {
        _cse[expr->to_string()] = expr;
        for (auto &arg: expr->_args) {
            auto it = _cse.find(arg->to_string());
            if (it != _cse.end()) {
                arg.reset(it->second, null_deleter);
            } else {
                arg->apply(this);
            }
        }
    }
    
    virtual void apply(dfunction_call<numeric_type> *expr) {
        _cse[expr->to_string()] = expr;
        auto it = _cse.find(expr->_d_arg->to_string());
        if (it != _cse.end()) {
            expr->_d_arg.reset(it->second, null_deleter);
        } else {
            expr->_d_arg->apply(this);
        }

        for (auto &arg: expr->_args) {
            auto it = _cse.find(arg->to_string());
            if (it != _cse.end()) {
                arg.reset(it->second, null_deleter);
            } else {
                arg->apply(this);
            }
        }
    }
    
    virtual void apply(addition<numeric_type> *expr) {
        _cse[expr->to_string()] = expr;
        auto it = _cse.find(expr->_a->to_string());
        if (it != _cse.end()) {
            expr->_a.reset(it->second, null_deleter);
        } else {
            expr->_a->apply(this);
        }
        it = _cse.find(expr->_b->to_string());
        if (it != _cse.end()) {
            expr->_b.reset(it->second, null_deleter);
        } else {
            expr->_b->apply(this);
        }
    }

    void apply(std::shared_ptr<ddf::math_expr<numeric_type> > &expr) {
        auto it = _cse.find(expr->to_string());
        if (it != _cse.end()) {
            expr.reset(it->second, null_deleter);
        } else {
            expr->apply(this);
        }
    }

    static void null_deleter(math_expr<numeric_type> *) { }

private:
    std::map<std::string, math_expr<numeric_type> *> _cse;
};

} // end namespace ddf

#endif /* CSE_H */
