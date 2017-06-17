#ifndef _CHECK_SIZE_H_
#define _CHECK_SIZE_H_

#include "expr.hh"

namespace ddf {

// Check if vector sizes are correct in specified expression
template <typename numeric_type>
class check_size : public math_expr_visitor<numeric_type> {
public:
    virtual void apply(constant<numeric_type> *expr) {
        _expr_size[expr] = expr->_v.size();
    }

    virtual void apply(identity<numeric_type> *expr) {
        _expr_size[expr] = expr->_size * expr->_size;
    }
    
    virtual void apply(variable<numeric_type> *expr) {
        _expr_size[expr] = expr->value().size();
    }
    
    virtual void apply(function_call<numeric_type> *expr) {
        int n_args = (int) expr->_args.size();
        for (int k = 0; k < n_args; k++) {
            auto &arg = expr->_args[k];
            int size_arg = size_of_expr(arg.get());
            expr->_op->prepare(k, vector<numeric_type>(size_arg));
        }
        _expr_size[expr] = expr->_op->size_f();
    }
    
    virtual void apply(dfunction_call<numeric_type> *expr) {
        expr->_d_arg->apply(this);
        int n_args = (int) expr->_args.size();
        for (int k = 0; k < n_args; k++) {
            auto &arg = expr->_args[k];
            int size_arg = size_of_expr(arg.get());
            expr->_op->prepare(k, vector<numeric_type>(size_arg));
        }
        _expr_size[expr] = expr->_op->size_f();
    }
    
    virtual void apply(addition<numeric_type> *expr) {
        int size_a = size_of_expr(expr->_a.get());
        int size_b = size_of_expr(expr->_b.get());
        if (size_a != size_b) {
            throw exception("lhs and rhs size does not match"
                " in addition expression");
        }
        _expr_size[expr] = size_a;
    }

    int size_of_expr(math_expr<numeric_type> *expr) {
        auto it = _expr_size.find(expr);
        if (it == _expr_size.end()) {
            expr->apply(this);
            it = _expr_size.find(expr);
            assert(("size of expression should have already be evaluated", 
                    it != _expr_size.end()));
        }
        return it->second;
    }
    
private:
    std::map<math_expr<numeric_type> *, int> _expr_size;
};

} // end namespace ddf

#endif /* _CHECK_SIZE_H_ */
