#ifndef _COLLECT_VARS_H_
#define _COLLECT_VARS_H_

// Traverse to find out all variables in expression

#include "expr.hh"
#include <map>
#include <set>

namespace ddf {

template <typename numeric_type>
class collect_variable : public math_expr_visitor<numeric_type> {
public:
    virtual void apply(constant<numeric_type> *expr) {
        // ignore
    }
    virtual void apply(identity<numeric_type> *expr) {
        // ignore
    }
    
    virtual void apply(variable<numeric_type> *expr) {
        _vars[expr->_var] = expr;
    }
    
    virtual void apply(function_call<numeric_type> *expr) {
        for (auto &arg: expr->_args) {
            arg->apply(this);
        }
    }
    
    virtual void apply(addition<numeric_type> *expr) {
        expr->_a->apply(this);
        expr->_b->apply(this);
    }

    std::map<std::string, variable<numeric_type> *> &vars(void) {
        return _vars;
    }
    
private:
    std::map<std::string, variable<numeric_type> *> _vars;
};

} // end namespace ddf

#endif /* _COLLECT_VARS_H_ */
