#ifndef _IGNITION_H_
#define _IGNITION_H_

// Do some initialization work with exprs

namespace ddf {

// Clear accumulated delta on nodes in specified expression, should be called
// before performing backpropagation
template <typename numeric_type>
class reset_delta : public math_expr_visitor<numeric_type> {
    virtual void apply(constant<numeric_type> *expr) {
        expr->delta.fill(0);
    }
    virtual void apply(identity<numeric_type> *expr) {
        expr->delta.fill(0);
    }
    virtual void apply(variable<numeric_type> *expr) {
        expr->delta.fill(0);
    }

    virtual void apply(function_call<numeric_type> *expr) {
        expr->delta.fill(0);
        size_t n_args = expr->_args.size();
        for (size_t k = 0; k < n_args; k++) {
            expr->_args[k]->apply(this);
        }
    }
        
    virtual void apply(addition<numeric_type> *expr) {
        expr->delta.fill(0);
        expr->_a->apply(this);
        expr->_b->apply(this);
    }
};

// Reset operator which has internal states, such as dropout (stores dropout
// masks inside the operator), etc.
template <typename numeric_type>
class reset_op : public math_expr_visitor<numeric_type> {
public:
    reset_op(mode m): _m(m) {
    }

    virtual void apply(constant<numeric_type> *expr) {}
    virtual void apply(identity<numeric_type> *expr) {}
    virtual void apply(variable<numeric_type> *expr) {}

    virtual void apply(function_call<numeric_type> *expr) {
        expr->reset_op(_m);
        size_t n_args = expr->_args.size();
        for (size_t k = 0; k < n_args; k++) {
            expr->_args[k]->apply(this);
        }
    }

    virtual void apply(addition<numeric_type> *expr) {
        expr->_a->apply(this);
        expr->_b->apply(this);
    }

private:
    mode _m;
};

// Easier to use functions so that you don't need to construct a visitor object
// each time you want to reset your expression

template <typename numeric_type>
void reset_expr_delta(math_expr<numeric_type> *expr) {
    reset_delta<numeric_type> rd;
    expr->apply(&rd);
}

template <typename numeric_type>
void set_expr_working_mode(math_expr<numeric_type> *expr, mode m) {
    reset_op<numeric_type> ro(m);
    expr->apply(&ro);
}

} // end namespace ddf

#endif /* _IGNITION_H_ */
