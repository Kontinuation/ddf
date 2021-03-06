#ifndef _BPROP_H_
#define _BPROP_H_

// Calculating gradient using backpropagation

#include "expr.hh"
#include "ignition.hh"

namespace ddf {

template <typename numeric_type>
class backpropagation : public math_expr_visitor<numeric_type> {
public:
    virtual void apply(constant<numeric_type> *expr) {}
    virtual void apply(identity<numeric_type> *expr) {}
    virtual void apply(variable<numeric_type> *expr) {}
    
    virtual void apply(function_call<numeric_type> *expr) {
        // accumulate delta
        auto &dy = expr->delta;
        size_t n_args = expr->_args.size();
        for (size_t k = 0; k < n_args; k++) {
            vector<numeric_type> &d = expr->_dxs[k];
            expr->_op->bprop(k, dy, d);
#ifdef DEBUG_BPROP
            logging::debug("expr %s bprop[%zu]: %s -> %s",
                expr->to_string().c_str(), k,
                dy.to_string().c_str(),
                d.to_string().c_str());
#endif
            accum_delta(expr->_args[k]->delta, d);
        }

        // backprop to sub expressions
        for (size_t k = 0; k < n_args; k++) {
            expr->_args[k]->apply(this);
        }
    }
        
    virtual void apply(addition<numeric_type> *expr) {
        // accumulate delta
        accum_delta(expr->_a->delta, expr->delta);
        accum_delta(expr->_b->delta, expr->delta);

        // backprop to sub expressions
        expr->_a->apply(this);
        expr->_b->apply(this);
    }

    void accum_delta(vector<numeric_type> &delta, vector<numeric_type> &d) {
        if (delta.size() != d.size()) {
            delta.copy_from(d);
        } else {
            delta += d;
        }
    }
};

template <typename numeric_type>
void bprop_expr(math_expr<numeric_type> *loss_expr, vector<numeric_type> &loss) {
    backpropagation<numeric_type> bprop;
    reset_expr_delta(loss_expr);
    loss_expr->eval(loss);
    loss_expr->delta.copy_from(loss);
    loss_expr->apply(&bprop);
}

} // end namespace ddf

#endif /* _BPROP_H_ */
