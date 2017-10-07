#ifndef _GRAD_CHECK_H_
#define _GRAD_CHECK_H_

#include "expr.hh"

namespace ddf {

// Calculate the finite differentiation of specified expression
template <typename numeric_type>
matrix<numeric_type> finite_diff(
    math_expr<numeric_type> *expr, variable<numeric_type> *var,
    numeric_type delta = 1e-5) {
    vector<numeric_type> expr_val, expr_val1;
    expr->eval(expr_val);
    auto &val = var->value();

    matrix<numeric_type> ret(expr_val.size(), val.size());
    numeric_type multiplier = 0.5 / delta;
        
    for (int i = 0; i < val.size(); i++) {
        numeric_type tmp = val[i];
        val[i] = tmp - delta;
        expr->eval(expr_val);
        val[i] = tmp + delta;
        expr->eval(expr_val1);
        expr_val1 -= expr_val;
        expr_val1 *= multiplier;
        val[i] = tmp;
        ret.set_column(i, expr_val1);
    }

    return ret;
}

template <typename numeric_type>
bool vector_diff(
    const vector<numeric_type> &x, const vector<numeric_type> &y,
    numeric_type delta = 1e-5) {
    if (x.size() != y.size()) {
        return true;
    }

    vector<numeric_type> d = x.clone();
    d -= y;
    numeric_type diff = std::sqrt(d.dot(d));
    numeric_type denom = std::sqrt(x.dot(x)) + std::sqrt(y.dot(y));
    if (diff < delta && denom < delta) {
        return false;
    } else {
        numeric_type rel_err = diff / denom;
        if (rel_err > delta) {
            logging::debug(
                "rel_err: %f, diff: %f, denom: %f\n",
                rel_err, diff, denom);
            return true;
        }
    }

    return false;
}

template <typename numeric_type>
bool vector_matrix_diff(
    const vector<numeric_type> &x, const matrix<numeric_type> &y,
    numeric_type delta = 1e-5) {
    return vector_diff(x, ddf::vector<double>(y.shape(1), y.raw_data()), delta);
}

template <typename numeric_type>
bool grad_check(
    math_expr<numeric_type> *expr, variable<numeric_type> *var,
    const vector<numeric_type> &df,
    numeric_type delta = 1e-5, numeric_type tol = 1e-7) {

    // evaluate numerical gradient
    vector<numeric_type> pos, neg;
    auto &val = var->value();
    numeric_type multiplier = 0.5 / delta;
    vector<numeric_type> numerical_grad(val.size());

    for (int i = 0; i < val.size(); i++) {
        numeric_type old_val = val[i];
        val[i] = old_val + delta;
        expr->eval(pos);
        val[i] = old_val - delta;
        expr->eval(neg);
        val[i] = old_val;
        pos -= neg;
        // printf("diff: %s\n", pos.to_string().c_str());
        pos *= df;
        // printf("diff * df: %s\n", pos.to_string().c_str());
        numerical_grad[i] = pos.sum() * multiplier;
        // printf("grad [%d]: %f\n", i, numerical_grad[i]);
    }

    // evaluate bprop gradient
    ddf::backpropagation<numeric_type> bprop;
    ddf::reset_delta<numeric_type> reset;
    expr->apply(&reset);
    expr->eval(pos);
    expr->delta.copy_from(df);  // force set training err
    expr->apply(&bprop);

    // printf("bprop delta: %s\n", var->delta.to_string().c_str());
    // printf("numerical_grad: %s\n", numerical_grad.to_string().c_str());

    // calculate diff
    return ddf::vector_diff(var->delta, numerical_grad, tol);
}

} // end namespace ddf

#endif /* _GRAD_CHECK_H_ */
