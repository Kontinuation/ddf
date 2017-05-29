#ifndef _DEBUGGING_H_
#define _DEBUGGING_H_

namespace ddf {

// Calculate the finite differentiation of specified expression
template <typename numeric_type>
matrix<numeric_type> finite_diff(
    math_expr<numeric_type> *expr, variable<numeric_type> *var,
    numeric_type delta = 1e-6) {
    vector<double> expr_val;
    expr->eval(expr_val);
    auto &val = var->value();

    matrix<numeric_type> ret(expr_val.size(), val.size());
    numeric_type multiplier = 1 / delta;
        
    for (int i = 0; i < val.size(); i++) {
        vector<numeric_type> expr_val1;
        numeric_type tmp = val[i];
        val[i] += delta;
        expr->eval(expr_val1);
        expr_val1 -= expr_val;
        expr_val1 *= multiplier;
        val[i] = tmp;
        ret.set_column(i, expr_val1);
    }

    return ret;
}

} // end namespace ddf

#endif /* _DEBUGGING_H_ */
