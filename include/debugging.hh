#ifndef _DEBUGGING_H_
#define _DEBUGGING_H_

#include "expr.hh"
#include <fstream>

namespace ddf {

// Calculate the finite differentiation of specified expression
template <typename numeric_type>
matrix<numeric_type> finite_diff(
    math_expr<numeric_type> *expr, variable<numeric_type> *var,
    numeric_type delta = 1e-3) {
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
    numeric_type delta = 1e-3) {
    if (x.size() != y.size()) {
        return true;
    }

    int vec_size = x.size();
    for (int i = 0; i < vec_size; i++) {
        numeric_type diff = x[i] - y[i];
        if (diff > delta || diff < -delta) {
            return true;
        }
    }

    return false;
}

template <typename numeric_type>
class dump_expr_as_dotfile : public math_expr_visitor<numeric_type> {
public:
    dump_expr_as_dotfile(const char *filename)
        : _ofs(filename, std::ofstream::out) {
        _ofs << "digraph ddf_expr {\n";
        _ofs << "  node [ fontname = \"Monospace\" fontsize = \"8\" ];\n";
    }
    ~dump_expr_as_dotfile(void) {
        _ofs << "}\n";
    }

    virtual void apply(constant<numeric_type> *expr) {
        _exprs.insert(expr);
        _ofs << "  addr_" << expr << " [label=\"const "
             << expr->to_string() << "\"];\n";
    }
    virtual void apply(identity<numeric_type> *expr) {
        _exprs.insert(expr);
        _ofs << "  addr_" << expr << " [label=\"identity "
             << expr->to_string() << "\"];\n";
    }
    
    virtual void apply(variable<numeric_type> *expr) {
        _exprs.insert(expr);
        _ofs << "  addr_" << expr << " [label=\"var "
             << expr->to_string() << "\"];\n";
    }
    
    virtual void apply(function_call<numeric_type> *expr) {
        _exprs.insert(expr);
        _ofs << "  addr_" << expr << " [label=\"func "
             << expr->to_string() << "\"];\n";
        for (auto &arg: expr->_args) {
            _ofs << "  addr_" << expr
                 << " -> addr_" << arg.get() << ";\n";
            if (_exprs.find(arg.get()) == _exprs.end()) {
                arg->apply(this);
            }
        }
    }
    
    virtual void apply(addition<numeric_type> *expr) {
        _exprs.insert(expr);
        _ofs << "  addr_" << expr << " [label=\"add "
             << expr->to_string() << "\"];\n";
        _ofs << "  addr_" << expr
             << " -> addr_" << expr->_a.get() << ";\n";
        _ofs << "  addr_" << expr
             << " -> addr_" << expr->_b.get() << ";\n";
        if (_exprs.find(expr->_a.get()) == _exprs.end()) {
            expr->_a->apply(this);
        }
        if (_exprs.find(expr->_b.get()) == _exprs.end()) {
            expr->_b->apply(this);
        }
    }
    
private:
    std::set<math_expr<numeric_type> *> _exprs;
    std::ofstream _ofs;
};

} // end namespace ddf

#endif /* _DEBUGGING_H_ */
