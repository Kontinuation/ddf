#ifndef _DRAW_EXPR_H_
#define _DRAW_EXPR_H_

#include <fstream>
#include "expr.hh"

namespace ddf {

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

template <typename numeric_type>
void draw_expr(math_expr<numeric_type> *expr, const char *filename) {
    dump_expr_as_dotfile<numeric_type> dumper(filename);
    expr->apply(&dumper);
}

} // end namespace ddf

#endif // _DRAW_EXPR_H_
