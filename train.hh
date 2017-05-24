#ifndef _TRAIN_H_
#define _TRAIN_H_

// Given a deep network, we need to find values for its hyper parameters in
// order to make this deep network perform certain task. This is achieved by a
// process called "training" or "learning".
// 
// In deep learning, we prepare lots of samples and feed these samples into the
// neural network, and algorithmically tune those hyper parameters to minimize
// the training loss, and hope the trained network to be general enough to
// solve such problem.

#include "expr.hh"
#include <map>
#include <set>

namespace ddf {

template <typename numeric_type>
class optimization {
public:
    typedef matrix<numeric_type> matrix_type;
    typedef vector<numeric_type> vector_type;
    typedef math_expr<numeric_type> expr_type;
    typedef variable<numeric_type> varexpr_type;
    typedef std::map<std::string, vector_type> feed_dict_type;

    // prepare to minimize loss on training samples specified in feed_dict
    virtual void minimize(
        expr_type *loss_expr, 
        const feed_dict_type &feed_dict) {

        // perform CSE on loss expression
        std::unique_ptr<expr_type> cse_loss_expr(loss_expr->clone());
        _loss_expr = std::move(cse_loss_expr);

        // collect variable nodes
        std::set<std::string> vars;
        
        
    }

    // caluclate current training loss
    virtual numeric_type loss(void) {
        
    }

    // run iterative training algorithm
    virtual void step(int n_epochs) = 0;

protected:
    std::unique_ptr<expr_type> _loss_expr;
    std::map<std::string, std::shared_ptr<expr_type> > _grad;
    std::map<std::string, varexpr_type *> _loss_var_dict;
    std::map<std::string, varexpr_type *> _grad_var_dict;
};

} // end namespace ddf

#endif /* _TRAIN_H_ */
