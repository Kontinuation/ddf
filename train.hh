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

#include "collect_vars.hh"

namespace ddf {

template <typename numeric_type>
class optimization {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;
    typedef math_expr<numeric_type> expr_type;
    typedef variable<numeric_type> varexpr_type;

    // prepare to minimize loss on training samples specified in feed_dict
    virtual void minimize(
        expr_type *loss_expr, 
        const std::map<std::string, matrix_type> *feed_dict) {

        // validate expression
        ddf::check_size<numeric_type> checker;
        loss_expr->apply(&checker);

        // remember input arguments
        _feed_dict = feed_dict;
        _loss_expr = loss_expr;
        if (_feed_dict->size() < 0) {
            throw ddf::exception("feed_dict should not be empty");
        } else {
            _n_samples = _feed_dict->begin()->second.shape(0);
            for (auto &kv: *feed_dict) {
                if (kv.second.shape(0) != _n_samples) {
                    throw ddf::exception(
                        "all matrices in feed_dict should have "
                        "same number of rows");
                }
            }
        }

        // collect variable nodes
        ddf::collect_variable<numeric_type> visitor;
        _loss_expr->apply(&visitor);
        for (auto &s: visitor.vars()) {
            const std::string &var = s.first;
            varexpr_type *var_expr = s.second;
            if (feed_dict->find(s.first) != feed_dict->end()) {
                _feed_var.insert(s);
            } else {
                _hyperparam_var.insert(s);
                _derivative[var] = vector_type(var_expr->value().size());
                _gradient[var] = matrix_type();
            }
        }

        // calculate gradient expressions on hyper parameters
        for (auto &s: _hyperparam_var) {
            _grad_expr[s.first] = std::shared_ptr<expr_type>(
                _loss_expr->derivative(s.first));
        }

        // perform CSE on ALL gradient expressions
        ddf::common_subexpr_elim<numeric_type> cse;
        for (auto &s: _grad_expr) {
            cse.apply(s.second);
        }
    }

    // calculate current training loss
    virtual numeric_type loss(void) {
        numeric_type sum_loss = 0;
        vector_type c;
        for (int k = 0; k < _n_samples; k++) {
            // set value for feeded variables
            for (auto &kv: _feed_var) {
                const std::string &var = kv.first;
                const matrix_type &arr_var = _feed_dict->find(var)->second;
                varexpr_type *var_expr = kv.second;
                
                // This does not work:
                //    var_expr->set_value(vector_type(vec_size, &arr_var(k, 0)));
                //
                // Since pointers were not shared amoung copies of vectors, but
                // bufferes were shared, so we need to copy direcly into the
                // underlying buffer
                var_expr->_val.copy_from(&arr_var(k, 0));
            }
            _loss_expr->eval(c);
            sum_loss += c[0];
        }
        return sum_loss;
    }

    // run iterative training algorithm
    virtual void step(int n_epochs) {
        for (int iter = 0; iter < n_epochs; iter++) {
            // initialize accumulative derivatives
            for (auto &kv: _derivative) {
                kv.second.fill(0);
            }

            // calculate derivative for each sample
            for (int k = 0; k < _n_samples; k++) {
                // set value for feeded variables
                for (auto &kv: _feed_var) {
                    const std::string &var = kv.first;
                    const matrix_type &arr_var = _feed_dict->find(var)->second;
                    varexpr_type *var_expr = kv.second;
                    var_expr->_val.copy_from(&arr_var(k, 0));
                }

                // calculate gradient for hyper parameters
                for (auto &kv: _grad_expr) {
                    const std::string &var = kv.first;
                    expr_type *grad_expr = kv.second.get();
                    matrix_type &D = _gradient[var];
                    D.fill(0);
                    grad_expr->grad(D);
                    _derivative[var] +=
                        vector<numeric_type>(D.shape(1), D.raw_data());
                }
            }

            // update hyper parameters according to their derivatives
            for (auto &kv: _hyperparam_var) {
                const std::string &var = kv.first;
                varexpr_type *var_expr = kv.second;
                vector_type &sum_deriv = _derivative[var];
                sum_deriv *= (_alpha / _n_samples);
                var_expr->_val -= sum_deriv;
            }
        }
    }

    void dbg_dump(void) {
        logging::debug("loss_expr:");
        logging::debug("  %s", _loss_expr->to_string().c_str());

        logging::debug("grad_expr:");
        for (auto &kv: _grad_expr) {
            logging::debug("  %s: %s",
                kv.first.c_str(), kv.second->to_string().c_str());
        }

        logging::debug("feed_var:");
        for (auto &kv: _feed_var) {
            logging::debug("  %s: vector(%d)",
                kv.first.c_str(), kv.second->value().size());
        }

        logging::debug("hyperparam_var:");
        for (auto &kv: _hyperparam_var) {
            logging::debug("  %s: vector(%d)",
                kv.first.c_str(), kv.second->value().size());
        }

        logging::debug("derivative:");
        for (auto &kv: _derivative) {
            logging::debug("  %s: vector(%d)",
                kv.first.c_str(), kv.second.size());
        }

        logging::debug("gradient:");
        for (auto &kv: _gradient) {
            logging::debug("  %s: matrix(%d, %d)",
                kv.first.c_str(), kv.second.shape(0), kv.second.shape(1));
        }
    }

protected:
    expr_type *_loss_expr = nullptr;
    const std::map<std::string, matrix_type> *_feed_dict = nullptr;
    int _n_samples = 0;
    numeric_type _alpha = 0.5; 
    std::map<std::string, std::shared_ptr<expr_type> > _grad_expr;
    std::map<std::string, varexpr_type *> _feed_var;
    std::map<std::string, varexpr_type *> _hyperparam_var;
    std::map<std::string, vector_type> _derivative;
    std::map<std::string, matrix_type> _gradient;
};

} // end namespace ddf

#endif /* _TRAIN_H_ */
