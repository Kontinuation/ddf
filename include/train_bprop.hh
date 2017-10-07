#ifndef _TRAIN_BPROP_H_
#define _TRAIN_BPROP_H_

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
class optimizer_bprop {
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
            }
        }

        // calculate initial loss
        ddf::vector<numeric_type> y;
        _training_loss = 0;
        for (int k = 0; k < _n_samples; k++) {
            for (auto &kv: _feed_var) {
                const std::string &var = kv.first;
                const matrix_type &arr_var = _feed_dict->find(var)->second;
                varexpr_type *var_expr = kv.second;
                var_expr->_val.copy_from(&arr_var(k, 0));
            }
            _loss_expr->eval(y);
            _training_loss += y[0];
        }

        this->dbg_dump();
    }

    // set learning rate
    virtual void set_learning_rate(numeric_type alpha) {
        _alpha = alpha;
    }

    // fetch current training loss
    virtual numeric_type loss(void) {
        return _training_loss;
    }

    // run iterative training algorithm
    virtual void step(int n_epochs) {
        ddf::backpropagation<numeric_type> bprop;
        ddf::reset_delta<numeric_type> reset;
        ddf::vector<numeric_type> loss;
        for (int iter = 0; iter < n_epochs; iter++) {
            // initialize accumulative derivatives
            for (auto &kv: _derivative) {
                kv.second.fill(0);
            }

            // calculate derivative for each sampl
            _training_loss = 0;
            for (int k = 0; k < _n_samples; k++) {
                // set value for feeded variables
                for (auto &kv: _feed_var) {
                    const std::string &var = kv.first;
                    const matrix_type &arr_var = _feed_dict->find(var)->second;
                    varexpr_type *var_expr = kv.second;
                    var_expr->_val.copy_from(&arr_var(k, 0));
                }

                // perform backpropagation
                _loss_expr->apply(&reset);
                _loss_expr->eval(loss);
                _loss_expr->delta.copy_from(loss);
                _loss_expr->apply(&bprop);
                _training_loss += loss[0];

                // calculate gradient for hyper parameters
                for (auto &kv: _hyperparam_var) {
                    const std::string &var = kv.first;
                    varexpr_type *var_expr = kv.second;
                    _derivative[var] += var_expr->delta;
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

protected:
    void dbg_dump(void) {
#ifdef DEBUG_BPROP
        logging::debug("loss_expr:");
        logging::debug("  %s", _loss_expr->to_string().c_str());

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
#endif
    }

protected:
    expr_type *_loss_expr = nullptr;
    const std::map<std::string, matrix_type> *_feed_dict = nullptr;
    int _n_samples = 0;
    numeric_type _alpha = 0.1; 
    std::map<std::string, varexpr_type *> _feed_var;
    std::map<std::string, varexpr_type *> _hyperparam_var;
    std::map<std::string, vector_type> _derivative;
    numeric_type _training_loss = 0;
};

} // end namespace ddf

#endif /* _TRAIN_H_ */
