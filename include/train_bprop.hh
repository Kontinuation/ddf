#ifndef _TRAIN_BPROP_H_
#define _TRAIN_BPROP_H_

// Given a deep network, we need to find values for its parameters in order to
// make this deep network perform certain task. This is achieved by a process
// called "training" or "learning".
// 
// In deep learning, we prepare lots of samples and feed these samples into the
// neural network, and algorithmically tune those parameters to minimize the
// training loss, and hope the trained network to be general enough to solve
// such problem.

#include "collect_vars.hh"
#include "bprop.hh"
#include "optimization.hh"

namespace ddf {

template <typename numeric_type>
class train_bprop {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;
    typedef math_expr<numeric_type> expr_type;
    typedef variable<numeric_type> varexpr_type;

    void set_optimizer(optimization<numeric_type> *opt) {
        _opt = opt;
    }

    // prepare to minimize loss on training samples specified in feed_dict
    virtual void minimize(
        expr_type *loss_expr, 
        const std::map<std::string, matrix_type> &feed_dict,
        optimization<numeric_type> *opt) {

        // validate expression
        ddf::check_size<numeric_type> checker;
        loss_expr->apply(&checker);

        // remember input arguments
        _feed_dict = feed_dict;
        _loss_expr = loss_expr;
        if (_feed_dict.size() < 0) {
            throw ddf::exception("feed_dict should not be empty");
        } else {
            _n_samples = _feed_dict.begin()->second.shape(0);
            for (auto &kv: _feed_dict) {
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
            if (_feed_dict.find(s.first) != _feed_dict.end()) {
                _feed_var.insert(s);
            } else {
                _param_var.insert(s);
                _derivative[var] = vector_type(var_expr->value().size());
            }
        }

        // initialize optimizer
        _opt = opt;
        _opt->set_params(&_param_var);
        _opt->reset();

        // done
        this->dbg_dump();
    }

    virtual void set_batch_size(int batch_size) {
        _batch_size = batch_size;
    }

    virtual void toggle_debug_log(bool onoff) {
        _show_debug_log = onoff;
    }

    // fetch current training loss
    virtual numeric_type loss(void) {
        return _training_loss/ _n_samples;
    }

    // run iterative training algorithm
    virtual void step(int n_epochs) {
        ddf::backpropagation<numeric_type> bprop;
        ddf::vector<numeric_type> loss;

        for (int iter = 0; iter < n_epochs; iter++) {
            _training_loss = 0;

            // select a batch randomly
            std::vector<unsigned int> batch_idx(_n_samples);
            std::iota(batch_idx.begin(), batch_idx.end(), 0);
            std::random_shuffle(batch_idx.begin(), batch_idx.end());
            int n_batches = (_n_samples + _batch_size - 1) / _batch_size;
            for (int i_batch = 0; i_batch < n_batches; i_batch++) {
                // reinitialize operators and init accumulative derivatives
                // before processing this batch
                set_expr_working_mode(_loss_expr, TRAINING);
                for (auto &kv: _derivative) {
                    kv.second.fill(0);
                }

                // accumulate errors on this batch
                int batch_begin = i_batch * _batch_size;
                int batch_end = std::min((i_batch + 1) * _batch_size, _n_samples);
                int batch_samples = batch_end - batch_begin;
                numeric_type batch_loss = 0;
                for (int k = batch_begin; k < batch_end; k++) {
                    int i_sample = batch_idx[k];

                    // set value for feeded variables
                    for (auto &kv: _feed_var) {
                        const std::string &var = kv.first;
                        const matrix_type &arr_var = _feed_dict[var];
                        varexpr_type *var_expr = kv.second;
                        var_expr->_val.copy_from(&arr_var(i_sample, 0));
                    }

                    // perform backpropagation
                    bprop_expr(_loss_expr, loss);
                    batch_loss += loss[0];
                    _training_loss += loss[0];

                    // accumulate gradients of parameters
                    for (auto &kv: _param_var) {
                        const std::string &var = kv.first;
                        varexpr_type *var_expr = kv.second;
                        _derivative[var] += var_expr->delta;
                    }
                }

                // update parameters according to their derivatives
                _opt->update_params(_derivative);

                if (_show_debug_log) {
                    logging::info("batch #%d [%d-%d), samples: %d, loss: %f",
                        i_batch, batch_begin, batch_end,
                        batch_samples, batch_loss / batch_samples);
                }
            }
        }
    }

protected:
    void dbg_dump(void) {
        if (_show_debug_log) {
            logging::debug("loss_expr:");
            logging::debug("  %s", _loss_expr->to_string().c_str());

            logging::debug("feed_var:");
            for (auto &kv: _feed_var) {
                logging::debug("  %s: vector(%d)",
                    kv.first.c_str(), kv.second->value().size());
            }

            logging::debug("param_var:");
            for (auto &kv: _param_var) {
                logging::debug("  %s: vector(%d)",
                    kv.first.c_str(), kv.second->value().size());
            }
        }
    }

protected:
    expr_type *_loss_expr = nullptr;
    optimization<numeric_type> *_opt = nullptr;
    int _n_samples = 0;
    int _batch_size = 256;
    bool _show_debug_log = false;
    std::map<std::string, matrix_type> _feed_dict;
    std::map<std::string, varexpr_type *> _feed_var;
    std::map<std::string, varexpr_type *> _param_var;
    std::map<std::string, vector_type> _derivative;
    numeric_type _training_loss = 0;
};

} // end namespace ddf

#endif /* _TRAIN_H_ */
