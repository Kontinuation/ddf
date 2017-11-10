#ifndef _OPTIMIZATION_H_
#define _OPTIMIZATION_H_

// Various parameter updater based on gradient descent. 
// - SGD: simply update along the negative gradient direction
// - Momentum: update accumulates the velocity 
// - Nesterov Momentum: Momentum with better "lookahead"
//

#include "expr.hh"

namespace ddf {

// Base class for all gradient-based optimizers
template <typename numeric_type>
class optimization {
public:
    typedef vector<numeric_type> vector_type;
    typedef variable<numeric_type> varexpr_type;

    virtual void set_params(std::map<std::string, varexpr_type *> *p_param_var) {
        _p_param_var = p_param_var;
    }

    virtual void reset(void) {}

    virtual void update_params(std::map<std::string, vector_type> &derivative) = 0;

protected:
    std::map<std::string, varexpr_type *> *_p_param_var;
};

// SGD
template <typename numeric_type>
class sgd_opt : public optimization<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef variable<numeric_type> varexpr_type;

    numeric_type alpha;         // learning rate

    sgd_opt(numeric_type alpha)
        : alpha(alpha) {
    }

    void update_params(std::map<std::string, vector_type> &derivative) {
        for (auto &kv: *this->_p_param_var) {
            const std::string &var = kv.first;
            varexpr_type *var_expr = kv.second;
            vector_type &dx = derivative[var];

            // sgd update
            dx *= alpha;
            var_expr->_val -= dx;
        }
    }
};

// Momentum
template <typename numeric_type>
class momentum_opt : public optimization<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef variable<numeric_type> varexpr_type;

    numeric_type alpha;         // learning rate
    numeric_type mu;            // friction
    std::map<std::string, vector_type> _velocity;

    momentum_opt(numeric_type alpha, numeric_type mu = 0.5)
        : alpha(alpha), mu(mu) {
    }

    void reset(void) {
        _velocity.clear();
        for (auto &kv: *this->_p_param_var) {
            const std::string &var = kv.first;
            const varexpr_type *var_expr = kv.second;
            _velocity[var] = vector_type(var_expr->value().size());
            _velocity[var].fill(0);
        }
    }

    void update_params(std::map<std::string, vector_type> &derivative) {
        for (auto &kv: *this->_p_param_var) {
            const std::string &var = kv.first;
            varexpr_type *var_expr = kv.second;
            vector_type &dx = derivative[var];
            vector_type &v = _velocity[var];

            // momentum update
            v *= mu;
            dx *= alpha;
            v -= dx;
            var_expr->_val += v;
        }
    }
};

} // end namespace ddf

#endif /* _OPTIMIZATION_H_ */
