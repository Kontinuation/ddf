#ifndef _PERSIST_VARS_H_
#define _PERSIST_VARS_H_

#include <cstdio>

#include "expr.hh"
#include "collect_vars.hh"

namespace ddf {

template <typename numeric_type>
bool persist_vars(const math_expr<numeric_type> *expr, FILE *fp) {
    collect_variable<numeric_type> collector;
    auto nonconst_expr = const_cast<math_expr<numeric_type> *>(expr);
    nonconst_expr->apply(&collector);

    // save numeric_type id
    uint32_t numtype_id = numeric_type_id<numeric_type>::value;
    if (fwrite(&numtype_id, sizeof(numtype_id), 1, fp) != 1) {
        return false;
    }

    // save number of variables
    const auto &varmap = collector.vars();
    uint32_t n_vars = (uint32_t) varmap.size();
    fwrite(&n_vars, sizeof(n_vars), 1, fp);

    // save weights
    for (auto kv : varmap) {
        const std::string varname = kv.first;
        variable<numeric_type> *var_expr = kv.second;

        // write varname
        uint32_t varname_size = (uint32_t) varname.size();
        if (fwrite(&varname_size, sizeof(varname_size), 1, fp) != 1) {
            return false;
        }
        if (fwrite(varname.data(), 1, varname_size, fp) != varname_size) {
            return false;
        }

        // write value
        auto &val = var_expr->value();
        uint32_t value_len = (uint32_t) val.size();
        if (fwrite(&value_len, sizeof(value_len), 1, fp) != 1) {
            return false;
        }
        if (fwrite(val.raw_data(), sizeof(numeric_type), value_len, fp) != value_len) {
            return false;
        }
    }

    return true;
}

template <typename numeric_type>
bool load_vars(math_expr<numeric_type> *expr, FILE *fp) {
    collect_variable<numeric_type> collector;
    expr->apply(&collector);
    const auto &varmap = collector.vars();

    // load numeric_type id
    uint32_t numtype_id = 0;
    if (fread(&numtype_id, sizeof(numtype_id), 1, fp) != 1) {
        logging::error("failed to read numeric type id");
        return false;
    }

    // read numeric type id
    if (numeric_type_id<numeric_type>::value != numtype_id) {
        logging::error("unrecognized numeric type id: %u", numtype_id);
        return false;
    }

    // read number of variables
    uint32_t n_vars = 0;
    if (fread(&n_vars, sizeof(n_vars), 1, fp) != 1) {
        logging::error("failed to read number of variables");
        return false;
    }

    if (n_vars != varmap.size()) {
        logging::error(
            "number of variables does not match with expr definition "
            "(%u vs %u)", n_vars, (uint32_t) varmap.size());
        return false;
    }

    // read weights data
    for (uint32_t i_var = 0; i_var < n_vars; i_var++) {
        // read variable
        uint32_t varname_len = 0;
        if (fread(&varname_len, sizeof(varname_len), 1, fp) != 1) {
            logging::error("failed to read variable name length");
            return false;
        }

        std::unique_ptr<char> var_buf(new char[varname_len + 1]);
        if (fread(var_buf.get(), 1, varname_len, fp) != varname_len) {
            logging::error("failed to read variable name");
            return false;
        }

        var_buf.get()[varname_len] = 0;
        auto iter = varmap.find(std::string(var_buf.get()));
        if (iter == varmap.end()) {
            logging::error("cannot find variable %s in expr", var_buf.get());
            return false;
        }

        // read value
        uint32_t val_len = 0;
        if (fread(&val_len, sizeof(val_len), 1, fp) != 1) {
            logging::error(
                "failed to read variable length for %s",
                var_buf.get());
            return false;
        }

        variable<numeric_type> *var_expr = iter->second;
        auto &val = var_expr->value();
        val.resize(val_len);

        if (fread(val.raw_data(), sizeof(numeric_type), val_len, fp) !=
            val_len) {
            logging::error(
                "failed to read value for variable %s, length: %u",
                var_buf.get(), val_len);
            return false;
        }
    }

    return true;
}

} // end namespace ddf

#endif /* _PERSIST_VARS_H_ */
