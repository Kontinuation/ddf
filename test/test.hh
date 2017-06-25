#ifndef _TEST_H_
#define _TEST_H_

// An extremely simple unit-test infrastructure

#include "logging.hh"

inline void expect_true(bool val, const char *msg) {
    if (val) {
        logging::info("[PASSED] %s", msg);
    } else {
        logging::warning("[FAILED] %s", msg);
    }
}

#endif /* _TEST_H_ */
