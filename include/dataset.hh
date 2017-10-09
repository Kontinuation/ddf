#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <fstream>
#include <errno.h>
#include <algorithm>
#include "logging.hh"


// dataset, could load features and labels from disk
class dataset {
public:
    enum element_type {
        UNKNOWN = 0x00,
        UNSIGNED_BYTE = 0x08,
        SIGNED_BYTE = 0x09,
        SHORT = 0x0B,
        INT = 0x0C,
        FLOAT = 0x0D,
        DOUBLE = 0x0E,
    };

    dataset(const char *path = nullptr) {
        if (path) {
            load(path);
        }
    }

    ~dataset(void) {
        unload();
    }

    // getters
    int shape(int k) const { return _shape[k]; }
    element_type elem_type() const { return _elem_type; }
    int dimension() const { return _dimension; }
    void *val() const { return _val; }

    // load data into dataset object
    bool load(const char *path);

    // unload dataset and apply changes
    void unload(void);

private:
    size_t _filesize = 0;
    element_type _elem_type = UNKNOWN;
    int _shape[256] = {0};
    int _dimension = 0;
    std::unique_ptr<uint8_t[]> _data;
    void *_val = nullptr;
};


#endif /* DATASET_H */
