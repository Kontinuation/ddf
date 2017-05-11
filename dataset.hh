#ifndef DATASET_H
#define DATASET_H

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <algorithm>
#include "logging.hh"


// dataset, could load features and labels from disk
class dataset {
public:
    dataset(const char *path = nullptr, bool read_only = true) {
        if (path) {
            load(path, read_only);
        }
    }

    ~dataset(void) {
        unload();
    }

    // getters
    int n_samples() const { return _n_samples; }
    int dimension() const { return _dimension; }
    float *features() const { return _fea; }
    int *labels() const { return _label; }
    int n_classes() const {
        const int *p = std::max_element(_label, _label + _n_samples);
        return *p + 1;
    }

    // load data into dataset object
    bool load(const char *path, bool read_only = true) {
        unload();

        // mmap the entire elf file into memory
        struct stat st;
        int open_flags = read_only? O_RDONLY: O_RDWR;
        int fd = open(path, open_flags);
        if (fd < 0) {
            logging::error("Cannot open file %s", path);
            return false;
        }
        if (fstat(fd, &st) != 0) {
            logging::error("Cannot fstat file");
            close(fd);
            return false;
        }
        if (st.st_size < 8) {
            logging::error("Invalid dataset file size");
            close(fd);
            return false;
        }
        int mmap_flags = read_only? PROT_READ: (PROT_READ | PROT_WRITE);
        void *mmap_base_ptr = mmap(NULL, st.st_size, mmap_flags, MAP_SHARED, fd, 0);
        if (mmap_base_ptr == MAP_FAILED) {
            logging::error("Failed to mmap file");
            close(fd);
            return false;
        }
        _mmap_ptr = mmap_base_ptr;
        _length = st.st_size;

        // locate data in dataset file
        int *header = (int *) _mmap_ptr;
        _n_samples = header[0];
        _dimension = header[1];
        // logging::info("# features: %d, feature dimension: %d", _n_samples, _dimension);
        if (_length < (size_t) (8 +              // header size
                (_n_samples * _dimension) * 4 + // feature size
                (_n_samples * 4))) {            // label size
            logging::error("Incomplete dataset file");
            close(fd);
            unload();
            return false;
        }
        _fea = (float *) &header[2];
        _label = (int *) &_fea[_n_samples * _dimension];
        return true;
    }

    // unload dataset and apply changes
    void unload(void) {
        if (_mmap_ptr) {
            if (!_read_only) {
                // flush changes back to dataset file
                if (msync(_mmap_ptr, _length, MS_SYNC) == -1) {
                    logging::error("msync failed");
                }
            }
            munmap(_mmap_ptr, _length);
        }
        _mmap_ptr = nullptr;
        _length = 0;
        _n_samples = 0;
        _dimension = 0;
        _fea = nullptr;
        _label = nullptr;
    }

    void *_mmap_ptr = nullptr;
    size_t _length = 0;
    bool _read_only = true;
    int _n_samples = 0;
    int _dimension = 0;
    float *_fea = nullptr;
    int *_label = nullptr;
};


#endif /* DATASET_H */
