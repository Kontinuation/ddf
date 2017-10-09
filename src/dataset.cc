#include "dataset.hh"
#include <algorithm>


inline void reverse_bytes(void *buf, int len) {
    uint8_t *p = (uint8_t *) buf;
    std::reverse(p, p + len);
}

inline void reverse_bytes_buf(void *buf, int n_data, int len) {
    uint8_t *p = (uint8_t *) buf;
    for (int i = 0; i < n_data; i++) {
        reverse_bytes(p, len);
        p += len;
    }
}

bool dataset::load(const char *path) {
    unload();

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::streamsize filesize = file.tellg();
    if (filesize < 12) {
        logging::error("Invalid dataset file size");
        return false;
    }

    file.seekg(0, std::ios::beg);
    _filesize = filesize;
    std::unique_ptr<uint8_t[]> data(new uint8_t[_filesize]);
    if (!file.read((char *) data.get(), filesize)) {
        logging::error("Failed to load file into memory");
        return false;            
    }

    // read metadata
    uint8_t *header = data.get();
    if (header[0] != 0 || header[1] != 0) {
        logging::error("Incorrect magic number");
        return false;
    }
    int et = header[2];
    if (et != UNSIGNED_BYTE && et != SIGNED_BYTE && et != SHORT &&
        et != INT && et != FLOAT && et != DOUBLE) {
        logging::error("Incorrect magic number");
        return false;            
    }

    int n_dim = header[3];
    int *dim = (int *) &header[4];
    int n_data = 1;
    for (int i = 0; i < n_dim; i++) {
        reverse_bytes(&dim[i], 4);
        _shape[i] = dim[i];
        n_data *= _shape[i];
    }

    void *val = (void *) &dim[n_dim];
    if (et != UNSIGNED_BYTE && et != SIGNED_BYTE) {
        reverse_bytes_buf(val, n_data,
            et == SHORT? 2: (et == DOUBLE? 8: 4));
    }

    _elem_type = (element_type) et;
    _dimension = n_dim;
    _data.reset(data.release());
    _val = val;
    return true;
}


void dataset::unload(void) {
    _filesize = 0;
    _elem_type = UNKNOWN;
    _dimension = 0;
    _data.reset(nullptr);
    _val = nullptr;
    memset(_shape, 0, sizeof _shape);
}
