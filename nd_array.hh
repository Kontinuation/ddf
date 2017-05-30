#ifndef ND_ARRAY_H
#define ND_ARRAY_H

// N-D array as values passed through out the entire deep learning algorithm,
// the underlying buffer is managed via shared pointer, so to avoid the cost of
// copying the entire data blocks when the vector is passed around.

#include <memory>
#include <cassert>
#include <cstring>
#include <algorithm>
#include <string>
#include <sstream>

#include "common.hh"
#include "logging.hh"

namespace ddf {

enum buffer_ownership {
    array_does_not_own_buffer = 0,
    array_owns_buffer = 1,
};

// multi dimensional array where each element is of type _T_numeric and the
// array has _n_dim dimensions
template <typename _numeric_type, int _n_dim>
struct nd_array_base {
    typedef _numeric_type numeric_type;
    static const int n_dim = _n_dim;

    explicit nd_array_base(std::initializer_list<int> s,
                      numeric_type *data = nullptr, 
                      buffer_ownership owned = array_does_not_own_buffer) {
        assert(("number of dimensions should match with array shape",
                s.size() == n_dim));
        int shape[n_dim];
        std::copy(s.begin(), s.end(), shape);
        construct(shape, data, owned);
    }

    explicit nd_array_base(const int *shape, numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer) {
        construct(shape, data, owned);
    }

    nd_array_base(const nd_array_base &) = default; // copy ctor
    nd_array_base(nd_array_base &&) = default; // move ctor
    virtual ~nd_array_base(void) = default;

    // constructing nd_array_base object from scratch or from specified
    // data buffer (owning/not owning)
    void construct(const int *shape, numeric_type *data = nullptr,
                   buffer_ownership owned = array_does_not_own_buffer) {
        int size = 1;
        for (int i = n_dim - 1; i >= 0; i--) {
            _dim_size[i] = size;
            _shape[i] = shape[i];
            size *= shape[i];
        }
        if (!data) {
            if (size > 0) {
#ifdef NOTICE_ALLOC_CALL
                logging::trace("allocating space for new array, size: %d", size); 
#endif
                _raw_data = new numeric_type[size];
            } else {
                _raw_data = nullptr;
            }
            _shared_data.reset(_raw_data, std::default_delete<numeric_type[]>());
        } else {
            _raw_data = data;
            if (owned == array_owns_buffer) {
                _shared_data.reset(
                    data, std::default_delete<numeric_type[]>());
            }
        }
        _buf_size = size;
    }

    void swap(nd_array_base &v) noexcept {
        std::swap(_shared_data, v._shared_data);
        std::swap(_raw_data, v._raw_data);
        std::swap(_buf_size, v._buf_size);
        std::swap(_shape, v._shape);
        std::swap(_dim_size, v._dim_size);
    }

    nd_array_base &operator = (const nd_array_base &v) {
#ifdef NOTICE_CTOR_CALL
        logging::trace("copy assignment");
#endif
        nd_array_base tmp(v);
        this->swap(tmp);
        return *this;
    }

    nd_array_base &operator = (nd_array_base &&v) {
#ifdef NOTICE_CTOR_CALL
        logging::trace("move assignment");
#endif
        this->swap(v);
        return *this;
    }

    nd_array_base &copy_from(const nd_array_base &v) {
        if (_raw_data != v._raw_data) {
            if (_buf_size == v._buf_size) {
                _buf_size = v._buf_size;
                std::copy_n(v._shape, n_dim, _shape);
                std::copy_n(v._dim_size, n_dim, _dim_size);
                copy_from(v._raw_data);
            } else {
                this->operator = (v.base_clone());
            }
        }
        return *this;
    }

    nd_array_base &copy_from(const numeric_type *buf) {
        std::copy_n(buf, _buf_size, _raw_data);
        return *this;
    }

    // member accessors, will return reference to the initial element of the
    // slice if fewer dimension were specified
    numeric_type &operator() (int n0) const {
        assert(("0th dimension access boundary check", n0 < _shape[0]));
        return _raw_data[n0 * _dim_size[0]];
    }
    numeric_type &operator() (int n0, int n1) const {
        static_assert(n_dim >= 2, "array should have at least 2 dimensions");
        assert(("0th dimension access boundary check", n0 < _shape[0]));
        assert(("1st dimension access boundary check", n1 < _shape[1]));
        return _raw_data[
            n0 * _dim_size[0]
            + n1 * _dim_size[1]];
    }
    numeric_type &operator() (int n0, int n1, int n2) const {
        static_assert(n_dim >= 3, "array should have at least 3 dimensions");
        assert(("0th dimension access boundary check", n0 < _shape[0]));
        assert(("1st dimension access boundary check", n1 < _shape[1]));
        assert(("2nd dimension access boundary check", n2 < _shape[2]));
        return _raw_data[
            n0 * _dim_size[0]
            + n1 * _dim_size[1]
            + n2 * _dim_size[2]];
    }
    numeric_type &operator() (int n0, int n1, int n2, int n3) const {
        static_assert(n_dim >= 4, "array should have at least 4 dimensions");
        assert(("0th dimension access boundary check", n0 < _shape[0]));
        assert(("1st dimension access boundary check", n1 < _shape[1]));
        assert(("2nd dimension access boundary check", n2 < _shape[2]));
        assert(("3rd dimension access boundary check", n3 < _shape[3]));
        return _raw_data[
            n0 * _dim_size[0]
            + n1 * _dim_size[1]
            + n2 * _dim_size[2]
            + n3 * _dim_size[3]];
    }

    int shape(int k) const {
        assert(("dimension check", k < n_dim));
        return _shape[k];
    }

    // get raw buffer
    numeric_type *raw_data(void) const { return _raw_data; }

    // make a deep copy
    nd_array_base base_clone(void) const {
#ifdef NOTICE_ALLOC_CALL
        logging::trace("cloning new array, size: %d", _buf_size); 
#endif
        numeric_type *new_data = new numeric_type[_buf_size];
        std::copy_n(_raw_data, _buf_size, new_data);
        return nd_array_base(_shape, new_data, array_owns_buffer);
    }

    // stringify array, a simple implementation without any care of
    // performance, please only use it for debugging and don't let it hurt your
    // performance when benchmarking
    std::string to_string(int indent = 2) const {
        std::stringstream ss;
        int cur_idx[n_dim] = {0};
        int cur_level = 0, last_level = n_dim - 1;
        ss << '[';
        for (int i = 0; i < _buf_size; i++) {
            for ( ; cur_level != last_level; ++cur_level) {
                ss << '\n';
                for (int l = 0; l < (cur_level + 1) * indent; l++) ss << ' ';
                ss << '[';
            }
            ss << _raw_data[i] <<  ", ";
            cur_idx[cur_level] += 1;
            while (cur_level > 0 && cur_idx[cur_level] == _shape[cur_level]) {
                if (cur_level == last_level) {
                    ss << ']';
                } else {
                    ss << '\n';
                    for (int l = 0; l < cur_level * indent; l++) ss << ' ';
                    ss << ']';
                }
                cur_idx[cur_level] = 0;
                cur_level -= 1;
                cur_idx[cur_level] += 1;
            }
        }
        ss << (n_dim > 1? "\n]": "]");
        return ss.str();
    }

    // simple arithmetics
    nd_array_base &operator += (const nd_array_base &v) {
        assert_same_size(v);
        for (int i = 0; i < _buf_size; i++) {
            _raw_data[i] += v._raw_data[i];
        }
        return *this;
    }

    nd_array_base &operator -= (const nd_array_base &v) {
        assert_same_size(v);
        for (int i = 0; i < _buf_size; i++) {
            _raw_data[i] -= v._raw_data[i];
        }
        return *this;
    }

    nd_array_base &operator *= (const nd_array_base &v) {
        // hadmard product
        assert_same_size(v);
        for (int i = 0; i < _buf_size; i++) {
            _raw_data[i] *= v._raw_data[i];
        }
        return *this;
    }

    nd_array_base &operator *= (numeric_type w) {
        for (int i = 0; i < _buf_size; i++) {
            _raw_data[i] *= w;
        }
        return *this;
    }

    void fill(_numeric_type val) {
        std::fill_n(_raw_data, _buf_size, val);
    }

    void fill_rand(void) {
        numeric_type factor = 1 / (numeric_type) RAND_MAX;
        for (int k = 0; k < _buf_size; k++) {
            _raw_data[k] = (rand() * factor - 0.5) * 2;
        }
    }

    void assert_same_size(const nd_array_base &v) const {
        assert(("dimension should match",
                memcmp(_shape, v._shape,
                    sizeof (numeric_type) * n_dim) == 0));
    }

    std::shared_ptr<numeric_type> _shared_data; // for managing data lifetime
    numeric_type *_raw_data;                    // raw buffer for storing data
    int _buf_size;                              // buffer size (# of elements)
    int _shape[n_dim];                          // size of each dimension
    int _dim_size[n_dim];                       // number of elements in each
                                                // dimension
};

#define ND_ARRAY_COMMON_PROCEDURES(n)                               \
    nd_array(const nd_array &) = default;                           \
    nd_array(nd_array &&) = default;                                \
                                                                    \
    nd_array clone(void) const {                                    \
        auto base = this->base_clone();                             \
        return *reinterpret_cast<nd_array *>(&base);                \
    }                                                               \
    nd_array & operator = (const nd_array &v) {                     \
        nd_array_base<_numeric_type, n>::operator = (v);            \
        return *this;                                               \
    }                                                               \
    nd_array & operator = (nd_array &&v) {                          \
        nd_array_base<_numeric_type, n>::operator = (std::move(v)); \
        return *this;                                               \
    }

// generic n-dimensional array is identical with base-line implementation
template <typename _numeric_type, int _n_dim>
struct nd_array : nd_array_base<_numeric_type, _n_dim> {
    explicit nd_array(std::initializer_list<int> s,
                      _numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer)
        : nd_array_base<_numeric_type, _n_dim>(s, data, owned) {
    }
    explicit nd_array(const int *shape, _numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer)
        : nd_array_base<_numeric_type, _n_dim>(shape, data, owned) {
    }

    ND_ARRAY_COMMON_PROCEDURES(_n_dim);
};

// special optimized version for 1-d arrays
template <typename _numeric_type>
struct nd_array<_numeric_type, 1> : nd_array_base<_numeric_type, 1> {
    explicit nd_array(std::initializer_list<int> s,
                      _numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer)
        : nd_array_base<_numeric_type, 1>(s, data, owned) {
    }
    explicit nd_array(const int *shape, _numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer)
        : nd_array_base<_numeric_type, 1>(shape, data, owned) {
    }
    explicit nd_array(int len = 0, _numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer)
        : nd_array_base<_numeric_type, 1>({len}, data, owned) {
    }

    ND_ARRAY_COMMON_PROCEDURES(1);

    int size() const {
        return this->shape(0);
    }

    // resize without zeroing all elements 
    void resize(int new_size) {
        if (this->_buf_size != new_size) {
            this->operator = (nd_array(new_size));
        }
    }

    _numeric_type &operator() (int n0) const {
        assert(("0th dimension access boundary check", n0 < this->_shape[0]));
        return this->_raw_data[n0];
    }
    _numeric_type &operator[] (int n0) const {
        return this->operator() (n0);
    }

    _numeric_type dot(const nd_array &v) const {
        this->assert_same_size(v);
        _numeric_type res = 0;
        int N = this->_shape[0];
        for (int k = 0; k < N; ++k) {
            res += this->_raw_data[k] * v._raw_data[k];
        }
        return res;
    }
};

// special optimized version for 2-d arrays
template <typename _numeric_type>
struct nd_array<_numeric_type, 2> : nd_array_base<_numeric_type, 2> {
    explicit nd_array(std::initializer_list<int> s,
                      _numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer)
        : nd_array_base<_numeric_type, 2>(s, data, owned) {
    }
    explicit nd_array(const int *shape, _numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer)
        : nd_array_base<_numeric_type, 2>(shape, data, owned) {
    }
    explicit nd_array(int m = 0, int n = 0, _numeric_type *data = nullptr,
                      buffer_ownership owned = array_does_not_own_buffer)
        : nd_array_base<_numeric_type, 2>({m, n}, data, owned) {
    }

    ND_ARRAY_COMMON_PROCEDURES(2);

    _numeric_type &operator() (int n0, int n1) const {
        assert(("0th dimension access boundary check", n0 < this->_shape[0]));
        assert(("1st dimension access boundary check", n1 < this->_shape[1]));
        return this->_raw_data[n0 * this->_dim_size[0] + n1];
    }

    // resize without zeroing all elements 
    void resize(int m, int n) {
        if (this->_shape[0] != m || this->_shape[1] != n) {
            this->operator = (nd_array(m, n));
        }
    }

    // set the value of column col to specified vector value
    void set_column(int col, const nd_array<_numeric_type, 1> &v) {
        int v_size = v.size();
        assert(("matrix-vector dimension should match", v_size == this->shape(0)));
        for (int row = 0; row < v_size; row++) {
            (*this)(row, col) = v[row];
        }
    }

    // matrix-vector multiplication
    nd_array<_numeric_type, 1> operator * (const nd_array<_numeric_type, 1> &v) {
        nd_array<_numeric_type, 1> ret(0);
        mult(v, ret);
        return ret;
    }

    // matrix-matrix multiplication
    nd_array<_numeric_type, 2> operator * (const nd_array<_numeric_type, 2> &v) {
        nd_array<_numeric_type, 2> ret(0,0);
        mult(v, ret);
        return ret;
    }

    // write result to specified vector to avoid memory allocation
    void mult(const nd_array<_numeric_type, 1> &v,
        nd_array<_numeric_type, 1> &res) const {
        const nd_array_base<_numeric_type, 2> &self = *this;
        int m = self.shape(0), n = self.shape(1);
        assert(("vector can be multiplied by matrix", v.size() == n));
        res.resize(m);
        for (int i = 0; i < m; i++) {
            res[i] = v.dot(nd_array<_numeric_type, 1>(n, &self(i)));
        }
    }

    void mult(const nd_array<_numeric_type, 2> &b,
        nd_array<_numeric_type, 2> &res) const {
        const nd_array_base<_numeric_type, 2> &self = *this;
        int m = self.shape(0), n = self.shape(1);
        int bn = b.shape(1);
        assert(("matrices can be multiplied", n == b.shape(0)));
        res.resize(m, bn);
        res.fill(0);
        for (int j = 0; j < bn; j++) {
            for (int i = 0; i < m; i++) {
                for (int k = 0; k < n; k++) {
                    res(i, j) += self(i, k) * b(k, j);
                }
            }
        }
    }
};

// aliases for commonly used nd_array types
template <typename _numeric_type>
using vector = nd_array<_numeric_type, 1>;

template <typename _numeric_type>
using matrix = nd_array<_numeric_type, 2>;

} // end namespace ddf

#endif /* ND_ARRAY_H */
