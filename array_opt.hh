#ifndef ARRAY_OPT_H
#define ARRAY_OPT_H

#include "nd_array.hh"

namespace ddf {

template <typename numeric_type>
void mult_strided_matrix(
    const matrix<numeric_type> &A, const matrix<numeric_type> &B, 
    matrix<numeric_type> &C) {
    int v_dim = B.shape(1)/B.shape(0);
    ddf::vector<numeric_type> v(v_dim, B.raw_data());

#ifndef NDEBUG
    // verify that B is a strided matrix
    for (int m = 0; m < B.shape(0); m++) {
        for (int n = 0; n < B.shape(1); n++) {
            if (n < m * v_dim || n >= (m + 1) * v_dim) {
                assert(("rhs should be strided matrix",
                        B(m, n) == 0));
            } else {
                assert(("rhs should be strided matrix", 
                        B(m, n) == v[n % v_dim]));
            }
        }
    }
#endif

    int n_row = A.shape(0);
    int n_colg = A.shape(1);
    C.resize(n_row, B.shape(1));
    for (int m = 0; m < n_row; m++) {
        for (int k = 0; k < n_colg; k++) {
            numeric_type *buf_c_stride = &C(m, k * v_dim);
            std::copy_n(v.raw_data(), v_dim, buf_c_stride);
            ddf::vector<numeric_type> c_stride(v_dim, buf_c_stride);
            c_stride *= A(m, k);
        }
    }
}

template <typename numeric_type>
void mult_strided_matrix(
    const matrix<numeric_type> &A, const vector<numeric_type> &v,
    matrix<numeric_type> &C) {
    int v_dim = v.size();
    int n_row = A.shape(0);
    int n_colg = A.shape(1);
    C.resize(n_row, v_dim * n_colg);
    for (int m = 0; m < n_row; m++) {
        for (int k = 0; k < n_colg; k++) {
            numeric_type *buf_c_stride = &C(m, k * v_dim);
            std::copy_n(v.raw_data(), v_dim, buf_c_stride);
            ddf::vector<numeric_type> c_stride(v_dim, buf_c_stride);
            c_stride *= A(m, k);
        }
    }    
}

} // end namespace ddf

#endif /* ARRAY_OPT_H */
