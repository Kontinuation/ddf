#ifndef ARRAY_OPT_H
#define ARRAY_OPT_H

#include "nd_array.hh"

namespace ddf {
namespace opt {

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

template <typename numeric_type>
void mult_by_strided_matrix(
    const matrix<numeric_type> &B, const vector<numeric_type> &v,
    matrix<numeric_type> &C) {
    static thread_local vector<numeric_type> bv;
    int v_dim = v.size();
    int n_row = B.shape(0) / v_dim;
    assert(("matrix and stride vector should match", B.shape(0) % v_dim == 0));
    int n_col = B.shape(1);
    C.resize(n_row, n_col);
    bv.resize(v_dim);
    for (int i = 0; i < n_row; i++) {
        for (int j = 0; j < n_col; j++) {
            int b_row = v_dim * i;
            for (int k = 0; k < v_dim; k++, b_row++) {
                bv[k] = B(b_row, j);
            }
            C(i, j) = v.dot(bv);
        }
    }
}

template <typename numeric_type>
void mult_relu_matrix(
    const matrix<numeric_type> &A, const vector<numeric_type> &v,
    matrix<numeric_type> &C) {
    int m = A.shape(0);
    int n = v.size();
    assert(("matrix size should be multiplicable", A.shape(1) == n));
    C.resize(m, n);
    for (int k = 0; k < n; k++) {
        if (v[k] > 0) {
            for (int l = 0; l < m; l++) C(l,k) = A(l,k);
        } else {
            for (int l = 0; l < m; l++) C(l,k) = 0;
        }
    }
}

template <typename numeric_type>
void mult_by_relu_matrix(
    const matrix<numeric_type> &B, const vector<numeric_type> &v,
    matrix<numeric_type> &C) {
    int m = v.size();
    int n = B.shape(1);
    assert(("matrix size should be multiplicable", m == B.shape(0)));
    C.resize(m, n);
    for (int k = 0; k < m; k++) {
        if (v[k] > 0) {
            std::copy_n(&B(k,0), n, &C(k,0));
        } else {
            std::fill_n(&C(k,0), n, 0);
        }
    }
}

} // end namespace opt
} // end namespace ddf

#endif /* ARRAY_OPT_H */
