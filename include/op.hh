#ifndef OP_H
#define OP_H

// Implementation of Ops

#include "op_base.hh"

namespace ddf {

// matrix variable multiplied by vector
//   y = x * v
//     where shape of x is (a, b) and shape of v is (b, 1);
//     y is a vector of (a, 1)
template <typename numeric_type>
class matrix_mult: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;

    matrix_mult(void): math_op<numeric_type>("matmul", 2, {
            {3, 3},             // A * Dw, Dw * B
            {1, 1}              // A * Dx, Dx * B
        }) {
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        if (k_param == 0) _w = v;
        else _x = v;
    }

    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _w;
        else return _x;
    }

    int size_f() {
        int n = _x.size();
        int m = _w.size() / n;
        if (m * n != _w.size()) {
            throw exception(
                "matrix size should be multiplier of vector size");
        }
        return m;
    }

    // y = _w * _x
    void f(vector_type &y) {
        matrix_type mat = matrix_view_of(_w);
        mat.mult(_x, y);
    }

    void bprop(int k_param, vector_type &d) {
        assert_param_dim(k_param);
        vector_type &dy = this->_dy;
        if (k_param == 0) {
            // dw = dy * x.T
            int n_row = dy.size();
            int n_col = _x.size();
            d.resize(_w.size());
            d.fill(0);
            for (int m = 0; m < n_row; m++) {
                vector_type row(n_col, &d[m * n_col]);
                _x.mult_add(dy[m], row);
            }
        } else {
            // dx = w.T * dy
            d.resize(_x.size());
            d.fill(0);
            int N = _x.size();
            int M = _w.size() / N;
            for (int k = 0; k < M; k++) {
                vector_type row(N, &_w[k * N]);
                row.mult_add(dy[k], d);
            }
        }
    }

    void Df(int k_param, matrix_type &D) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            D_w(D);
        } else {
            D_x(D);
        }
    }

    void D_w(matrix_type &D) {
        matrix_type mat = matrix_view_of(_w);
        int m = mat.shape(0), n = mat.shape(1);
        D.resize(m, _w.size());
        D.fill(0);
        for (int k = 0; k < _w.size(); k++) {
            D(k / n, k) = _x[k % n];
        }
    }

    void D_x(matrix_type &D) {
        matrix_type mat = matrix_view_of(_w);
        D.resize(mat.shape(0), mat.shape(1));
        D.copy_from(mat.raw_data());
    }

    matrix_type matrix_view_of(const vector_type &w) const {
        int n = _x.size();
        int m = w.size() / n;
        assert(("matrix size should be multiplier of vector size",
                m * n == w.size()));
        return matrix_type(m, n, w.raw_data());
    }

    // We need to apply an optimization for matrix mult here: the Jacobian of
    // `D (W * x)` has significiant pattern, which would lead to a good
    // optimization

    void mult_grad(int k_param, const matrix_type &B, matrix_type &DB) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            opt::mult_by_strided_matrix(B, _w, DB);
        } else {
            matrix_type D = matrix_view_of(_w);
            D.mult(B, DB);
        }
    }

    void mult_by_grad(int k_param, const matrix_type &A, matrix_type &AD) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            opt::mult_strided_matrix(A, _x, AD);
        } else {
            matrix_type D = matrix_view_of(_w);
            A.mult(D, AD);
        }
    }

    virtual int cost_mult_grad(int k_param, const matrix_type &B) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            return B.shape(0) * B.shape(1);
        } else {
            matrix_type D = matrix_view_of(_w);
            return D.shape(0) * D.shape(1) * B.shape(1);
        }
    }

    virtual int cost_mult_by_grad(int k_param, const matrix_type &A) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            return A.shape(0) * A.shape(1);
        } else {
            matrix_type D = matrix_view_of(_w);
            return A.shape(0) * A.shape(1) * D.shape(1);
        }
    }

protected:
    vector_type _w;
    vector_type _x;
};

// sum cross_entropy(label, softmax(x)) where label in labels
template <typename numeric_type>
class softmax_cross_entropy_with_logits: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;
    
    softmax_cross_entropy_with_logits(void)
        : math_op<numeric_type>("DS", 2) {
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            _w = v;
            numeric_type shift = *std::max_element(
                v.raw_data(), v.raw_data() + v.size());
            
            // precalculate exp(w) and multiplier
            numeric_type divider = 0;
            int n = _w.size();            
            _exp_w.resize(n);
            for (int k = 0; k < n; k++) {
                numeric_type exp_wk = exp(_w[k] - shift);
                _exp_w[k] = exp_wk;
                divider += exp_wk;
            }
            numeric_type multiplier = 1 / divider;
            for (int k = 0; k < n; k++) {
                _exp_w[k] *= multiplier;
            }
        } else if (k_param == 1) {
            _l = v;
        }
    }

    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _w;
        else if (k_param == 1) return _l;
        else return vector_type();
    }

    int size_f() {
        return 1;
    }

    void f(vector_type &y) {
        int n = _w.size();
        assert(("prediction size should match with label", n == _l.size()));
        numeric_type sum_ce = 0;
        for (int k = 0; k < n; k++) {
            sum_ce -= _l[k] * log(_exp_w[k]);
        }

        y.resize(1);
        y[0] = sum_ce;
    }

    void bprop(int k_param, vector_type &d) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            int n = _w.size();
            d.resize(n);
            for (int i = 0; i < n; i++) {
                d[i] = _exp_w[i] - _l[i];
            }
        } else {
            // throw exception("bprop of label is not implemented");
        }
    }

    void Df(int k_param, matrix_type &D) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            D_w(D);
        } else {
            throw exception("gradient of label is not implemented");
        }
    }
    
    void D_w(matrix_type &D) {
        int n = _w.size();
        assert(("prediction size should match with label", n == _l.size()));
        D.resize(1, n);
        for (int i = 0; i < n; i++) {
            D(0, i) = _exp_w[i] - _l[i];
        }
    }

protected:
    vector_type _w;
    vector_type _l;
    vector_type _exp_w;
};

// Rectifier
template <typename numeric_type>
class relu: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;
    
    relu(): math_op<numeric_type>("relu", 1, {
            {2, 2},             // A * Dx, Dx * B
        }) {
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        _x = v;
    }
    
    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _x;
        else return vector_type();
    }

    int size_f() {
        return _x.size();
    }

    void f(vector_type &y) {
        int n = _x.size();
        y.resize(n);
        for (int k = 0; k < n; k++) {
            y[k] = _x[k] > 0? _x[k]: 0;
        }
    }

    void bprop(int k_param, vector_type &d) {
        assert_param_dim(k_param);
        vector_type &dy = this->_dy;
        int n = _x.size();
        d.resize(n);
        for (int k = 0; k < n; k++) {
            d[k] = _x[k] > 0? dy[k]: 0;
        }
    }

    void Df(int k_param, matrix_type &D) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            D_x(D);
        }
    }

    void D_x(matrix_type &D) {
        int n = _x.size();
        D.resize(n, n);
        D.fill(0);
        for (int k = 0; k < n; k++) {
            D(k, k) = _x[k] > 0? 1: 0;
        }
    }

    // the jacobian of `D relu` is a diagonal matrix, which would lead to a
    // good optimization

    void mult_grad(int k_param, const matrix_type &B, matrix_type &DB) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            opt::mult_by_relu_matrix(B, _x, DB);
        }
    }

    void mult_by_grad(int k_param, const matrix_type &A, matrix_type &AD) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            opt::mult_relu_matrix(A, _x, AD);
        }
    }

    virtual int cost_mult_grad(int k_param, const matrix_type &B) {
        assert_param_dim(k_param);
        return B.shape(0) * B.shape(1);
    }

    virtual int cost_mult_by_grad(int k_param, const matrix_type &A) {
        assert_param_dim(k_param);
        return A.shape(0) * A.shape(1);
    }

protected:
    vector_type _x;
};

// convolutional operator
template <typename numeric_type>
class convolution: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;

    convolution(int w, int h, int d, int fw, int fh, int od, int stride, int padding)
        : math_op<numeric_type>("conv", 3),
          _w(w), _h(h), _d(d), _fw(fw), _fh(fh), _fd(_d * od), _od(od),
          _s(stride), _p(padding),
          _input({0,0,0}), _filter({0,0,0}) {
        if ((_w - _fw + _p + _p) % _s != 0 || 
            (_h - _fh + _p + _p) % _s != 0) {
            throw exception("convnet output does not fit");
        }

        // output slice size
        _out_w = (_w - _fw + _p + _p) / _s + 1;
        _out_h = (_h - _fh + _p + _p) / _s + 1;
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            _v_input = v;
            _input = nd_array<numeric_type, 3>({_d, _h, _w}, v.raw_data());
        } else if (k_param == 1) {
            _v_filter = v;
            _filter = nd_array<numeric_type, 3>({_fd, _fh, _fw}, v.raw_data());
        } else if (k_param == 2) {
            _bias = v;
        }
    }

    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _v_input;
        else if (k_param == 1) return _v_filter;
        else return _bias;
    }
 
    int size_f() {
        if (_w * _h * _d != _v_input.size()) {
            throw exception(
                "input volume size does not match with declaration");
        } else if (_fw * _fh * _fd != _v_filter.size()) {
            throw exception(
                "filter volume size does not match with declaration");
        } else if (_od != _bias.size()) {
            throw exception(
                "bias vector size does not match with declaration");
        }
        return (_w - _fw + _p + _p) / _s + 1;
    }

    void f(vector_type &y) {
        y.resize(_out_w * _out_h * _od);
        nd_array<numeric_type, 3> out({ _od, _out_h, _out_w}, y.raw_data());
        for (int i_od = 0; i_od < _od; i_od++) {
            for (int i = - _p; i < _h + _p; i += _s) {
                for (int j = - _p; j < _w + _p; j += _s) {
                    // calculating inner product of input volume slice and
                    // filter volume
                    numeric_type val = 0;
                    for (int k = 0; k < _fh; k++) {
                        // skip paddings
                        int ii = i + k;
                        int jj = j;
                        int fw = _fw;
                        // skip padding row
                        if (ii < 0 || ii >= _h) continue;
                        // skip padding area
                        if (j < 0) { fw += j; jj = 0; } // left padding
                        else if (j + fw >= _w) { fw = (_w - j); } // right padding
                        // skip slice containing only paddings
                        if (fw <= 0) continue;
                        
                        for (int d = 0; d < _d; d++) {
                            vector_type input_slice(_fw, &_input(d, ii, jj));
                            vector_type filter_slice(_fw, &_filter(i_od * d, k, 0));
                            val += input_slice.dot(filter_slice);
                        }
                    }
                    out(i_od, i, j) = val;
                }
            }
        }
    }

    void bprop(int k_param, vector_type &d) {
        assert_param_dim(k_param);
        if (k_param == 0) {
            bprop_input(d);
        } else if (k_param == 1) {
            bprop_filter(d);
        } else if (k_param == 2) {
            bprop_bias(d);
        }
        throw exception("not implemented yet");
    }

    void bprop_input(vector_type &d) {
        vector_type &dy = this->_dy;
        nd_array<numeric_type, 3> dy_3d({_od, _out_h, _out_w}, dy.raw_data());

        // prepare error volume of filter
        d.resize(_d * _h * _w);
        nd_array<numeric_type, 3> d_3d({_d, _h, _w}, d.raw_data());
        d_3d.fill(0);

        // distribute errors to corresponding elements in filter volume
        for (int i_od = 0; i_od < _od; i_od++) {
            int out_i = 0;
            for (int i = - _p; i < _h + _p; i += _s, out_i++) {
                int out_j = 0;
                for (int j = - _p; j < _w + _p; j += _s, out_j++) {
                    // calculating delta for this patch
                    for (int k = 0; k < _fh; k++) {
                        // skip paddings
                        int ii = i + k;
                        int jj = j;
                        int fw = _fw;
                        // skip padding row
                        if (ii < 0 || ii >= _h) continue;
                        // skip padding area
                        if (j < 0) { fw += j; jj = 0; } // left padding
                        else if (j + fw >= _w) { fw = (_w - j); } // right padding
                        // skip slice containing only paddings
                        if (fw <= 0) continue;

                        for (int i_d = 0; i_d < _d; i_d++) {
                            vector_type d_slice(_fw, &d_3d(i_d, ii, jj));
                            vector_type filter_slice(_fw, &_filter(i_od * i_d, k, 0));
                            // d_slice += filter_slice * dy_3d(i_od, out_i, out_j);
                            filter_slice.mult_add(dy_3d(i_od, out_i, out_j), d_slice);
                        }
                    }
                }
            }
        }
    }

    void bprop_filter(vector_type &d) {
        vector_type &dy = this->_dy;
        nd_array<numeric_type, 3> dy_3d({_od, _out_h, _out_w}, dy.raw_data());

        // prepare error volume of filter
        d.resize(_od * _out_h * _out_w);
        nd_array<numeric_type, 3> d_3d({_fd, _fh, _fw}, d.raw_data());
        d_3d.fill(0);

        // distribute errors to corresponding elements in filter volume
        for (int i_od = 0; i_od < _od; i_od++) {
            int out_i = 0;
            for (int i = - _p; i < _h + _p; i += _s, out_i++) {
                int out_j = 0;
                for (int j = - _p; j < _w + _p; j += _s, out_j++) {
                    // calculating delta for this patch
                    for (int k = 0; k < _fh; k++) {
                        // skip paddings
                        int ii = i + k;
                        int jj = j;
                        int fw = _fw;
                        // skip padding row
                        if (ii < 0 || ii >= _h) continue;
                        // skip padding area
                        if (j < 0) { fw += j; jj = 0; } // left padding
                        else if (j + fw >= _w) { fw = (_w - j); } // right padding
                        // skip slice containing only paddings
                        if (fw <= 0) continue;

                        for (int i_d = 0; i_d < _d; i_d++) {
                            vector_type input_slice(_fw, &_input(i_d, ii, jj));
                            vector_type d_slice(_fw, &d_3d(i_od * i_d, k, 0));
                            // d_slice += input_slice * dy_3d(i_od, out_i, out_j);
                            input_slice.mult_add(dy_3d(i_od, out_i, out_j), d_slice);
                        }
                    }
                }
            }
        }
    }

    void bprop_bias(vector_type &d) {
        d.resize(_od);
        vector_type &dy = this->_dy;
        nd_array<numeric_type, 3> dy_3d({_od, _out_h, _out_w}, dy.raw_data());
        for (int i_od = 0; i_od < _od; i_od++) {
            matrix_type slice(_out_h, _out_w, &dy_3d(i_od, 0, 0));
            d[i_od] = slice.sum();
        }
    } 

protected:
    int _w, _h, _d;             // size of input volume
    int _fw, _fh, _fd;          // size of filters
    int _od;                    // output depth (_od = _fd / _d)
    int _s;                     // stride size
    int _p;                     // size of zero padding
    int _out_w, _out_h;         // output slice size
    nd_array<numeric_type, 3> _input;
    nd_array<numeric_type, 3> _filter;
    vector<numeric_type> _bias;
    vector<numeric_type> _v_input;
    vector<numeric_type> _v_filter;
};

} // end namespace ddf

#endif /* OP_H */
