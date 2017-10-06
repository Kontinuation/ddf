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

    matrix_mult(void): math_op<numeric_type>("matmul", 2) {
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

    matrix_type matrix_view_of(const vector_type &w) const {
        int n = _x.size();
        int m = w.size() / n;
        assert(("matrix size should be multiplier of vector size",
                m * n == w.size()));
        return matrix_type(m, n, w.raw_data());
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
            // bprop of label _l is not useful since label should always be a
            // constant, so bprop of _l is not implemented
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
    
    relu(): math_op<numeric_type>("relu", 1) {
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
          _w(w), _h(h), _d(d), _fw(fw), _fh(fh), _od(od),
          _s(stride), _p(padding),
          _input({0,0,0}), _filter({0,0,0,0}) {
        if (_w - _fw + _p + _p <= 0 || _h - _fh + _p + _p <= 0) {
            throw exception("convnet input size is too small");
        }
        if ((_w - _fw + _p + _p) % _s != 0 || 
            (_h - _fh + _p + _p) % _s != 0) {
            throw exception("convnet filter size does not fit with input");
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
            _filter = nd_array<numeric_type, 4>({_od, _d, _fh, _fw}, v.raw_data());
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
        } else if (_fw * _fh * _d * _od != _v_filter.size()) {
            throw exception(
                "filter volume size does not match with declaration");
        } else if (_od != _bias.size()) {
            throw exception(
                "bias vector size does not match with declaration");
        }
        return _out_w * _out_h * _od;
    }

    void f(vector_type &y) {
        y.resize(_out_w * _out_h * _od);
        nd_array<numeric_type, 3> out({ _od, _out_h, _out_w}, y.raw_data());
        for (int i_od = 0; i_od < _od; i_od++) {
            for (int i = - _p, i_out = 0; i_out < _out_h; i += _s, i_out++) {
                for (int j = - _p, j_out = 0; j_out < _out_w; j += _s, j_out++) {
                    // calculating inner product of input volume slice and
                    // filter volume
                    numeric_type val = 0;
                    for (int k = 0; k < _fh; k++) {
                        // skip paddings
                        int ii = i + k;
                        int jj = j;
                        int fw = _fw;
                        int f_j = 0;
                        // skip padding row
                        if (ii < 0 || ii >= _h) continue;
                        // skip padding area
                        if (j < 0) { fw += j; f_j -= j; jj = 0; } // left padding
                        else if (j + fw >= _w) { fw = (_w - j); } // right padding
                        // skip slice containing only paddings
                        if (fw <= 0) continue;

                        for (int i_d = 0; i_d < _d; i_d++) {
                            vector_type input_slice(fw, &_input(i_d, ii, jj));
                            vector_type filter_slice(fw, &_filter(i_od, i_d, k, f_j));
                            val += input_slice.dot(filter_slice);
                        }
                    }
                    out(i_od, i_out, j_out) = val + _bias[i_od];
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
    }

    void bprop_input(vector_type &d) {
        vector_type &dy = this->_dy;
        nd_array<numeric_type, 3> dy_3d({_od, _out_h, _out_w}, dy.raw_data());

        // prepare error volume of filter
        d.resize(_d * _h * _w);
        nd_array<numeric_type, 3> d_3d({_d, _h, _w}, d.raw_data());
        d_3d.fill(0);

        // distribute errors to corresponding elements in input volume
        for (int i_od = 0; i_od < _od; i_od++) {
            for (int i = - _p, i_out = 0; i_out < _out_h; i += _s, i_out++) {
                for (int j = - _p, j_out = 0; j_out < _out_w; j += _s, j_out++) {
                    // calculating delta for this patch
                    for (int k = 0; k < _fh; k++) {
                        // skip paddings
                        int ii = i + k;
                        int jj = j;
                        int fw = _fw;
                        int f_j = 0;
                        // skip padding row
                        if (ii < 0 || ii >= _h) continue;
                        // skip padding area
                        if (j < 0) { fw += j; f_j -= j; jj = 0; } // left padding
                        else if (j + fw >= _w) { fw = (_w - j); } // right padding
                        // skip slice containing only paddings
                        if (fw <= 0) continue;

                        for (int i_d = 0; i_d < _d; i_d++) {
                            vector_type d_slice(fw, &d_3d(i_d, ii, jj));
                            vector_type filter_slice(fw, &_filter(i_od, i_d, k, f_j));
                            // d_slice += filter_slice * dy_3d(i_od, i_out, j_out);
                            filter_slice.mult_add(dy_3d(i_od, i_out, j_out), d_slice);
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
        d.resize(_od * _d * _fh * _fw);
        nd_array<numeric_type, 4> d_4d({_od, _d, _fh, _fw}, d.raw_data());
        d_4d.fill(0);

        // distribute errors to corresponding elements in filter volume
        for (int i_od = 0; i_od < _od; i_od++) {
            for (int i = - _p, i_out = 0; i_out < _out_h; i += _s, i_out++) {
                for (int j = - _p, j_out = 0; j_out < _out_w; j += _s, j_out++) {
                    // calculating delta for this patch
                    for (int k = 0; k < _fh; k++) {
                        // skip paddings
                        int ii = i + k;
                        int jj = j;
                        int fw = _fw;
                        int f_j = 0;
                        // skip padding row
                        if (ii < 0 || ii >= _h) continue;
                        // skip padding area
                        if (j < 0) { fw += j; f_j -= j; jj = 0; } // left padding
                        else if (j + fw >= _w) { fw = (_w - j); } // right padding
                        // skip slice containing only paddings
                        if (fw <= 0) continue;

                        for (int i_d = 0; i_d < _d; i_d++) {
                            vector_type input_slice(fw, &_input(i_d, ii, jj));
                            vector_type d_slice(fw, &d_4d(i_od, i_d, k, f_j));
                            // d_slice += input_slice * dy_3d(i_od, out_i, out_j);
                            input_slice.mult_add(dy_3d(i_od, i_out, j_out), d_slice);
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
    int _fw, _fh;               // size of filters
    int _od;                    // output depth
    int _s;                     // stride size
    int _p;                     // size of zero padding
    int _out_w, _out_h;         // output slice size
    nd_array<numeric_type, 3> _input;
    nd_array<numeric_type, 4> _filter;
    vector<numeric_type> _bias;
    vector<numeric_type> _v_input;
    vector<numeric_type> _v_filter;
};

// Pooling operator squeezes big tensors to smaller ones
template <typename numeric_type>
class pooling: public math_op<numeric_type> {
public:
    typedef vector<numeric_type> vector_type;
    typedef matrix<numeric_type> matrix_type;

    pooling(int w, int h, int d, int extent, int stride, int padding)
        : math_op<numeric_type>("pool", 1),
          _w(w), _h(h), _d(d), _ks(extent), _s(stride), _p(padding),
          _input({0,0,0}) {
        if (_w + _p + _p < _ks || h + _p + _p < _ks) {
            throw exception("pooling input size is too small");
        }
        if ((_w - _ks + _p + _p) % _s != 0 ||
            (_h - _ks + _p + _p) % _s != 0) {
            throw exception(
                "pooling input size does not fit with extent and stride");
        }
        _out_w = (_w - _ks + _p + _p) / _s + 1;
        _out_h = (_h - _ks + _p + _p) / _s + 1;
        _out_d = d;
    }

    void prepare(int k_param, const vector_type &v) {
        assert_param_dim(k_param);
        _x = v;
        _input = nd_array<numeric_type, 3>({_d, _h, _w}, v.raw_data());
    }

    vector_type get_param(int k_param) {
        assert_param_dim(k_param);
        if (k_param == 0) return _x;
        else return vector_type();
    }

    int size_f() {
        return _out_w * _out_h * _out_d;
    }

    void f(vector_type &y) {
        y.resize(_out_w * _out_h * _out_d);
        nd_array<numeric_type, 3> out({ _out_d, _out_h, _out_w}, y.raw_data());
        for (int i_od = 0; i_od < _out_d; i_od++) {
            for (int i = - _p, i_out = 0; i_out < _out_h; i += _s, i_out++) {
                for (int j = - _p, j_out = 0; j_out < _out_w; j += _s, j_out++) {
                    // calculate maximum value in this tile
                    numeric_type max_val = std::numeric_limits<numeric_type>::lowest();
                    for (int k = 0; k < _ks; k++) {
                        // skip paddings
                        int ii = i + k;
                        int jj = j;
                        int fw = _ks;
                        // skip padding row
                        if (ii < 0 || ii >= _h) continue;
                        // skip padding area
                        if (j < 0) { fw += j; jj = 0; } // left padding
                        else if (j + fw >= _w) { fw = (_w - j); } // right padding
                        // find maximum value in this row
                        for ( ; fw > 0; --fw, ++jj) {
                            numeric_type val = _input(i_od, ii, jj);
                            max_val = (val > max_val? val: max_val);
                        }
                    }
                    out(i_od, i_out, j_out) = max_val;
                }
            }
        }
    }

    void bprop(int k_param, vector_type &d) {
        assert_param_dim(k_param);
        bprop_input(d);
    }

    void bprop_input(vector_type &d) {
        vector_type &dy = this->_dy;
        nd_array<numeric_type, 3> dy_3d({_out_d, _out_h, _out_w}, dy.raw_data());

        // prepare error volume of filter
        d.resize(_d * _h * _w);
        nd_array<numeric_type, 3> d_3d({_d, _h, _w}, d.raw_data());
        d_3d.fill(0);

        // route errors to corresponding elements in input volume
        for (int i_od = 0; i_od < _out_d; i_od++) {
            for (int i = - _p, i_out = 0; i_out < _out_h; i += _s, i_out++) {
                for (int j = - _p, j_out = 0; j_out < _out_w; j += _s, j_out++) {
                    // calculate the location of the maximum value in this tile
                    numeric_type max_val = std::numeric_limits<numeric_type>::lowest();
                    int i_max = -1, j_max = -1;
                    for (int k = 0; k < _ks; k++) {
                        // skip paddings
                        int ii = i + k;
                        int jj = j;
                        int fw = _ks;
                        // skip padding row
                        if (ii < 0 || ii >= _h) continue;
                        // skip padding area
                        if (j < 0) { fw += j; jj = 0; } // left padding
                        else if (j + fw >= _w) { fw = (_w - j); } // right padding
                        // find maximum value in this row
                        for ( ; fw > 0; --fw, ++jj) {
                            numeric_type val = _input(i_od, ii, jj);
                            if (val > max_val) {
                                max_val = val;
                                i_max = ii;
                                j_max = jj;
                            }
                        }
                    }

                    // accumulate error on that input cell
                    if (i_max >= 0 && j_max >= 0) {
                        d_3d(i_od, i_max, j_max) += dy_3d(i_od, i_out, j_out);
                    }
                }
            }
        }
    }

protected:
    int _w, _h, _d;             // input size
    int _out_w, _out_h, _out_d; // output size
    int _ks, _s, _p;            // hyperparams _ks: extent; _s: stride; _p:
                                // zero-padding
    vector_type _x;             // input data
    nd_array<numeric_type, 3> _input; // input data in 3d tensor view
};

} // end namespace ddf

#endif /* OP_H */
