#ifndef TENSOR_H
#define TENSOR_H

template <typename _numeric_type>
struct tensor {
    typedef _numeric_type numeric_type;
    numeric_type *_v;
    int _dim;
};

#endif /* TENSOR_H */
