#ifndef _MODELS_H_
#define _MODELS_H_

#include "deep_dark_fantasy.hh"

// predefined models as showcases

template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_1_model(
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls)
{
    // -- identify problem size --
    int dimension = xs.shape(1);
    int n_classes = ls.shape(1);
        
    // -- prepare deep model --
    int len_w0 = n_classes * dimension;
    int len_b0 = n_classes;

    ddf::variable<numeric_type> *var_w0 =
        new ddf::variable<numeric_type>("w0", ddf::vector<numeric_type>(len_w0));
    ddf::variable<numeric_type> *var_b0 =
        new ddf::variable<numeric_type>("b0", ddf::vector<numeric_type>(len_b0));

    // initial value of hyper parameters
    var_w0->value().fill_rand();
    var_b0->value().fill_rand();
        
    ddf::variable<numeric_type> *var_x = 
        new ddf::variable<numeric_type>("x", ddf::vector<numeric_type>(dimension));
    ddf::variable<numeric_type> *var_l = 
        new ddf::variable<numeric_type>("l", ddf::vector<numeric_type>(n_classes));

    // predict: w0 * x + b0
    auto matmul_0 = new ddf::matrix_mult<numeric_type>();
    auto predict = new ddf::addition<numeric_type>(
        new ddf::function_call<numeric_type>(
            matmul_0,
            var_w0, 
            var_x),
        var_b0);

    // loss: DS(predict, l)
    auto DS = new ddf::softmax_cross_entropy_with_logits<numeric_type>();
    auto loss = new ddf::function_call<numeric_type>(
        DS, predict, var_l);

    return loss;
}


template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_2_model(
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls, int n_hidden)
{
    // -- identify problem size --
    int dimension = xs.shape(1);
    int n_classes = ls.shape(1);
        
    // -- prepare deep model --
    int len_w0 = n_hidden * dimension;
    int len_b0 = n_hidden;
    int len_w1 = n_classes * n_hidden;
    int len_b1 = n_classes;

    ddf::variable<numeric_type> *var_w0 =
        new ddf::variable<numeric_type>("w0", ddf::vector<numeric_type>(len_w0));
    ddf::variable<numeric_type> *var_b0 =
        new ddf::variable<numeric_type>("b0", ddf::vector<numeric_type>(len_b0));
    ddf::variable<numeric_type> *var_w1 =
        new ddf::variable<numeric_type>("w1", ddf::vector<numeric_type>(len_w1));
    ddf::variable<numeric_type> *var_b1 =
        new ddf::variable<numeric_type>("b1", ddf::vector<numeric_type>(len_b1));

    // initial value of hyper parameters
    var_w0->value().fill_rand();
    var_b0->value().fill_rand();
    var_w1->value().fill_rand();
    var_b1->value().fill_rand();
        
    ddf::variable<numeric_type> *var_x = 
        new ddf::variable<numeric_type>("x", ddf::vector<numeric_type>(dimension));
    ddf::variable<numeric_type> *var_l = 
        new ddf::variable<numeric_type>("l", ddf::vector<numeric_type>(n_classes));

    // predict: w1 * (relu(w0 * x + b0)) + b1
    auto matmul_0 = new ddf::matrix_mult<numeric_type>();
    auto matmul_1 = new ddf::matrix_mult<numeric_type>();
    auto relu_0 = new ddf::relu<numeric_type>();
    auto predict =
        new ddf::addition<numeric_type>(
            new ddf::function_call<numeric_type>(
                matmul_1,
                var_w1, 
                new ddf::function_call<numeric_type>(
                    relu_0,
                    new ddf::addition<numeric_type>(
                        new ddf::function_call<numeric_type>(
                            matmul_0,
                            var_w0, 
                            var_x),
                        var_b0))),
            var_b1);

    // loss: DS(predict, l)
    auto DS = new ddf::softmax_cross_entropy_with_logits<numeric_type>();
    auto loss = new ddf::function_call<numeric_type>(
        DS, predict, var_l);

    return loss;
}


template <typename numeric_type>
ddf::math_expr<numeric_type> *conv_model(
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls)
{
    // -- problem size --
    //  input:  28 * 28
    //  output: 10
    ddf::variable<numeric_type> *var_x = 
        new ddf::variable<numeric_type>(
            "x", ddf::vector<numeric_type>(28 * 28));
    var_x->value().fill_rand();
    ddf::variable<numeric_type> *var_l = 
        new ddf::variable<numeric_type>(
            "l", ddf::vector<numeric_type>(10));
    var_l->value().fill(0);
    var_l->value()[0] = 1;

    // -- prepare deep model --

    // conv 28 * 28 * 1 => 24 * 24 * 4
    auto conv_0 = new ddf::convolution<numeric_type>(
        28, 28, 1,              // input
        5, 5, 4,                // conv filters
        1, 0);                  // stride, padding

    ddf::variable<numeric_type> *var_c0 =
        new ddf::variable<numeric_type>(
            "c0", ddf::vector<numeric_type>(conv_0->filter_size()));
    ddf::variable<numeric_type> *var_b0 =
        new ddf::variable<numeric_type>(
            "b0", ddf::vector<numeric_type>(conv_0->depth()));

    // pool 24 * 24 * 4 => 12 * 12 * 4
    auto pool_0 = new ddf::pooling<numeric_type>(
        24, 24, 4,              // input
        2, 2, 0);               // sx, stride, padding


    // conv 12 * 12 * 4 => 8 * 8 * 6
    auto conv_1 = new ddf::convolution<numeric_type>(
        12, 12, 4,              // input
        5, 5, 6,                // conv filters
        1, 0);                  // stride, padding

    ddf::variable<numeric_type> *var_c1 =
        new ddf::variable<numeric_type>(
            "c1", ddf::vector<numeric_type>(conv_1->filter_size()));
    ddf::variable<numeric_type> *var_b1 =
        new ddf::variable<numeric_type>(
            "b1", ddf::vector<numeric_type>(conv_1->depth()));

    // pool 8 * 8 * 6 => 4 * 4 * 6
    auto pool_1 = new ddf::pooling<numeric_type>(
        8, 8, 6,                // input
        2, 2, 0);               // sx, stride, padding


    // conv 4 * 4 * 6 => 2 * 2 * 10
    auto conv_2 = new ddf::convolution<numeric_type>(
        4, 4, 6,                // input
        3, 3, 10,               // conv filters
        1, 0);                  // stride, padding

    ddf::variable<numeric_type> *var_c2 =
        new ddf::variable<numeric_type>(
            "c2", ddf::vector<numeric_type>(conv_2->filter_size()));
    ddf::variable<numeric_type> *var_b2 =
        new ddf::variable<numeric_type>(
            "b2", ddf::vector<numeric_type>(conv_2->depth()));

    // pool 2 * 2 * 10 => 1 * 1 * 10
    auto pool_2 = new ddf::pooling<numeric_type>(
        2, 2, 10,                // input
        2, 1, 0);                // sx, stride, padding

    
    // fc: 10 => 10
    auto fc = new ddf::matrix_mult<numeric_type>();
    ddf::variable<numeric_type> *var_w =
        new ddf::variable<numeric_type>(
            "w", ddf::vector<numeric_type>(100));


    // initial value of hyper parameters
    var_c0->value().fill_rand(-0.5, 0.5);
    var_b0->value().fill(0);
    var_c1->value().fill_rand(-0.5, 0.5);
    var_b1->value().fill(0);
    var_c2->value().fill_rand(-0.5, 0.5);
    var_b2->value().fill(0);
    var_w->value().fill_rand(-0.5, 0.5);

    auto relu_0 = new ddf::relu<numeric_type>();
    auto relu_1 = new ddf::relu<numeric_type>();
    auto relu_2 = new ddf::relu<numeric_type>();
    
    // predict:
    //   conv0 -> relu0 -> pool0
    //   -> conv1 -> relu1 -> pool1
    //   -> conv2 -> relu2 -> pool2
    //   -> fc
    ddf::math_expr<numeric_type> *predict =
        new ddf::function_call<numeric_type>(
            fc,
            var_w,
            new ddf::function_call<numeric_type>(
                pool_2,
                new ddf::function_call<numeric_type>(
                    relu_2,
                    new ddf::function_call<numeric_type>(
                        conv_2,
                        new ddf::function_call<numeric_type>(
                            pool_1,
                            new ddf::function_call<numeric_type>(
                                relu_1,
                                new ddf::function_call<numeric_type>(
                                    conv_1,
                                    new ddf::function_call<numeric_type>(
                                        pool_0, 
                                        new ddf::function_call<numeric_type>(
                                            relu_0, 
                                            new ddf::function_call<numeric_type>(
                                                conv_0,
                                                var_x, var_c0, var_b0))),
                                    var_c1, var_b1))),
                        var_c2, var_b2))));
        
    // loss: DS(predict, l)
    auto DS = new ddf::softmax_cross_entropy_with_logits<numeric_type>();
    auto loss = new ddf::function_call<numeric_type>(
        DS, predict, var_l);

    return loss;
}

#endif /* _MODELS_H_ */
