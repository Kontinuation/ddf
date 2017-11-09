#ifndef _MODELS_H_
#define _MODELS_H_

#include "deep_dark_fantasy.hh"

// predefined models as showcases

template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_1_model(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls,
    std::vector<ddf::variable<numeric_type> *> &vec_vars)
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

    vec_vars.push_back(var_w0);
    vec_vars.push_back(var_b0);
 
    // predict: w0 * x + b0
    auto matmul_0 = new ddf::matrix_mult<numeric_type>();
    auto predict = new ddf::addition<numeric_type>(
        new ddf::function_call<numeric_type>(
            matmul_0,
            var_w0, 
            var_x),
        var_b0);

    return predict;
}

template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_1_sigmoid_model(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls,
    std::vector<ddf::variable<numeric_type> *> &vec_vars)
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

    vec_vars.push_back(var_w0);
    vec_vars.push_back(var_b0);
 
    // predict: sigmoid(w0 * x + b0)
    auto matmul_0 = new ddf::matrix_mult<numeric_type>();
    auto sigmoid_0 = new ddf::sigmoid<numeric_type>();
    auto predict =
        new ddf::function_call<numeric_type>(
            sigmoid_0,
            new ddf::addition<numeric_type>(
                new ddf::function_call<numeric_type>(
                    matmul_0,
                    var_w0, 
                    var_x),
                var_b0));

    return predict;
}

template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_2_model(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls,
    int n_hidden, std::vector<ddf::variable<numeric_type> *> &vec_vars)
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
    var_w0->value().fill_randn();
    var_b0->value().fill_randn();
    var_w1->value().fill_randn();
    var_b1->value().fill_randn();

    vec_vars.push_back(var_w0);
    vec_vars.push_back(var_b0);
    vec_vars.push_back(var_w1);
    vec_vars.push_back(var_b1);

    auto dropout_0 = new ddf::dropout<numeric_type>(0.5);
        
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
                    dropout_0,
                    new ddf::function_call<numeric_type>(
                        relu_0,
                        new ddf::addition<numeric_type>(
                            new ddf::function_call<numeric_type>(
                                matmul_0,
                                var_w0, 
                                var_x),
                            var_b0)))),
            var_b1);

    return predict;
}

template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_2_sigmoid_model(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls,
    int n_hidden, std::vector<ddf::variable<numeric_type> *> &vec_vars)
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
    var_w0->value().fill_randn();
    var_b0->value().fill_randn();
    var_w1->value().fill_randn();
    var_b1->value().fill_randn();

    vec_vars.push_back(var_w0);
    vec_vars.push_back(var_b0);
    vec_vars.push_back(var_w1);
    vec_vars.push_back(var_b1);
        
    // predict: sigmoid(w1 * (sigmoid(w0 * x + b0)) + b1)
    auto matmul_0 = new ddf::matrix_mult<numeric_type>();
    auto matmul_1 = new ddf::matrix_mult<numeric_type>();
    auto sigmoid_0 = new ddf::sigmoid<numeric_type>();
    auto sigmoid_1 = new ddf::sigmoid<numeric_type>();
    auto predict =
        new ddf::function_call<numeric_type>(
            sigmoid_1,
            new ddf::addition<numeric_type>(
                new ddf::function_call<numeric_type>(
                    matmul_1,
                    var_w1,
                    new ddf::function_call<numeric_type>(
                        sigmoid_0,
                        new ddf::addition<numeric_type>(
                            new ddf::function_call<numeric_type>(
                                matmul_0,
                                var_w0, 
                                var_x),
                            var_b0))),
                var_b1));

    return predict;
}

template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_3_model(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls,
    int n_hidden0, int n_hidden1,
    std::vector<ddf::variable<numeric_type> *> &vec_vars)
{
    // -- identify problem size --
    int dimension = xs.shape(1);
    int n_classes = ls.shape(1);
        
    // -- prepare deep model --
    int len_w0 = n_hidden0 * dimension;
    int len_b0 = n_hidden0;
    int len_w1 = n_hidden1 * n_hidden0;
    int len_b1 = n_hidden1;
    int len_w2 = n_classes * n_hidden1;
    int len_b2 = n_classes;

    ddf::variable<numeric_type> *var_w0 =
        new ddf::variable<numeric_type>("w0", ddf::vector<numeric_type>(len_w0));
    ddf::variable<numeric_type> *var_b0 =
        new ddf::variable<numeric_type>("b0", ddf::vector<numeric_type>(len_b0));
    ddf::variable<numeric_type> *var_w1 =
        new ddf::variable<numeric_type>("w1", ddf::vector<numeric_type>(len_w1));
    ddf::variable<numeric_type> *var_b1 =
        new ddf::variable<numeric_type>("b1", ddf::vector<numeric_type>(len_b1));
    ddf::variable<numeric_type> *var_w2 =
        new ddf::variable<numeric_type>("w2", ddf::vector<numeric_type>(len_w2));
    ddf::variable<numeric_type> *var_b2 =
        new ddf::variable<numeric_type>("b2", ddf::vector<numeric_type>(len_b2));

    // initial value of hyper parameters
    var_w0->value().fill_rand();
    var_b0->value().fill_rand();
    var_w1->value().fill_rand();
    var_b1->value().fill_rand();
    var_w2->value().fill_rand();
    var_b2->value().fill_rand();

    vec_vars.push_back(var_w0);
    vec_vars.push_back(var_b0);
    vec_vars.push_back(var_w1);
    vec_vars.push_back(var_b1);
    vec_vars.push_back(var_w2);
    vec_vars.push_back(var_b2);

    // predict: w1 * (relu(w0 * x + b0)) + b1
    auto matmul_0 = new ddf::matrix_mult<numeric_type>();
    auto matmul_1 = new ddf::matrix_mult<numeric_type>();
    auto matmul_2 = new ddf::matrix_mult<numeric_type>();
    auto relu_0 = new ddf::relu<numeric_type>();
    auto relu_1 = new ddf::relu<numeric_type>();
    auto predict =
        new ddf::addition<numeric_type>(
            new ddf::function_call<numeric_type>(
                matmul_2,
                var_w2,
                new ddf::function_call<numeric_type>(
                    relu_1,
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
                        var_b1))),
            var_b2);

    return predict;
}

template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_3_sigmoid_model(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls,
    int n_hidden0, int n_hidden1,
    std::vector<ddf::variable<numeric_type> *> &vec_vars)
{
    // -- identify problem size --
    int dimension = xs.shape(1);
    int n_classes = ls.shape(1);
        
    // -- prepare deep model --
    int len_w0 = n_hidden0 * dimension;
    int len_b0 = n_hidden0;
    int len_w1 = n_hidden1 * n_hidden0;
    int len_b1 = n_hidden1;
    int len_w2 = n_classes * n_hidden1;
    int len_b2 = n_classes;

    ddf::variable<numeric_type> *var_w0 =
        new ddf::variable<numeric_type>("w0", ddf::vector<numeric_type>(len_w0));
    ddf::variable<numeric_type> *var_b0 =
        new ddf::variable<numeric_type>("b0", ddf::vector<numeric_type>(len_b0));
    ddf::variable<numeric_type> *var_w1 =
        new ddf::variable<numeric_type>("w1", ddf::vector<numeric_type>(len_w1));
    ddf::variable<numeric_type> *var_b1 =
        new ddf::variable<numeric_type>("b1", ddf::vector<numeric_type>(len_b1));
    ddf::variable<numeric_type> *var_w2 =
        new ddf::variable<numeric_type>("w2", ddf::vector<numeric_type>(len_w2));
    ddf::variable<numeric_type> *var_b2 =
        new ddf::variable<numeric_type>("b2", ddf::vector<numeric_type>(len_b2));

    // initial value of hyper parameters
    var_w0->value().fill_rand();
    var_b0->value().fill_rand();
    var_w1->value().fill_rand();
    var_b1->value().fill_rand();
    var_w2->value().fill_rand();
    var_b2->value().fill_rand();

    vec_vars.push_back(var_w0);
    vec_vars.push_back(var_b0);
    vec_vars.push_back(var_w1);
    vec_vars.push_back(var_b1);
    vec_vars.push_back(var_w2);
    vec_vars.push_back(var_b2);

    // predict: w1 * (relu(w0 * x + b0)) + b1
    auto matmul_0 = new ddf::matrix_mult<numeric_type>();
    auto matmul_1 = new ddf::matrix_mult<numeric_type>();
    auto matmul_2 = new ddf::matrix_mult<numeric_type>();
    auto sigmoid_0 = new ddf::sigmoid<numeric_type>();
    auto sigmoid_1 = new ddf::sigmoid<numeric_type>();
    auto sigmoid_2 = new ddf::sigmoid<numeric_type>();
    auto predict =
        new ddf::function_call<numeric_type>(
            sigmoid_2,
            new ddf::addition<numeric_type>(
                new ddf::function_call<numeric_type>(
                    matmul_2,
                    var_w2,
                    new ddf::function_call<numeric_type>(
                        sigmoid_1,
                        new ddf::addition<numeric_type>(
                            new ddf::function_call<numeric_type>(
                                matmul_1,
                                var_w1,
                                new ddf::function_call<numeric_type>(
                                    sigmoid_0,
                                    new ddf::addition<numeric_type>(
                                        new ddf::function_call<numeric_type>(
                                            matmul_0,
                                            var_w0, 
                                            var_x),
                                        var_b0))),
                            var_b1))),
                var_b2));

            return predict;
            }

template <typename numeric_type>
ddf::math_expr<numeric_type> *conv_model_tiny(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls, 
    std::vector<ddf::variable<numeric_type> *> &vec_vars)
{
    // -- problem size --
    //  input:  28 * 28
    //  output: 10

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

    // fc: 12 * 12 * 4 => 10
    auto fc = new ddf::matrix_mult<numeric_type>();
    ddf::variable<numeric_type> *var_w =
        new ddf::variable<numeric_type>(
            "w", ddf::vector<numeric_type>(12 * 12 * 4 * 10));

    auto relu_0 = new ddf::relu<numeric_type>();

    // initial value of hyper parameters
    var_c0->value().fill_rand(-0.5, 0.5);
    var_b0->value().fill(0);
    var_w->value().fill_rand(-0.5, 0.5);

    vec_vars.push_back(var_c0);
    vec_vars.push_back(var_b0);
    vec_vars.push_back(var_w);

    ddf::math_expr<numeric_type> *predict =
        new ddf::function_call<numeric_type>(
            fc,
            var_w,
            new ddf::function_call<numeric_type>(
                relu_0,
                new ddf::function_call<numeric_type>(
                    pool_0,
                    new ddf::function_call<numeric_type>(
                        conv_0,
                        var_x, var_c0, var_b0))));
    return predict;
}

template <typename numeric_type>
ddf::math_expr<numeric_type> *conv_model_small(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls, 
    std::vector<ddf::variable<numeric_type> *> &vec_vars) {

    // -- problem size --
    //  input:  28 * 28
    //  output: 10

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

    vec_vars.push_back(var_c0);
    vec_vars.push_back(var_b0);
    vec_vars.push_back(var_c1);
    vec_vars.push_back(var_b1);
    vec_vars.push_back(var_c2);
    vec_vars.push_back(var_b2);
    vec_vars.push_back(var_w);

    auto relu_0 = new ddf::relu<numeric_type>();
    auto relu_1 = new ddf::relu<numeric_type>();
    auto relu_2 = new ddf::relu<numeric_type>();

    auto dropout_0 = new ddf::dropout<numeric_type>(0.5);
    auto dropout_1 = new ddf::dropout<numeric_type>(0.5);
    
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
                relu_2,
                new ddf::function_call<numeric_type>(
                    pool_2,
                    new ddf::function_call<numeric_type>(
                        conv_2,
                        new ddf::function_call<numeric_type>(
                            dropout_1,
                            new ddf::function_call<numeric_type>(
                                relu_1,
                                new ddf::function_call<numeric_type>(
                                    pool_1,
                                    new ddf::function_call<numeric_type>(
                                        conv_1,
                                        new ddf::function_call<numeric_type>(
                                            dropout_0,
                                            new ddf::function_call<numeric_type>(
                                                relu_0, 
                                                new ddf::function_call<numeric_type>(
                                                    pool_0, 
                                                    new ddf::function_call<numeric_type>(
                                                        conv_0,
                                                        var_x, var_c0, var_b0)))),
                                        var_c1, var_b1)))),
                        var_c2, var_b2))));

    return predict;
}

template <typename numeric_type>
ddf::math_expr<numeric_type> *conv_model_medium(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls, 
    std::vector<ddf::variable<numeric_type> *> &vec_vars)
{
    // -- problem size --
    //  input:  28 * 28
    //  output: 10

    // -- prepare deep model --

    // conv 28 * 28 * 1 => 24 * 24 * 8
    auto conv_0 = new ddf::convolution<numeric_type>(
        28, 28, 1,              // input
        5, 5, 8,                // conv filters
        1, 0);                  // stride, padding

    ddf::variable<numeric_type> *var_c0 =
        new ddf::variable<numeric_type>(
            "c0", ddf::vector<numeric_type>(conv_0->filter_size()));
    ddf::variable<numeric_type> *var_b0 =
        new ddf::variable<numeric_type>(
            "b0", ddf::vector<numeric_type>(conv_0->depth()));

    // pool 24 * 24 * 8 => 12 * 12 * 8
    auto pool_0 = new ddf::pooling<numeric_type>(
        24, 24, 8,              // input
        2, 2, 0);               // sx, stride, padding


    // conv 12 * 12 * 8 => 8 * 8 * 16
    auto conv_1 = new ddf::convolution<numeric_type>(
        12, 12, 8,              // input
        5, 5, 16,                // conv filters
        1, 0);                  // stride, padding

    ddf::variable<numeric_type> *var_c1 =
        new ddf::variable<numeric_type>(
            "c1", ddf::vector<numeric_type>(conv_1->filter_size()));
    ddf::variable<numeric_type> *var_b1 =
        new ddf::variable<numeric_type>(
            "b1", ddf::vector<numeric_type>(conv_1->depth()));

    // pool 8 * 8 * 16 => 4 * 4 * 16
    auto pool_1 = new ddf::pooling<numeric_type>(
        8, 8, 16,                // input
        2, 2, 0);               // sx, stride, padding


    // conv 4 * 4 * 16 => 2 * 2 * 32
    auto conv_2 = new ddf::convolution<numeric_type>(
        4, 4, 16,                // input
        3, 3, 32,               // conv filters
        1, 0);                  // stride, padding

    ddf::variable<numeric_type> *var_c2 =
        new ddf::variable<numeric_type>(
            "c2", ddf::vector<numeric_type>(conv_2->filter_size()));
    ddf::variable<numeric_type> *var_b2 =
        new ddf::variable<numeric_type>(
            "b2", ddf::vector<numeric_type>(conv_2->depth()));

    // pool 2 * 2 * 32 => 1 * 1 * 32
    auto pool_2 = new ddf::pooling<numeric_type>(
        2, 2, 32,                // input
        2, 1, 0);                // sx, stride, padding

    
    // fc: 32 => 10
    auto fc = new ddf::matrix_mult<numeric_type>();
    ddf::variable<numeric_type> *var_w =
        new ddf::variable<numeric_type>(
            "w", ddf::vector<numeric_type>(320));


    // initial value of hyper parameters
    var_c0->value().fill_rand(-0.5, 0.5);
    var_b0->value().fill(0);
    var_c1->value().fill_rand(-0.5, 0.5);
    var_b1->value().fill(0);
    var_c2->value().fill_rand(-0.5, 0.5);
    var_b2->value().fill(0);
    var_w->value().fill_rand(-0.5, 0.5);

    vec_vars.push_back(var_c0);
    vec_vars.push_back(var_b0);
    vec_vars.push_back(var_c1);
    vec_vars.push_back(var_b1);
    vec_vars.push_back(var_c2);
    vec_vars.push_back(var_b2);
    vec_vars.push_back(var_w);

    auto relu_0 = new ddf::relu<numeric_type>();
    auto relu_1 = new ddf::relu<numeric_type>();
    auto relu_2 = new ddf::relu<numeric_type>();

    auto dropout_0 = new ddf::dropout<numeric_type>(1.0);
    auto dropout_1 = new ddf::dropout<numeric_type>(1.0);
    
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
                relu_2,
                new ddf::function_call<numeric_type>(
                    pool_2,
                    new ddf::function_call<numeric_type>(
                        conv_2,
                        new ddf::function_call<numeric_type>(
                            dropout_1,
                            new ddf::function_call<numeric_type>(
                                relu_1,
                                new ddf::function_call<numeric_type>(
                                    pool_1,
                                    new ddf::function_call<numeric_type>(
                                        conv_1,
                                        new ddf::function_call<numeric_type>(
                                            dropout_0,
                                            new ddf::function_call<numeric_type>(
                                                relu_0, 
                                                new ddf::function_call<numeric_type>(
                                                    pool_0, 
                                                    new ddf::function_call<numeric_type>(
                                                        conv_0,
                                                        var_x, var_c0, var_b0)))),
                                        var_c1, var_b1)))),
                        var_c2, var_b2))));

    return predict;
}

template <typename numeric_type>
ddf::math_expr<numeric_type> *conv_tutorial_model(
    ddf::variable<numeric_type> *var_x, ddf::variable<numeric_type> *var_l, 
    const ddf::matrix<numeric_type> &xs, const ddf::matrix<numeric_type> &ls, 
    std::vector<ddf::variable<numeric_type> *> &vec_vars)
{
    // -- problem size --
    //  input:  28 * 28
    //  output: 10

    // -- prepare deep model --

    // conv 28 * 28 * 1 => 24 * 24 * 20
    auto conv_0 = new ddf::convolution<numeric_type>(
        28, 28, 1,              // input
        5, 5, 20,                // conv filters
        1, 0);                  // stride, padding

    ddf::variable<numeric_type> *var_c0 =
        new ddf::variable<numeric_type>(
            "c0", ddf::vector<numeric_type>(conv_0->filter_size()));
    ddf::variable<numeric_type> *var_b0 =
        new ddf::variable<numeric_type>(
            "b0", ddf::vector<numeric_type>(conv_0->depth()));

    // pool 24 * 24 * 20 => 12 * 12 * 20
    auto pool_0 = new ddf::pooling<numeric_type>(
        24, 24, 20,              // input
        2, 2, 0);                // sx, stride, padding

    // fc1: 12 * 12 * 20 => 100
    auto fc1 = new ddf::matrix_mult<numeric_type>();
    ddf::variable<numeric_type> *var_w1 =
        new ddf::variable<numeric_type>(
            "w1", ddf::vector<numeric_type>(12 * 12 * 20 * 100));
    ddf::variable<numeric_type> *var_b1 =
        new ddf::variable<numeric_type>(
            "b1", ddf::vector<numeric_type>(100));

    // fc2: 100 => 10
    auto fc2 = new ddf::matrix_mult<numeric_type>();
    ddf::variable<numeric_type> *var_w2 =
        new ddf::variable<numeric_type>(
            "w2", ddf::vector<numeric_type>(100 * 10));
    ddf::variable<numeric_type> *var_b2 =
        new ddf::variable<numeric_type>(
            "b2", ddf::vector<numeric_type>(10));

    auto sigmoid_0 = new ddf::sigmoid<numeric_type>();
    auto sigmoid_1 = new ddf::sigmoid<numeric_type>();
    auto sigmoid_2 = new ddf::sigmoid<numeric_type>();

    // initial value of hyper parameters
    var_c0->value().fill_randn();
    var_b0->value().fill_randn();
    var_w1->value().fill_randn();
    var_b1->value().fill_randn();
    var_w2->value().fill_randn();
    var_b2->value().fill_randn();

    vec_vars.push_back(var_c0);
    vec_vars.push_back(var_b0);
    vec_vars.push_back(var_w1);
    vec_vars.push_back(var_b1);
    vec_vars.push_back(var_w2);
    vec_vars.push_back(var_b2);
    
    ddf::math_expr<numeric_type> *predict =
        new ddf::function_call<numeric_type>(
            sigmoid_2, 
            new ddf::addition<numeric_type>(
                new ddf::function_call<numeric_type>(
                    fc2,
                    var_w2,
                    new ddf::function_call<numeric_type>(
                        sigmoid_1,
                        new ddf::addition<numeric_type>(
                            new ddf::function_call<numeric_type>(
                                fc1,
                                var_w1,
                                new ddf::function_call<numeric_type>(
                                    pool_0,
                                    // new ddf::function_call<numeric_type>(
                                    //     sigmoid_0,
                                        new ddf::function_call<numeric_type>(
                                            conv_0,
                                            var_x, var_c0, var_b0))),
                            var_b1))),
                var_b2));

    return predict;
}


#endif /* _MODELS_H_ */
