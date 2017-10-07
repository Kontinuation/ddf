#include "deep_dark_fantasy.hh"
#include "test.hh"

void test_nd_array(void)
{
    double data[3][5] = {
        {1,2,3,4,5},
        {10,20,30,40,50},
        {100,200,300,400,500},
    };
    ddf::nd_array<double, 2> w(3,5, (double *)data);
    auto w1 = w;
    auto w2 = w.clone();
    w1(2,1) = 1000;

    expect_true(w1(2,1) == 1000, "assigment to element in nd_array");
    expect_true(w(2,1) == 1000, "shallow copy affected by assignment");
    expect_true(w2(2,1) == 200, "deep copy not affected by assignment");

    ddf::vector<double> v(3, (double *) data);
    auto v1 = v;
    auto v2 = v1.clone();
    v[1] = 1000;
    v1 += v2;
    expect_true(
        v1[0] == 2 && v1[1] == 1002 && v1[2] == 6,
        "vector addition");
    expect_true(v1.dot(v2) == 2024, "vector dot product");

    double mat0[3][4] = {
        { 1, 2, 3, 4 },
        { 5, 6, 7, 8 },
        { 9, 10, 11, 12},
    };
    double mat1[4][2] = {
        { 1, 2 },
        { 3, 4 },
        { 5, 6 },
        { 7, 8 },
    };
    ddf::matrix<double> m0(3,4, (double *) mat0);
    ddf::matrix<double> m1(4,2, (double *) mat1);
    ddf::matrix<double> m2 = m0 * m1;
    double expected_m2[3][2] = {
        { 50, 60 },
        { 114, 140 },
        { 178, 220 },
    };
    expect_true(
        m2.shape(0) == 3 && m2.shape(1) == 2,
        "dimension of matrix multiplication result");
    bool diff = ddf::vector_diff(
        ddf::vector<double>(6, m2.raw_data()),
        ddf::vector<double>(6, (double *) expected_m2));
    expect_true(!diff, "matrix multiplication result");
}

template <typename numeric_type>
void test_softmax_op(int len, int min, int max)
{
    auto op_ds = new ddf::softmax_cross_entropy_with_logits<numeric_type>();

    ddf::variable<numeric_type> *var_x =
        new ddf::variable<numeric_type>(
            "x", ddf::vector<numeric_type>(len));
    var_x->value().fill_rand(min, max);

    ddf::variable<numeric_type> *var_l = 
        new ddf::variable<numeric_type>(
            "l", ddf::vector<numeric_type>(len));
    var_l->value().fill(0);
    var_l->value()[0] = 1;

    auto loss = std::shared_ptr<ddf::math_expr<numeric_type> >(
        new ddf::function_call<numeric_type>(
            op_ds, var_x, var_l));

    // backprop gradient
    ddf::backpropagation<numeric_type> bprop;
    ddf::reset_delta<numeric_type> reset;
    ddf::vector<numeric_type> y;
    loss->apply(&reset);
    loss->eval(y);    
    loss->apply(&bprop);

    auto x_diff = ddf::finite_diff(loss.get(), var_x);
    bool b_input_diff = ddf::vector_matrix_diff(var_x->delta, x_diff);
    expect_true(!b_input_diff, "softmax bprop x diff");
}

template <typename numeric_type>
void test_linear_model(void) {
    srand(time(0));
    const int dimension = 8;
    const int n_classes = 4;
    const int n_hidden = 3;
    const int len_w0 = n_hidden * dimension;
    const int len_b0 = n_hidden;
    const int len_w1 = n_classes * n_hidden;
    const int len_b1 = n_classes;

#define LENGTH(x) (sizeof(x) / sizeof(x[0]))

    ddf::variable<numeric_type> *var_w0 =
        new ddf::variable<numeric_type>("w0", ddf::vector<numeric_type>(len_w0));
    ddf::variable<numeric_type> *var_b0 =
        new ddf::variable<numeric_type>("b0", ddf::vector<numeric_type>(len_b0));
    ddf::variable<numeric_type> *var_w1 =
        new ddf::variable<numeric_type>("w1", ddf::vector<numeric_type>(len_w1));
    ddf::variable<numeric_type> *var_b1 =
        new ddf::variable<numeric_type>("b1", ddf::vector<numeric_type>(len_b1));
    ddf::variable<numeric_type> *var_x = 
        new ddf::variable<numeric_type>("x", ddf::vector<numeric_type>(dimension));
    ddf::variable<numeric_type> *var_l = 
        new ddf::variable<numeric_type>("l", ddf::vector<numeric_type>(n_classes));
        
    // initial value of hyper parameters
    var_w0->value().fill_rand();
    var_b0->value().fill_rand();
    var_w1->value().fill_rand();
    var_b1->value().fill_rand();

    // initial value of input and target
    var_x->value().fill_rand();
    auto &val_l = var_l->value();
    val_l.fill(0);
    val_l[0] = 1;

    // predict: w1 * (relu(w0 * x + b0)) + b1
    auto matmul_0 = new ddf::matrix_mult<numeric_type>();
    auto matmul_1 = new ddf::matrix_mult<numeric_type>();
    auto relu_0 = new ddf::relu<numeric_type>();
    ddf::math_expr<numeric_type> *predict =
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
    auto loss = std::shared_ptr<ddf::math_expr<numeric_type> >(
        new ddf::function_call<numeric_type>(DS, 
            predict, /* predict->clone() */
            var_l));

    // backprop gradient
    ddf::backpropagation<numeric_type> bprop;
    ddf::reset_delta<numeric_type> reset;
    ddf::vector<numeric_type> y;
    loss->apply(&reset);
    loss->eval(y);
    loss->apply(&bprop);

    ddf::matrix<numeric_type> D_dw0 = ddf::finite_diff(loss.get(), var_w0);
    ddf::matrix<numeric_type> D_dw1 = ddf::finite_diff(loss.get(), var_w1);
    ddf::matrix<numeric_type> D_db0 = ddf::finite_diff(loss.get(), var_b0);
    ddf::matrix<numeric_type> D_db1 = ddf::finite_diff(loss.get(), var_b1);
    bool w0_finite_vs_bprop = ddf::vector_matrix_diff(var_w0->delta, D_dw0);
    bool w1_finite_vs_bprop = ddf::vector_matrix_diff(var_w1->delta, D_dw1);
    bool b0_finite_vs_bprop = ddf::vector_matrix_diff(var_b0->delta, D_db0);
    bool b1_finite_vs_bprop = ddf::vector_matrix_diff(var_b1->delta, D_db1);
        
    expect_true(!w0_finite_vs_bprop, "back propagation of w0");
    expect_true(!w1_finite_vs_bprop, "back propagation of w1");
    expect_true(!b0_finite_vs_bprop, "back propagation of b0");
    expect_true(!b1_finite_vs_bprop, "back propagation of b1");
}

void test_conv_op_0(void)
{
    double x[30];
    for (int i = 0; i < 30; i++) {
        x[i] = i;
    }

    double c[12];
    for (int i = 0; i < 12; i++) {
        c[i] = i;
    }

    double b[1] = {0};

    ddf::convolution<double> op_conv(
        5, 6, 1,                // input
        3, 4, 1,                // conv filters
        1, 0);                  // stride, padding
    
    op_conv.prepare(0, ddf::vector<double>(30, x));
    op_conv.prepare(1, ddf::vector<double>(12, c));
    op_conv.prepare(2, ddf::vector<double>(1, b));
    op_conv.size_f();

    ddf::vector<double> out;
    op_conv.f(out);
    double expected[9] = {794, 860, 926, 1124, 1190, 1256, 1454, 1520, 1586};
    expect_true(!ddf::vector_diff(
            out, ddf::vector<double>(9, expected)),
        "basic convolution");
}

void test_conv_op_1(void)
{
    // example from http://cs231n.github.io/convolutional-networks/
    double x[5 * 5 * 3] = {
        // d0
        0, 2, 2, 2, 2,
        0, 2, 2, 1, 0,
        2, 1, 2, 1, 1,
        2, 1, 0, 2, 1,
        2, 2, 2, 2, 2,
        // d1
        2, 2, 0, 2, 1,
        0, 1, 1, 1, 0,
        0, 0, 1, 1, 1,
        2, 2, 2, 1, 1,
        2, 2, 2, 0, 2,
        // d2
        1, 1, 1, 1, 0,
        2, 0, 1, 1, 2,
        1, 0, 2, 1, 0,
        2, 0, 0, 2, 1,
        0, 0, 1, 1, 2,
    };
    
    double c[3 * 3 * 6] = {
        // filter 0:
        // d0
        0, 0, -1,
        1, -1, 1,
        0, 1, 1,
        // d1
        0, -1, -1,
        1, -1, 1,
        1, 0, 1,
        // d2
        0, 1, 1,
        1, 0, 1,
        -1, 0, 0,

        // filter 1:
        // d0
        -1, 1, 0,
        0, 1, -1,
        -1, -1, 0,
        // d1
        0, -1, 0,
        -1, 0, 0,
        -1, 0, 1,
        // d2
        1, 0, -1,
        0, 0, 1,
        1, 0, -1
    };

    double b[2] = {1, 0};

    auto op_conv = new ddf::convolution<double>(
        5, 5, 3,                // input
        3, 3, 2,                // conv filters
        2, 1);                  // stride, padding
    
    op_conv->prepare(0, ddf::vector<double>(sizeof(x)/sizeof(x[0]), x));
    op_conv->prepare(1, ddf::vector<double>(sizeof(c)/sizeof(c[0]), c));
    op_conv->prepare(2, ddf::vector<double>(sizeof(b)/sizeof(b[0]), b));
    op_conv->size_f();

    ddf::vector<double> y;
    op_conv->f(y);

    double expected[2][3][3] = {
        {
            {7, 14, 3, },
            {4, 6, 4, },
            {-2, 1, 0, },
        },
        {
            {0, -6, -1, },
            {1, -4, -2, },
            {0, -6, 2, },
        }
    };
    expect_true(
        !ddf::vector_diff(ddf::vector<double>(18, (double *) expected), y),
        "cs231n convolution example");
}

void test_conv_op_2(int w, int h, int d, int fw, int fh, int od, int stride, int padding)
{
    auto op_conv = new ddf::convolution<double>(
        w, h, d,
        fw, fh, od,
        stride, padding);

    int size_output = op_conv->size_f();
    
    ddf::variable<double> *var_x =
        new ddf::variable<double>("x", ddf::vector<double>(w * h * d));
    var_x->value().fill_rand();

    ddf::variable<double> *var_c =
        new ddf::variable<double>("c",
            ddf::vector<double>(fw * fh * od * d));
    var_c->value().fill_rand();

    ddf::variable<double> *var_b =
        new ddf::variable<double>("b",
            ddf::vector<double>(od));
    var_b->value().fill_rand();
    
    ddf::variable<double> *var_l = 
        new ddf::variable<double>("l", ddf::vector<double>(size_output));
    var_l->value().fill(0);
    var_l->value()[2] = 1;

    ddf::math_expr<double> *predict = new ddf::function_call<double>(
        op_conv, var_x, var_c, var_b);

    auto DS = new ddf::softmax_cross_entropy_with_logits<double>();
    auto loss = std::shared_ptr<ddf::math_expr<double> >(
        new ddf::function_call<double>(DS, 
            predict, /* predict->clone() */
            var_l));

    ddf::vector<double> y;
    loss->eval(y);
    expect_true(y[0] > 0, "evaluation of convolution expr loss");

    auto x_diff = ddf::finite_diff(loss.get(), var_x);
    auto c_diff = ddf::finite_diff(loss.get(), var_c);
    auto b_diff = ddf::finite_diff(loss.get(), var_b);

    // backprop gradient
    ddf::backpropagation<double> bprop;
    ddf::reset_delta<double> reset;
    loss->apply(&reset);
    loss->eval(y);
    loss->apply(&bprop);

    bool x_err = ddf::vector_matrix_diff(var_x->delta, x_diff);
    bool c_err = ddf::vector_matrix_diff(var_c->delta, c_diff);
    bool b_err = ddf::vector_matrix_diff(var_b->delta, b_diff);

    expect_true(!x_err, "convolution bprop x diff");
    expect_true(!c_err, "convolution bprop c diff");
    expect_true(!b_err, "convolution bprop b diff");
}

void test_conv_fc_relu(void)
{        
    auto op_conv = new ddf::convolution<double>(
        5, 6, 3,                // input
        3, 4, 2,                // conv filters
        1, 0);                  // stride, padding

    auto matmul = new ddf::matrix_mult<double>();
    auto relu_0 = new ddf::relu<double>();

    ddf::variable<double> *var_c =
        new ddf::variable<double>("c", ddf::vector<double>(72));
    ddf::variable<double> *var_x =
        new ddf::variable<double>("x", ddf::vector<double>(90));
    ddf::variable<double> *var_cb =
        new ddf::variable<double>("cb", ddf::vector<double>(2));
        
    ddf::variable<double> *var_w =
        new ddf::variable<double>("w", ddf::vector<double>(7 * 18));
    ddf::variable<double> *var_b =
        new ddf::variable<double>("b", ddf::vector<double>(7));
    ddf::variable<double> *var_l =
        new ddf::variable<double>("l", ddf::vector<double>(7));

    var_c->value().fill_rand();
    var_x->value().fill_rand();
    var_cb->value().fill_rand();
    var_w->value().fill_rand();
    var_b->value().fill_rand();
    var_l->value().fill(0);
    var_l->value()[2] = 1;
    
    ddf::math_expr<double> *predict =
        new ddf::addition<double>(
            new ddf::function_call<double>(
                matmul,
                var_w,
                new ddf::function_call<double>(
                    relu_0,
                    new ddf::function_call<double>(
                        op_conv,
                        var_x, var_c, var_cb))),
            var_b);

    auto DS = new ddf::softmax_cross_entropy_with_logits<double>();
    auto loss = std::shared_ptr<ddf::math_expr<double> >(
        new ddf::function_call<double>(DS, 
            predict, /* predict->clone() */
            var_l));

    auto bias_diff = ddf::finite_diff(loss.get(), var_cb, 1e-6);
    auto input_diff = ddf::finite_diff(loss.get(), var_x, 1e-6);
    auto filter_diff = ddf::finite_diff(loss.get(), var_c, 1e-6);
    auto w_diff = ddf::finite_diff(loss.get(), var_w, 1e-6);
    auto b_diff = ddf::finite_diff(loss.get(), var_b, 1e-6);

    // backprop gradient
    ddf::backpropagation<double> bprop;
    ddf::reset_delta<double> reset;
    ddf::vector<double> y(0);
    loss->apply(&reset);
    loss->eval(y);
    loss->apply(&bprop);

    bool b_conv_bias_diff = ddf::vector_matrix_diff(var_cb->delta,bias_diff);
    bool b_conv_input_diff = ddf::vector_matrix_diff(var_x->delta, input_diff);
    bool b_conv_filter_diff = ddf::vector_matrix_diff(var_c->delta, filter_diff);
    bool b_matmul_w_diff = ddf::vector_matrix_diff(var_w->delta, w_diff);
    bool b_matmul_b_diff = ddf::vector_matrix_diff(var_b->delta, b_diff);

    expect_true(!b_conv_bias_diff, "conv bias in conv-fc model");
    expect_true(!b_conv_input_diff, "conv input in conv-fc model");
    expect_true(!b_conv_filter_diff, "conv filter in conv-fc model");
    expect_true(!b_matmul_w_diff, "matmul w in conv-fc model");
    expect_true(!b_matmul_b_diff, "matmul b in conv-fc model");
}

void test_pool_op()
{
    double x[5 * 5 * 3] = {
        // d0
        1,2,3,4,5,
        2,3,4,5,6,
        3,4,5,6,7,
        4,5,6,7,8,
        5,6,7,8,9,
        // d1
        5,4,3,2,1,
        6,5,4,3,2,
        7,6,5,4,3,
        8,7,6,5,4,
        9,8,7,6,5,
        // d2
        9,8,7,6,5,
        5,4,3,2,1,
        6,5,4,3,2,
        7,6,5,4,3,
        8,7,6,5,4,
    };
    
    double expected[3][3][3] = {
        {
            {3, 5, 6, },
            {5, 7, 8, },
            {6, 8, 9, },
        },
        {
            {6, 5, 3, },
            {8, 7, 5, },
            {9, 8, 6, },
        },
        {
            {9, 8, 6, },
            {7, 6, 4, },
            {8, 7, 5, },
        },
    };

    // test backprop
    ddf::variable<double> *var_x =
        new ddf::variable<double>("x",
            ddf::vector<double>(sizeof(x)/sizeof(x[0]), x));

    double l[27] = {0};
    l[2] = 1;
    ddf::variable<double> *var_l = 
        new ddf::variable<double>("l",
            ddf::vector<double>(sizeof(l)/sizeof(l[0]), l));

    auto op_pool = new ddf::pooling<double>(
        5, 5, 3,                // input size
        3, 2, 1);               // extent, stride, padding

    ddf::math_expr<double> *predict = new ddf::function_call<double>(
        op_pool, var_x);

    ddf::vector<double> y;
    predict->eval(y);

    expect_true(
        !ddf::vector_diff(ddf::vector<double>(27, (double *) expected), y),
        "evaluation of pooling operator");

    auto DS = new ddf::softmax_cross_entropy_with_logits<double>();
    auto loss = std::shared_ptr<ddf::math_expr<double> >(
        new ddf::function_call<double>(DS, 
            predict, /* predict->clone() */
            var_l));

    loss->eval(y);
    expect_true(y[0] > 0, "evaluation of pooling expr loss");

    auto input_diff = ddf::finite_diff(loss.get(), var_x);

    // backprop gradient
    ddf::backpropagation<double> bprop;
    ddf::reset_delta<double> reset;
    loss->apply(&reset);
    loss->eval(y);
    loss->apply(&bprop);

    bool b_input_diff = ddf::vector_matrix_diff(var_x->delta, input_diff);
    expect_true(!b_input_diff, "pooling bprop input diff");
}

void test_pool_op_1(int w, int h, int d, int extent, int stride, int padding)
{
    auto op_pool = new ddf::pooling<double>(
        w, h, d,                  // input size
        extent, stride, padding); // extent, stride, padding

    int size_output = op_pool->size_f();
    
    ddf::variable<double> *var_x =
        new ddf::variable<double>("x", ddf::vector<double>(w * h * d));
    var_x->value().fill_rand();

    ddf::variable<double> *var_l = 
        new ddf::variable<double>("l", ddf::vector<double>(size_output));
    var_l->value().fill(0);
    var_l->value()[2] = 1;

    ddf::math_expr<double> *predict = new ddf::function_call<double>(
        op_pool, var_x);

    auto DS = new ddf::softmax_cross_entropy_with_logits<double>();
    auto loss = std::shared_ptr<ddf::math_expr<double> >(
        new ddf::function_call<double>(DS, 
            predict, /* predict->clone() */
            var_l));

    ddf::vector<double> y;
    loss->eval(y);
    expect_true(y[0] > 0, "evaluation of pooling expr loss");

    auto input_diff = ddf::finite_diff(loss.get(), var_x);

    // backprop gradient
    ddf::backpropagation<double> bprop;
    ddf::reset_delta<double> reset;
    loss->apply(&reset);
    loss->eval(y);
    loss->apply(&bprop);

    bool b_input_diff = ddf::vector_matrix_diff(var_x->delta, input_diff);
    expect_true(!b_input_diff, "pooling bprop input diff");
}

int main(int argc, char *argv[])
{
    printf("Patchouli Go!\n");
    test_nd_array();
    for (int k = 0; k < 5; k++) {
        test_softmax_op<double>(10, -100, 100);
    }
    for (int k = 0; k < 5; k++) {
        test_linear_model<double>();
    }
    test_conv_op_0();
    test_conv_op_1();
    for (int k = 0; k < 5; k++) {
        test_conv_op_2(5, 5, 3, 3, 3, 2, 2, 1);
    }
    for (int k = 0; k < 5; k++) {
        test_conv_fc_relu();
    }
    test_pool_op();
    for (int k = 0; k < 5; k++) {
        test_pool_op_1(5, 5, 3, 3, 2, 1);
        test_pool_op_1(5, 5, 3, 3, 2, 0);
        test_pool_op_1(4, 4, 1, 2, 1, 0);
        test_pool_op_1(4, 4, 1, 3, 1, 0);
    }

    return 0;
}
