#include "deep_dark_fantasy.hh"

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

    printf("w: %s\n", w.to_string().c_str());
    printf("w1: %s\n", w1.to_string().c_str());
    printf("w2: %s\n", w2.to_string().c_str());

    ddf::vector<double> v(3, (double *) data);
    auto v1 = v;
    auto v2 = v1.clone();
    v[1] = 1000;
    printf("v: %s\n", v.to_string().c_str());
    printf("v1: %s\n", v1.to_string().c_str());
    printf("v2: %s\n", v2.to_string().c_str());
    v1 += v2;
    printf("v1: %s\n", v1.to_string().c_str());
    printf("v1 .* v2: %f\n", v1.dot(v2));

    ddf::nd_array<double, 3> cube({2,2,3}, (double *) data);
    printf("cube: %s\n", cube.to_string().c_str());

    v1 = v2.clone();
    printf("v1: %s, v2: %s\n", v1.to_string().c_str(), v2.to_string().c_str());

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
    printf("m2 = m0 * m1: %s\n", m2.to_string().c_str());
}

void test_matmul(void)
{
    double mat[3][4] = {
        { 1.0, 2.0, 3.0, 4.0 },
        { 5.0, 6.0, 7.0, 8.0 },
        { 9.0, 10.0, 11.0, 12.0 },
    };
    double v0[4] = {1.0, 2.0, 3.0, 4.0};
    ddf::vector<double> w(12, (double *) mat);
    ddf::vector<double> v(4, v0);
    printf("w: %s\n", w.to_string().c_str());
    printf("v: %s\n", v.to_string().c_str());

    ddf::matrix_mult<double> op_matmul;
    ddf::vector<double> y(0);
    op_matmul.f_x(w, v, y);
    printf("f_x: %s\n", y.to_string().c_str());

    ddf::matrix<double> D(0,0);
    op_matmul.Df_x(w, v, 0, D);
    printf("Df_x: %s\n", D.to_string().c_str());
}

void test_softmax(void)
{
    logging::info("test softmax");
    double label[4] = { 0.0, 0.0, 1.0, 0.0 }; // 3rd class
    double w_data[4] = {1.0, 2.0, 3.0, 4.0};
    ddf::vector<double> w(4, w_data);
    ddf::vector<double> y(1);
    ddf::vector<double> l(4, label);
    printf("w: %s, l: %s\n", w.to_string().c_str(), l.to_string().c_str());
    ddf::softmax_cross_entropy_with_logits<double> op_DS;
    op_DS.prepare(0, w);
    op_DS.prepare(1, l);
    op_DS.ready();
    op_DS.f(y);
    printf("f_x: %s\n", y.to_string().c_str());

    ddf::matrix<double> D(0,0);
    op_DS.Df(0, D);
    printf("Df_x: %s\n", D.to_string().c_str());

    op_DS.f(y);
    double delta = 1e-10;
    ddf::vector<double> y1(0);
    for (int k = 0; k < 4; k++) {
        ddf::vector<double> w1 = w.clone();
        w1[k] += delta;
        op_DS.f_x(w1, y1);
        // printf("w: %s, w1: %s\n", w.to_string().c_str(), w1.to_string().c_str());
        // printf("f_x%d: %f, %f\n", k, y1[0], y[0]);
        double dfx = (y1[0] - y[0]) / delta;
        printf("df_x%d: %f\n", k, dfx);
    }
}

void test_expr(void)
{
    // training sample
    double x[4] = {1.0, 2.0, 3.0, 4.0}; // sample
    double l[3] = {0.0, 1.0, 0.0};      // label

    // initial parameters
    double w0[12] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };
    double b0[3] = { 7, 13, 42 };

    ddf::variable<double> *var_w =
        new ddf::variable<double>("w", ddf::vector<double>(12, w0));
    ddf::variable<double> *var_x =
        new ddf::variable<double>("x", ddf::vector<double>(4, x));
    ddf::variable<double> *var_b =
        new ddf::variable<double>("b", ddf::vector<double>(3, b0));

    // predict: w * x + b
    ddf::matrix_mult<double> matmul;
    ddf::math_expr<double> *predict =
        new ddf::addition<double>(
            new ddf::function_call<double>(
                &matmul,
                var_w, var_x),
            var_b);

    // loss: DS(predict, l)
    ddf::softmax_cross_entropy_with_logits<double> DS;
    ddf::math_expr<double> *loss =
        new ddf::function_call<double>(&DS, 
            predict /* predict->clone() */,
            new ddf::constant<double>(ddf::vector<double>(3, l))
            );

    printf("predict: %s\n", predict->to_string().c_str());
    printf("loss: %s\n", loss->to_string().c_str());

    ddf::vector<double> y(3);
    predict->eval(y);
    printf("predict result: %s\n", y.to_string().c_str());

    loss->eval(y);
    printf("loss result: %s\n", y.to_string().c_str());

    // forward gradient
    ddf::math_expr<double> *dloss = loss->derivative("w");
    printf("d loss: %s\n", dloss->to_string().c_str());
    ddf::matrix<double> dm(0,0);
    dloss->grad(dm);
    printf("dloss result: %s\n", dm.to_string().c_str());
    dm = ddf::finite_diff(loss, var_w);
    printf("dloss finite diff result: %s\n", dm.to_string().c_str());
    delete dloss;

    ddf::math_expr<double> *dloss_b = loss->derivative("b");
    printf("d loss of b: %s\n", dloss_b->to_string().c_str());
    ddf::matrix<double> dm_b(0,0);
    dloss_b->grad(dm_b);
    printf("dloss_b result: %s\n", dm_b.to_string().c_str());
    dm_b = ddf::finite_diff(loss, var_b);
    printf("dloss_b finite diff result: %s\n", dm_b.to_string().c_str());
    delete dloss_b;

    // backprop gradient
    ddf::backpropagation<double> bprop;
    ddf::reset_delta<double> reset;
    loss->apply(&reset);
    loss->eval(y);
    loss->apply(&bprop);
    printf("dloss_w: %s\n", var_w->delta.to_string().c_str());
    printf("dloss_b: %s\n", var_b->delta.to_string().c_str());

    delete loss;
}

template <typename numeric_type>
struct myop_f: ddf::math_op<numeric_type> {
    myop_f(void): ddf::math_op<numeric_type>("f", 1) {
    }

    void prepare(int k_param, const ddf::vector<numeric_type> &v) {
        assert_param_dim(k_param);
        _x = v;
    }

    ddf::vector<numeric_type> get_param(int k_param) {
        assert_param_dim(k_param);
        return _x;
    }

    int size_f() {
        return 3;
    }

    void f(ddf::vector<numeric_type> &y) {
        y.resize(3);
        y[0] = sin(_x[0]);
        y[1] = _x[0] - _x[1];
        y[2] = _x[0] * _x[1];
    }

    void bprop(int k_param, ddf::vector<numeric_type> &dx) {
        assert_param_dim(k_param);
        
    }

    void Df(int k_param, ddf::matrix<numeric_type> &D) {
        assert_param_dim(k_param);
        D.resize(3, 2);
        D(0,0) = cos(_x[0]);
        D(0,1) = 0;
        D(1,0) = 1;
        D(1,1) = -1;
        D(2,0) = _x[1];
        D(2,1) = _x[0];
    }

    ddf::vector<numeric_type> _x;
};

template <typename numeric_type>
struct myop_g: ddf::math_op<numeric_type> {
    myop_g(void): ddf::math_op<numeric_type>("g", 1) {
    }

    void prepare(int k_param, const ddf::vector<numeric_type> &v) {
        assert_param_dim(k_param);
        _x = v;
    }

    ddf::vector<numeric_type> get_param(int k_param) {
        assert_param_dim(k_param);
        return _x;
    }

    int size_f() {
        return 2;
    }

    void f(ddf::vector<numeric_type> &y) {
        y.resize(2);
        y[0] = _x[0] + _x[1];
        y[1] = _x[0] * _x[1];
    }

    void bprop(int k_param, ddf::vector<numeric_type> &dx) {
        assert_param_dim(k_param);
        
    }

    void Df(int k_param, ddf::matrix<numeric_type> &D) {
        assert_param_dim(k_param);
        D.resize(2, 2);
        D(0,0) = 1;
        D(0,1) = 1;
        D(1,0) = _x[1];
        D(1,1) = _x[0];
    }

    ddf::vector<numeric_type> _x;
};

void test_fg()
{
    myop_f<double> f;
    myop_g<double> g;
    double x0[2] = { 0.3, 0.6 };

    ddf::variable<double> *var_x =
        new ddf::variable<double>("x", ddf::vector<double>(2, x0));
    ddf::math_expr<double> *fg =
        new ddf::function_call<double>(
            &f,
            new ddf::function_call<double>(
                &g,
                var_x));

    ddf::math_expr<double> *d_fg = fg->derivative("x");

    printf("fg: %s\n", fg->to_string().c_str());
    printf("d_fg: %s\n", d_fg->to_string().c_str());

    ddf::vector<double> fg_val(0);
    fg->eval(fg_val);
    printf("fg_val: %s\n", fg_val.to_string().c_str());
    fg->eval(fg_val);

    ddf::matrix<double> dfg_val(3, 2);
    d_fg->grad(dfg_val);
    printf("dfg_val: %s\n", dfg_val.to_string().c_str());

    ddf::matrix<double> expected(3,2);
    expected(0,0) = cos(x0[0] + x0[1]);
    expected(0,1) = cos(x0[0] + x0[1]);
    expected(1,0) = 1 - x0[1];
    expected(1,1) = 1 - x0[0];
    expected(2,0) = 2*x0[0]*x0[1] + x0[1]*x0[1];
    expected(2,1) = 2*x0[0]*x0[1] + x0[0]*x0[0];
    printf("expected: %s\n", expected.to_string().c_str());

    double delta = 1e-6;
    ddf::vector<double> fg_val1(0);

    double tmp = var_x->_val[0];
    var_x->_val[0] += delta;
    fg->eval(fg_val1);
    fg_val1 -= fg_val;
    fg_val1 *= (1 / delta);
    printf("dx0: %s\n", fg_val1.to_string().c_str());
    var_x->_val[0] = tmp;
    
    var_x->_val[1] += delta;
    fg->eval(fg_val1);
    fg_val1 -= fg_val;
    fg_val1 *= (1 / delta);
    printf("dx1: %s\n", fg_val1.to_string().c_str());
    var_x->_val[1] = tmp;    

    delete fg;
    delete d_fg;
}

void test_array_opt(void)
{
    // matrix mult
    {
        logging::info("test strided mat mul");
        double w0[12] = {
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
        };
        double x0[4] = {1.0, 2.0, 3.0, 4.0}; // sample
        ddf::vector<double> w(12, w0);
        ddf::vector<double> x(4, x0);
        ddf::matrix_mult<double> matmul;

        matmul.prepare(0, w);
        matmul.prepare(1, x);
        matmul.ready();
        
        ddf::matrix<double> D;
        matmul.Df(0, D);

        printf("D: %s\n", D.to_string().c_str());

        double a0[6] = {
            2, 3, 5,
            1, 4, 6,
        };    
        ddf::matrix<double> A(2, 3, a0);
        ddf::matrix<double> AD(0,0);
        printf("A * D: %s\n", (A * D).to_string().c_str());
        matmul.mult_by_grad(0, A, AD);
        printf("AD: %s\n", AD.to_string().c_str());

        double b0[24] = {
            1,2,3,4,5,6,7,8,9,10,11,12,
            2,3,4,5,6,7,8,9,10,11,12,13,
        };
        ddf::matrix<double> B(12, 2, b0);
        ddf::matrix<double> DB(0,0);
        printf("D * B: %s\n", (D * B).to_string().c_str());

        matmul.prepare(0, x);
        matmul.ready();        
        matmul.mult_grad(0, B, DB);
        printf("DB: %s\n", DB.to_string().c_str());
    }

    // relu
    {
        logging::info("test relu");
        ddf::relu<double> relu;
        double x0[4] = {1.0, 2.0, -3.0, 4.0}; // sample
        double b0[12] = {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
            10, 11, 12,
        };

        ddf::vector<double> x(4, x0);
        relu.prepare(0, x);
        relu.ready();
            
        ddf::matrix<double> B(4,3, b0);
        ddf::matrix<double> D, DB, AD;
        relu.Df(0, D);
        printf("D * B: %s\n", (D * B).to_string().c_str());
        relu.mult_grad(0, B, DB);
        printf("DB: %s\n", DB.to_string().c_str());

        ddf::matrix<double> A(3,4, b0);
        printf("A * D: %s\n", (A * D).to_string().c_str());
        relu.mult_by_grad(0, A, AD);
        printf("AD: %s\n", AD.to_string().c_str());
    }
        
}

void test_relu(void)
{
    double x0[5] = {1.0, -2.0, 3.0, -4.0, -5.0}; // sample
    ddf::vector<double> x(5, x0);
    ddf::relu<double> rl;
    rl.prepare(0, x);
    rl.ready();
    
    ddf::matrix<double> D_rl;
    ddf::vector<double> relu_x;
    rl.f(relu_x);
    rl.Df(0, D_rl);
    printf("relu_x: %s\nD_rl: %s\n", 
        relu_x.to_string().c_str(), D_rl.to_string().c_str());

    rl.slow_Df(0, D_rl);
    printf("slow_D_rl: %s\n",
        D_rl.to_string().c_str());
}

template <typename numeric_type>
void test_expr_visitor(void) {
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
    ddf::matrix_mult<numeric_type> matmul_0, matmul_1;
    ddf::relu<numeric_type> relu_0;
    ddf::math_expr<numeric_type> *predict =
        new ddf::addition<numeric_type>(
            new ddf::function_call<numeric_type>(
                &matmul_1,
                var_w1, 
                new ddf::function_call<numeric_type>(
                    &relu_0,
                    new ddf::addition<numeric_type>(
                        new ddf::function_call<numeric_type>(
                            &matmul_0,
                            var_w0, 
                            var_x),
                        var_b0))),
            var_b1);

    // loss: DS(predict, l)
    ddf::softmax_cross_entropy_with_logits<numeric_type> DS;
    auto loss = std::shared_ptr<ddf::math_expr<numeric_type> >(
        new ddf::function_call<numeric_type>(&DS, 
            predict, /* predict->clone() */
            var_l));

    std::shared_ptr<ddf::math_expr<numeric_type> > dloss_dw0(loss->derivative("w0"));
    std::shared_ptr<ddf::math_expr<numeric_type> > dloss_db0(loss->derivative("b0"));
    std::shared_ptr<ddf::math_expr<numeric_type> > dloss_dw1(loss->derivative("w1"));
    std::shared_ptr<ddf::math_expr<numeric_type> > dloss_db1(loss->derivative("b1"));

    // backprop gradient
    ddf::backpropagation<numeric_type> bprop;
    ddf::reset_delta<numeric_type> reset;
    ddf::vector<numeric_type> y;
    loss->apply(&reset);
    loss->eval(y);
    loss->apply(&bprop);

    ddf::matrix<numeric_type> D_dw0 = ddf::finite_diff(loss.get(), var_w0);
    logging::info("finite diff D_dw0: %s", D_dw0.to_string().c_str());
    dloss_dw0->grad(D_dw0);
    logging::info("auto diff D_dw0: %s", D_dw0.to_string().c_str());
    logging::info("bprop dw0:\n  %s", var_w0->delta.to_string().c_str());

    ddf::matrix<numeric_type> D_dw1 = ddf::finite_diff(loss.get(), var_w1);
    logging::info("finite diff D_dw1: %s", D_dw1.to_string().c_str());
    dloss_dw1->grad(D_dw1);
    logging::info("auto diff D_dw1: %s", D_dw1.to_string().c_str());
    logging::info("bprop dw1:\n  %s", var_w1->delta.to_string().c_str());

    ddf::matrix<numeric_type> D_db0 = ddf::finite_diff(loss.get(), var_b0);
    logging::info("finite diff D_db0: %s", D_db0.to_string().c_str());
    dloss_db0->grad(D_db0);
    logging::info("auto diff D_db0: %s", D_db0.to_string().c_str());
    logging::info("bprop db0:\n  %s", var_b0->delta.to_string().c_str());
    
    ddf::matrix<numeric_type> D_db1 = ddf::finite_diff(loss.get(), var_b1);
    logging::info("finite diff D_db1: %s", D_db1.to_string().c_str());
    dloss_db1->grad(D_db1);
    logging::info("auto diff D_db1: %s", D_db1.to_string().c_str());
    logging::info("bprop db1:\n  %s", var_b1->delta.to_string().c_str());


    ddf::dump_expr_as_dotfile<numeric_type> dump_loss("loss.dot");
    loss->apply(&dump_loss);
    ddf::dump_expr_as_dotfile<numeric_type> dump_dw0("dw0.dot");
    dloss_dw0->apply(&dump_dw0);
    ddf::dump_expr_as_dotfile<numeric_type> dump_dw1("dw1.dot");
    dloss_dw1->apply(&dump_dw1);

    ddf::common_subexpr_elim<numeric_type> cse;
    cse.apply(loss);
    cse.apply(dloss_dw0);
    cse.apply(dloss_dw1);
    cse.apply(dloss_db0);
    cse.apply(dloss_db1);
    ddf::dump_expr_as_dotfile<numeric_type> dump_all("cse_all.dot");
    loss->apply(&dump_all);
    dloss_dw0->apply(&dump_all);
    dloss_dw1->apply(&dump_all);
    dloss_db0->apply(&dump_all);
    dloss_db1->apply(&dump_all);


    // std::shared_ptr<ddf::math_expr<numeric_type> > loss_2(loss->clone());
    // ddf::collect_variable<numeric_type> visitor;
    // loss_2->apply(&visitor);
    // for (auto &s: visitor.vars()) {
    //     logging::info("var: %s", s.first.c_str());
    // }
}

int main(int argc, char *argv[])
{
    printf("Patchouli Go!\n");
    test_nd_array();;
    test_softmax();
    test_matmul();
    test_array_opt();
    test_relu();
    test_expr();
    // test_fg();
    test_expr_visitor<float>();
    test_expr_visitor<double>();
    return 0;
}