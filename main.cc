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
}

void test_matmul(void)
{
    double mat[3][4] = {
        { 1.0, 2.0, 3.0, 4.0 },
        { 5.0, 6.0, 7.0, 8.0 },
        { 9.0, 10.0, 11.0, 12.0 },
    };
    double v[4] = {1.0, 2.0, 3.0, 4.0};
    ddf::vector<double> w(12, (double *) mat);
    printf("w: %s\n", w.to_string().c_str());

    ddf::matrix_mult<double> op_matmul(ddf::vector<double>(4, v));
    ddf::vector<double> y(0);
    op_matmul.f_x(w, y);
    printf("f_x: %s\n", y.to_string().c_str());

    for (int k = 0; k < w.size(); k++) {
        op_matmul.f_x(w, y);
        printf("f_x%d: %s\n", k, y.to_string().c_str());
    }

    ddf::matrix<double> D(0,0);
    op_matmul.Df_x(w, D);
    printf("Df_x: %s\n", D.to_string().c_str());
}

void test_softmax(void)
{
    double label[4] = { 0.0, 0.0, 1.0, 0.0 }; // 3rd class
    double w_data[4] = {1.0, 2.0, 3.0, 4.0};
    ddf::vector<double> w(4, w_data);
    ddf::vector<double> y(1);
    ddf::vector<double> l(4, label);
    printf("w: %s, l: %s\n", w.to_string().c_str(), l.to_string().c_str());
    ddf::softmax_cross_entropy_with_logits<double> op_DS(l);
    op_DS.f_x(w, y);
    printf("f_x: %s\n", y.to_string().c_str());
    op_DS.f_x(w, y);
    printf("f_x: %s\n", y.to_string().c_str());
    op_DS.f_x(w, y);
    printf("f_x: %s\n", y.to_string().c_str());
    op_DS.f_x(w, y);
    printf("f_x: %s\n", y.to_string().c_str());

    ddf::matrix<double> D(0,0);
    op_DS.Df_x(w, D);
    printf("Df_x: %s\n", D.to_string().c_str());

    op_DS.f_x(w, y);
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

    // predict: w * x + b
    ddf::matrix_mult<double> matmul(ddf::vector<double>(4, x));
    ddf::math_expr<double> *predict =
        new ddf::addition<double>(
            new ddf::function_call<double>(
                &matmul,
                new ddf::variable<double>("w", ddf::vector<double>(12, w0))),
            new ddf::variable<double>("b", ddf::vector<double>(3, b0)));

    // cost: DS(predict, l)
    ddf::softmax_cross_entropy_with_logits<double> DS(ddf::vector<double>(3, l));
    ddf::math_expr<double> *cost =
        new ddf::function_call<double>(&DS, predict->clone());

    printf("predict: %s\n", predict->to_string().c_str());
    printf("cost: %s\n", cost->to_string().c_str());

    ddf::vector<double> y(3);
    predict->eval(y);
    printf("predict result: %s\n", y.to_string().c_str());

    cost->eval(y);
    printf("cost result: %s\n", y.to_string().c_str());

    ddf::math_expr<double> *dcost = cost->derivative("w");
    printf("d cost: %s\n", dcost->to_string().c_str());
    ddf::matrix<double> dm(0,0);
    dcost->grad(dm);
    printf("dcost result: %s\n", dm.to_string().c_str());
}

int main(int argc, char *argv[])
{
    printf("Patchouli Go!\n");
    test_nd_array();
    test_softmax();
    test_matmul();
    test_expr();
    return 0;
}
