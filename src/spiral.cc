#include <signal.h>
#include <cstring>
#include "deep_dark_fantasy.hh"
#include "logging.hh"
#include "models.hh"


int main(int argc, char *argv[]) {
    int N = 100;
    int K = 3;
    double alpha = 0.1;
    if (argc > 1) {
        alpha = atof(argv[1]);
    }

    ddf::matrix<double> xs(N * K, 2);
    ddf::matrix<double> ls(N * K, K);
    ls.fill(0);

    double r_step = 1.0 / N;
    for (int j = 0; j < K; j++) {
        for (int i = 0; i < N; i++) {
            int ix = N * j + i;
            double r = i * r_step;
            double t = j * 4 + i * r_step * 4 + ((double) rand() / RAND_MAX * 0.2);
            xs(ix, 0) = r * std::sin(t);
            xs(ix, 1) = r * std::cos(t);
            ls(ix, j) = 1;
            // printf("%f %f %d\n", xs(ix, 0), xs(ix, 1), j);
        }
    }

    ddf::variable<double> *var_x = 
        new ddf::variable<double>("x", ddf::vector<double>(2));
    ddf::variable<double> *var_l = 
        new ddf::variable<double>("l", ddf::vector<double>(K));
    
    // auto predict = fc_1_model(var_x, var_l, xs, ls);
    auto predict = fc_2_model(var_x, var_l, xs, ls, 20);

    auto DS = new ddf::softmax_cross_entropy_with_logits<double>();
    auto loss = std::unique_ptr<ddf::math_expr<double> >(
        new ddf::function_call<double>(
            DS, predict, var_l));


    // construct optimizer
    ddf::optimizer_bprop<double> optimizer;
    std::map<std::string, ddf::matrix<double> > feed_dict = {
        {"x", xs },
        {"l", ls }
    };
    optimizer.minimize(loss.get(), &feed_dict);

    // initial loss
    double training_loss = optimizer.loss();
    logging::info("initial loss: %f", training_loss);

    // perform iterative optimization to reduce training loss
    optimizer.set_learning_rate(alpha);
    for (int iter = 0; iter < 1000; iter++) {
        clock_t start = clock();
        optimizer.step(1);
        clock_t end = clock();
        logging::info("iter: %d, loss: %f, cost: %f sec",
            iter, optimizer.loss(),
            (double)(end - start) / CLOCKS_PER_SEC);
    }

    return 0;
}
