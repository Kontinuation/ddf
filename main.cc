#include <signal.h>
#include <cstring>
#include "logging.hh"
#include "dataset.hh"
#include "deep_dark_fantasy.hh"

void show_fea_as_image(const float *fea, int w, int h, float threshold = -0.2) {
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            putc(fea[i * w + j] < threshold? ' ': '.', stdout);
            putc(' ', stdout);
        }
        printf("\n");
    }
}

volatile bool g_signal_quit = false;

int main(int argc, char *argv[]) {
    if (argc != 4) {
        printf(
            "usage: %s train train_dataset out_model_file\n"
            "       %s predict test_dataset model_file\n",
            argv[0], argv[0]);
        return 0;
    }

    const char *cmd = argv[1];
    const char *dataset_file = argv[2];
    const char *model_file = argv[3];
    if (!strcmp(cmd, "train")) {
        dataset d_train(dataset_file);
        float *fea = d_train.features();
        int *label = d_train.labels();
        int dimension = d_train.dimension();
        int n_samples = d_train.n_samples();
        int n_classes = d_train.n_classes();

        int len_w0 = n_classes * dimension;
        int len_b0 = n_classes;
        float *w0 = new float[len_w0];
        float *b0 = new float[len_b0];
        float *x = new float[dimension];
        float *l = new float[n_classes];
        std::fill_n(w0, len_w0, 0.1);
        std::fill_n(b0, len_b0, 0.1);

        ddf::variable<float> *var_w =
            new ddf::variable<float>("w", ddf::vector<float>(len_w0, w0));
        ddf::variable<float> *var_b =
            new ddf::variable<float>("b", ddf::vector<float>(len_b0, b0));

        // predict: w * x + b
        ddf::matrix_mult<float> matmul(ddf::vector<float>(dimension, x));
        ddf::math_expr<float> *predict =
            new ddf::addition<float>(
                new ddf::function_call<float>(
                    &matmul,
                    var_w),
                var_b);

        // cost: DS(predict, l)
        ddf::softmax_cross_entropy_with_logits<float>
            DS(ddf::vector<float>(n_classes, l));
        ddf::math_expr<float> *cost =
        new ddf::function_call<float>(&DS, predict /* predict->clone() */);

        // training process: gradient descent
        // signal(SIGINT, [](int) {
        //         g_signal_quit = true;
        //     });

        ddf::math_expr<float> *dcost_dw = cost->derivative("w");
        ddf::math_expr<float> *dcost_db = cost->derivative("b");
        ddf::vector<float> sum_dw(len_w0);
        ddf::vector<float> sum_db(n_classes);
        ddf::matrix<float> dw(0,0);
        ddf::matrix<float> db(0,0);
        ddf::vector<float> c(0);
        float alpha = 0.00001;
        // float alpha = 0.000003;

        printf("len_w0: %d, len_b0: %d, dimension: %d, n_samples: %d\n", len_w0, len_b0, dimension, n_samples);

        // float sum_cost = 0;
        // for (int k = 0; k < n_samples; k++) {
        //     std::copy_n(fea + k * dimension, dimension, x);
        //     std::fill_n(l, n_classes, 0);
        //     l[label[k]] = 1;

        //     cost->eval(c);
        //     sum_cost += c[0];
        // }
        // printf("initial cost: %f\n", sum_cost / n_samples);

        // while (!g_signal_quit) {
        for (int iter = 0; iter < 10; iter++) {
            for (int k = 0; k < n_samples; k++) {

                // printf("sample: %d\n", k);
                // copy training data to placeholder
                std::copy_n(fea + k * dimension, dimension, x);

                // one-hot encoding for label
                std::fill_n(l, n_classes, 0);
                l[label[k]] = 1;
                
                dcost_dw->grad(dw);
                dcost_db->grad(db);
                sum_dw += ddf::vector<float>(len_w0, dw.raw_data());
                sum_db += ddf::vector<float>(len_b0, dw.raw_data());
            }

            sum_dw *= (alpha / n_samples);
            sum_db *= (alpha / n_samples);
            var_w->_val -= sum_dw;
            var_b->_val -= sum_db;

            // cost->eval(c);
            // c *= n_samples;
            // printf("cost: %s\n", c.to_string().c_str());

            // sum_cost = 0;
            // for (int k = 0; k < n_samples; k++) {
            //     std::copy_n(fea + k * dimension, dimension, x);
            //     std::fill_n(l, n_classes, 0);
            //     l[label[k]] = 1;
            //     cost->eval(c);
            //     sum_cost += c[0];
            // }
            // printf("cost [%d]: %f\n", iter, sum_cost / n_samples);

            // printf("dw: %s\ndb: %s\n", sum_dw.to_string().c_str(), sum_db.to_string().c_str());
        }
        // }
        

        // logging::info("initial loss: %f", lg.loss(theta));

        // gradient_descent<std::function<void (float *, float *)> > gd;
        // signal(SIGINT, [](int) {
        //         g_signal_quit = true;
        //     });
        // int superstep = 0;
        // while (!g_signal_quit) {
        //     clock_t start = clock();
        //     gd.solve(
        //         std::bind(&logistic_opt<float>::d_opt, lg, _1, _2),
        //         theta, lg.fea_dim * lg.n_classes);
        //     clock_t end = clock();
        //     logging::info("superstep: %d, loss: %f, cost: %f sec",
        //         ++superstep, lg.loss(theta),
        //         (double)(end - start) / CLOCKS_PER_SEC);
        // }

        // logging::info("saving model to file ...");
        // int fd = open(model_file, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
        // if (fd != 0) {
        //     int model_bytes = len_theta * sizeof(float);
        //     if (write(fd, theta, model_bytes) != model_bytes) {
        //         logging::error("failed saving model to file");
        //     }
        //     close(fd);
        // } else {
        //     logging::error("failed to open model file");
        // }

        // delete [] theta;
    }
    else if (!strcmp(cmd, "predict")) {
        dataset d_test(dataset_file);
        int len_theta = d_test.n_classes() * d_test.dimension();
        float *theta = new float[len_theta];
        int fd = open(model_file, O_RDONLY);
        if (fd != 0) {
            int model_bytes = len_theta * sizeof(float);
            if (model_bytes != read(fd, theta, model_bytes)) {
                logging::error("failed to read model file");
                close(fd);
                return -1;
            }
            close(fd);

            // predict all samples in dataset

        } else {
            logging::error("failed to open model file");
            return -1;
        }
        delete [] theta;
    }
    else {
        printf("unknown command %s\n", cmd);
    }
    return 0;
}
