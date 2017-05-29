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

        int n_hidden = 20;
        int len_w0 = n_hidden * dimension;
        int len_b0 = n_hidden;
        int len_w1 = n_classes * n_hidden;
        int len_b1 = n_classes;
        float *w0 = new float[len_w0];
        float *b0 = new float[len_b0];
        float *w1 = new float[len_w1];
        float *b1 = new float[len_b1];
        float *x = new float[dimension];
        float *l = new float[n_classes];

        ddf::variable<float> *var_w0 =
            new ddf::variable<float>("w0", ddf::vector<float>(len_w0, w0));
        ddf::variable<float> *var_b0 =
            new ddf::variable<float>("b0", ddf::vector<float>(len_b0, b0));
        ddf::variable<float> *var_w1 =
            new ddf::variable<float>("w1", ddf::vector<float>(len_w1, w1));
        ddf::variable<float> *var_b1 =
            new ddf::variable<float>("b1", ddf::vector<float>(len_b1, b1));

        // initial value of hyper parameters
        var_w0->value().fill_rand();
        var_b0->value().fill_rand();
        var_w1->value().fill_rand();
        var_b1->value().fill_rand();
        
        ddf::variable<float> *var_x = 
            new ddf::variable<float>("x", ddf::vector<float>(dimension, x));
        ddf::variable<float> *var_l = 
            new ddf::variable<float>("l", ddf::vector<float>(n_classes, l));

        // predict: w1 * (relu(w0 * x + b0)) + b1
        ddf::matrix_mult<float> matmul_0, matmul_1;
        ddf::relu<float> relu_0;
        ddf::math_expr<float> *predict =
            new ddf::addition<float>(
                new ddf::function_call<float>(
                    &matmul_1,
                    var_w1, 
                    new ddf::function_call<float>(
                        &relu_0,
                        new ddf::addition<float>(
                            new ddf::function_call<float>(
                                &matmul_0,
                                var_w0, 
                                var_x),
                            var_b0))),
                var_b1);

        // loss: DS(predict, l)
        ddf::softmax_cross_entropy_with_logits<float> DS;
        ddf::math_expr<float> *loss =
            new ddf::function_call<float>(&DS, 
                predict, /* predict->clone() */
                var_l);

        ddf::math_expr<float> *dloss_dw0 = loss->derivative("w0");
        ddf::math_expr<float> *dloss_db0 = loss->derivative("b0");
        ddf::math_expr<float> *dloss_dw1 = loss->derivative("w1");
        ddf::math_expr<float> *dloss_db1 = loss->derivative("b1");
        
        logging::info("predict: %s", predict->to_string().c_str());
        logging::info("loss: %s", loss->to_string().c_str());
        logging::info("dloss_dw0: %s", dloss_dw0->to_string().c_str());
        logging::info("dloss_db0: %s", dloss_db0->to_string().c_str());
        logging::info("dloss_dw1: %s", dloss_dw1->to_string().c_str());
        logging::info("dloss_db1: %s", dloss_db1->to_string().c_str());
        
        ddf::vector<float> sum_dw0(len_w0);
        ddf::vector<float> sum_db0(len_b0);
        ddf::vector<float> sum_dw1(len_w1);
        ddf::vector<float> sum_db1(len_b1);
        ddf::matrix<float> dw0(0,0);
        ddf::matrix<float> db0(0,0);
        ddf::matrix<float> dw1(0,0);
        ddf::matrix<float> db1(0,0);
        
        ddf::vector<float> c(0);
        float alpha = 0.5;
        
        n_samples = 1000;
        
        logging::info("len_w0: %d, len_b0: %d, dimension: %d, n_samples: %d, n_classes: %d",
            len_w0, len_b0, dimension, n_samples, n_classes);

        float sum_loss = 0;
        for (int k = 0; k < n_samples; k++) {
            std::copy_n(fea + k * dimension, dimension, x);
            std::fill_n(l, n_classes, 0);
            l[label[k]] = 1;
            loss->eval(c);
            sum_loss += c[0];
        }
        logging::info("initial loss: %f", sum_loss);

        for (int iter = 0; iter < 10; iter++) {
            clock_t start = clock();
            sum_dw0.fill(0);
            sum_db0.fill(0);
            sum_dw1.fill(0);
            sum_db1.fill(0);
            
            for (int k = 0; k < n_samples; k++) {

                // copy training data to placeholder
                std::copy_n(fea + k * dimension, dimension, x);

                // one-hot encoding for label
                std::fill_n(l, n_classes, 0);
                l[label[k]] = 1;

                // gradient descent
                dw0.fill(0); db0.fill(0);
                dw1.fill(0); db1.fill(0);
                dloss_dw0->grad(dw0); dloss_db0->grad(db0);
                dloss_dw1->grad(dw1); dloss_db1->grad(db1);
                sum_dw0 += ddf::vector<float>(len_w0, dw0.raw_data());
                sum_db0 += ddf::vector<float>(len_b0, db0.raw_data());
                sum_dw1 += ddf::vector<float>(len_w1, dw1.raw_data());
                sum_db1 += ddf::vector<float>(len_b1, db1.raw_data());
            }

            sum_dw0 *= (alpha / n_samples);
            sum_db0 *= (alpha / n_samples);
            sum_dw1 *= (alpha / n_samples);
            sum_db1 *= (alpha / n_samples);
            
            var_w0->_val -= sum_dw0;
            var_b0->_val -= sum_db0;
            var_w1->_val -= sum_dw1;
            var_b1->_val -= sum_db1;

            sum_loss = 0;
            for (int k = 0; k < n_samples; k++) {
                std::copy_n(fea + k * dimension, dimension, x);
                std::fill_n(l, n_classes, 0);
                l[label[k]] = 1;
                loss->eval(c);
                sum_loss += c[0];
            }

            clock_t end = clock();
            logging::info("iter: %d, loss: %f, cost: %f sec",
                iter, sum_loss,
                (double)(end - start) / CLOCKS_PER_SEC);
            
            // logging::info("w0: %s", ddf::vector<float>(10, var_w0->value().raw_data()).to_string().c_str());
            // logging::info("b0: %s", ddf::vector<float>(10, var_b0->value().raw_data()).to_string().c_str());
            // logging::info("w1: %s", ddf::vector<float>(10, var_w1->value().raw_data()).to_string().c_str());
            // logging::info("b1: %s", ddf::vector<float>(10, var_b1->value().raw_data()).to_string().c_str());
        }
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
