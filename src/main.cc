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

template <typename numeric_type>
ddf::math_expr<numeric_type> *fc_model(
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
    
    // ddf::matrix_mult<numeric_type> matmul_0, matmul_1;
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
        // -- read training data from file --
        dataset d_train(dataset_file);
        float *fea = d_train.features();
        int *label = d_train.labels();
        int dimension = d_train.dimension();
        int n_samples = d_train.n_samples();
        int n_classes = d_train.n_classes();

        // truncate the training data for faster shakedown run
        n_samples = std::min(1000, n_samples);

        // prepare feeded data 
        ddf::matrix<float> xs(n_samples, dimension, fea);
        ddf::matrix<float> ls(n_samples, n_classes);
        ls.fill(0);
        for (int k = 0; k < n_samples; k++) {
            ls(k, label[k]) = 1;
        }

        // -- train this model using optimizer defined in train.hh --
        auto loss = std::unique_ptr<ddf::math_expr<float> >(fc_model(xs, ls, 20));
        
        // construct optimizer
        ddf::optimizer_bprop<float> optimizer;
        std::map<std::string, ddf::matrix<float> > feed_dict = {
            {"x", xs },
            {"l", ls }
        };
        optimizer.minimize(loss.get(), &feed_dict);

        // initial loss
        float training_loss = optimizer.loss();
        logging::info("initial loss: %f", training_loss);

        // perform iterative optimization to reduce training loss
        for (int iter = 0; iter < 10; iter++) {
            clock_t start = clock();
            optimizer.step(1);
            clock_t end = clock();
            logging::info("iter: %d, loss: %f, cost: %f sec",
                iter, optimizer.loss(),
                (double)(end - start) / CLOCKS_PER_SEC);
        }

        // TODO: -- save trained model to file --
    }
    else if (!strcmp(cmd, "predict")) {
        dataset d_test(dataset_file);
        int len_theta = d_test.n_classes() * d_test.dimension();
        float *theta = new float[len_theta];
        int fd = open(model_file, O_RDONLY);
        if (fd != 0) {
            // TODO: load model from file
            
            close(fd);

            // TODO: predict all samples in dataset

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
