#include <signal.h>
#include <cstring>
#include "deep_dark_fantasy.hh"
#include "logging.hh"
#include "dataset.hh"
#include "models.hh"

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
    if (argc < 4) {
        printf(
            "usage: %s train train_dataset out_model_file\n"
            "       %s predict test_dataset model_file\n",
            argv[0], argv[0]);
        return 0;
    }

    const char *cmd = argv[1];
    const char *dataset_file = argv[2];
    const char *model_file = argv[3];
    float alpha = 0.1;
    if (argc > 4) {
        alpha = atof(argv[4]);
    }
    if (!strcmp(cmd, "train")) {
        // -- read training data from file --
        dataset d_train(dataset_file);
        float *fea = d_train.features();
        int *label = d_train.labels();
        int dimension = d_train.dimension();
        int n_samples = d_train.n_samples();
        int n_classes = d_train.n_classes();

        // // truncate the training data for faster shakedown run
        // n_samples = std::min(1000, n_samples);

        // prepare feeded data 
        ddf::matrix<float> xs(n_samples, dimension, fea);
        ddf::matrix<float> ls(n_samples, n_classes);
        ls.fill(0);
        for (int k = 0; k < n_samples; k++) {
            ls(k, label[k]) = 1;
        }

        logging::info(
            "dimension: %d, n_samples: %d, n_classes: %d",
            dimension, n_samples, n_classes);

        // -- train this model using optimizer defined in train.hh --
        // auto loss = std::unique_ptr<ddf::math_expr<float> >(fc_1_model(xs, ls));
        // auto loss = std::unique_ptr<ddf::math_expr<float> >(fc_2_model(xs, ls, 20));
        auto loss = std::unique_ptr<ddf::math_expr<float> >(conv_model(xs, ls));
        
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
        optimizer.set_learning_rate(alpha);
        for (int iter = 0; iter < 100000; iter++) {
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
