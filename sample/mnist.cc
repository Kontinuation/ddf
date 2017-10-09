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

bool load_image_data(const char *data_file, ddf::matrix<float> &td) {
    dataset ds(data_file);
    if (!ds.load(data_file)) {
        logging::error("Failed to load image data");
        return false;
    }

    dataset::element_type et = ds.elem_type();
    int dim = ds.dimension();
    if (dim != 3 || et != dataset::UNSIGNED_BYTE) {
        logging::error("Image is not 3-dim unsigned byte array");
        return false;
    }

    int n_samples = ds.shape(0);
    int width = ds.shape(1);
    int height = ds.shape(2);
    int sample_size = width * height;
    td.resize(n_samples, sample_size);
    uint8_t *p_val = (uint8_t *) ds.val();
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < sample_size; j++) {
            td(i, j) = (*p_val / 255.0 - 0.5);
            p_val++;
        }
    }
    return true;
}

bool load_label_data(const char *data_file, ddf::matrix<float> &td) {
    dataset ds(data_file);
    if (!ds.load(data_file)) {
        logging::error("Failed to load label data");
        return false;
    }

    dataset::element_type et = ds.elem_type();
    int dim = ds.dimension();
    if (dim != 1 || et != dataset::UNSIGNED_BYTE) {
        logging::error("Label is not 1-dim unsigned byte array");
        return false;
    }

    int n_samples = ds.shape(0);
    uint8_t *p_val = (uint8_t *) ds.val();
    uint8_t n_classes = *std::max_element(p_val, p_val + n_samples) + 1;
    td.resize(n_samples, n_classes);
    td.fill(0);
    for (int i = 0; i < n_samples; i++) {
        td(i, *p_val) = 1.0;
        p_val++;
    }
    return true;
}


int main(int argc, char *argv[]) {
    if (argc < 6) {
        printf(
            "usage: %s train train_data train_label model_type out_model_file\n"
            "       %s predict test_data test_label model_type model_file\n",
            argv[0], argv[0]);
        return 0;
    }

    const char *cmd = argv[1];
    const char *data_file = argv[2];
    const char *label_file = argv[3];
    const char *model_type = argv[4];
    const char *model_file = argv[5];
    float alpha = 0.1;
    if (argc > 6) {
        alpha = atof(argv[6]);
    }

    ddf::matrix<float> images(0,0), labels(0,0);
    load_image_data(data_file, images);
    load_label_data(label_file, labels);
    int n_samples = images.shape(0);
    int dimension = images.shape(1);
    int n_classes = labels.shape(1);

    if (images.shape(0) != labels.shape(0)) {
        logging::error("size of image data and label data does not match");
        return -1;
    }
    if (dimension != 28 * 28) {
        logging::error("size of image is not 28 * 28");
        return -1;
    }
    if (n_classes != 10) {
        logging::error("number of classes is not 10");
        return -1;
    }

    int n_train_samples = 10000;
    int n_test_samples = 1000;
    ddf::matrix<float> xs(n_train_samples, dimension, &images(0,0));
    ddf::matrix<float> ls(n_train_samples, n_classes, &labels(0,0));
    ddf::matrix<float> txs(n_test_samples, dimension, &images(n_train_samples, 0));
    ddf::matrix<float> tls(n_test_samples, n_classes, &labels(n_train_samples,0));

    logging::info(
        "dimension: %d, n_samples: %d, n_classes: %d",
        dimension, n_samples, n_classes);

    // show some data
    for (int i = 0; i < 10; i++) {
        show_fea_as_image(&xs(i,0), 28, 28);
        printf("%s\n", ddf::vector<float>(10, &ls(i,0)).to_string().c_str());
    }

    // show some data
    for (int i = 0; i < 10; i++) {
        show_fea_as_image(&txs(i,0), 28, 28);
        printf("%s\n", ddf::vector<float>(10, &tls(i,0)).to_string().c_str());
    }
    
    ddf::variable<float> *var_x = 
        new ddf::variable<float>("x", ddf::vector<float>(dimension));
    ddf::variable<float> *var_l = 
        new ddf::variable<float>("l", ddf::vector<float>(n_classes));

    ddf::math_expr<float> *predict = nullptr;
    if (!strcmp(model_type, "conv")) {
        predict = conv_model(var_x, var_l, xs, ls);
    } else if (!strcmp(model_type, "fc2")) {
        predict = fc_2_model(var_x, var_l, xs, ls, 20);
    } else if (!strcmp(model_type, "fc1")) {
        predict = fc_1_model(var_x, var_l, xs, ls);
    } else {
        logging::error(
            "unknown model type: %s, should be conv, fc2 or fc1",
            model_type);
        return -1;
    }

    auto DS = new ddf::softmax_cross_entropy_with_logits<float>();
    auto loss = std::unique_ptr<ddf::math_expr<float> >(
        new ddf::function_call<float>(
            DS, predict, var_l));
        
    // construct optimizer
    ddf::optimizer_bprop<float> optimizer;
    std::map<std::string, ddf::matrix<float> > feed_dict = {
        {"x", xs },
        {"l", ls }
    };
    optimizer.minimize(loss.get(), &feed_dict);

    // perform iterative optimization to reduce training loss
    optimizer.set_learning_rate(alpha);
    ddf::vector<float> y;
    for (int iter = 0; iter < 100000; iter++) {
        clock_t start = clock();
        optimizer.step(10);
        clock_t end = clock();
        logging::info("iter: %d, loss: %f, cost: %f sec",
            iter, optimizer.loss(),
            (double)(end - start) / CLOCKS_PER_SEC);

        if (iter % 10 == 0) {
            // evaluate model performance on test samples
            auto &vec_x = var_x->value();
            auto &vec_l = var_l->value();
            int n_correct = 0;
            for (int i = 0; i < n_test_samples; i++) {
                vec_x.copy_from(&txs(i, 0));
                vec_l.copy_from(&tls(i, 0));
                predict->eval(y);
                int pred_l = std::distance(
                    &y[0],
                    std::max_element(&y[0], &y[0] + n_classes));
                int actual_l = std::distance(
                    &vec_l[0],
                    std::max_element(&vec_l[0], &vec_l[0] + n_classes));
                n_correct += (pred_l == actual_l);
            }
            printf("accuracy: %f\n", (double) n_correct / n_test_samples);
        }
    }

    return 0;
}
