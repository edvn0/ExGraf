#include <exgraf.hpp>

#include <algorithm>
#include <armadillo>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>

#include <unordered_map>
#include <vector>

using namespace ExGraf;

auto main() -> int {
  using T = double;
  std::size_t batch_size = 32;
  std::size_t input_dim = 784; // 28*28
  std::size_t hidden_dim = 256;
  std::size_t num_classes = 10;
  std::size_t epochs = 10;

  try {
    auto &&[train_images, train_labels] =
        ExGraf::MNIST::load_mnist("https://raw.githubusercontent.com/fgnt/"
                                  "mnist/master/t10k-images-idx3-ubyte.gz",
                                  "https://raw.githubusercontent.com/fgnt/"
                                  "mnist/master/t10k-labels-idx1-ubyte.gz");

    auto &&[train_images_tf, train_labels_tf] =
        ExGraf::MNIST::load_mnist_taskflow(
            "https://raw.githubusercontent.com/fgnt/"
            "mnist/master/t10k-images-idx3-ubyte.gz",
            "https://raw.githubusercontent.com/fgnt/"
            "mnist/master/t10k-labels-idx1-ubyte.gz");
    (void)train_images_tf;
    (void)train_labels_tf;

    auto adam = std::make_unique<AdamOptimizer<T>>(0.001, 0.9, 0.999);
    auto sgd = std::make_unique<SgdOptimizer<T>>(0.001);
    Model<T> model(input_dim, hidden_dim, num_classes, std::move(sgd));

    auto total_samples = train_images.data->n_rows;

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
      info("\n[Epoch {}]", epoch + 1);

      arma::uvec indices =
          arma::linspace<arma::uvec>(0, total_samples - 1, total_samples);
      indices = arma::shuffle(indices);

      for (std::size_t start_idx = 0; start_idx < total_samples;
           start_idx += batch_size) {
        std::size_t end_idx =
            std::min(start_idx + batch_size, std::size_t(total_samples));
        std::size_t curr_batch_size = end_idx - start_idx;

        auto batch_indices = indices.subvec(start_idx, end_idx - 1);

        arma::Mat<T> x_batch(curr_batch_size, input_dim);
        arma::Mat<T> y_batch(curr_batch_size, num_classes);

        for (std::size_t i = 0; i < curr_batch_size; ++i) {
          auto idx = batch_indices(i);
          x_batch.row(i) = train_images.data->row(idx);
          y_batch.row(i) = train_labels.data->row(idx);
        }

        Tensor batch_x(x_batch);
        Tensor batch_y(y_batch);

        auto output = model.forward(batch_x);
        T loss = model.compute_loss(output, batch_y);
        model.backward(output);
        model.step();
        model.zero_grad();

        info("[Batch {}/{}] Loss: {}", start_idx / batch_size + 1,
             total_samples / batch_size, loss);
      }
    }
  } catch (const std::exception &e) {
    error("Exception: {}", e.what());
    return 1;
  }
}