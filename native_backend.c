/*
 * U-Ray's built-in CPU kernels.
 *
 * This file has no third-party dependencies.  AVX-512, AVX2, and scalar
 * implementations are selected at runtime so the same source can be built
 * for machines with different x86 capabilities.
 */

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define URAY_X86_GNU 1
#else
#define URAY_X86_GNU 0
#endif

static void gemm_scalar(
    size_t m, size_t k, size_t n,
    const double *restrict a,
    const double *restrict b,
    double *restrict c
) {
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double total = 0.0;
            for (size_t p = 0; p < k; ++p) {
                total += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = total;
        }
    }
}

#if URAY_X86_GNU
__attribute__((target("avx512f,fma")))
static void gemm_avx512(
    size_t m, size_t k, size_t n,
    const double *restrict a,
    const double *restrict b,
    double *restrict c
) {
    size_t i = 0;
    size_t j = 0;

    for (; i + 8 <= m; i += 8) {
        for (j = 0; j + 16 <= n; j += 16) {
            __m512d c00 = _mm512_setzero_pd(), c01 = _mm512_setzero_pd();
            __m512d c10 = _mm512_setzero_pd(), c11 = _mm512_setzero_pd();
            __m512d c20 = _mm512_setzero_pd(), c21 = _mm512_setzero_pd();
            __m512d c30 = _mm512_setzero_pd(), c31 = _mm512_setzero_pd();
            __m512d c40 = _mm512_setzero_pd(), c41 = _mm512_setzero_pd();
            __m512d c50 = _mm512_setzero_pd(), c51 = _mm512_setzero_pd();
            __m512d c60 = _mm512_setzero_pd(), c61 = _mm512_setzero_pd();
            __m512d c70 = _mm512_setzero_pd(), c71 = _mm512_setzero_pd();

            for (size_t p = 0; p < k; ++p) {
                __m512d b0 = _mm512_loadu_pd(b + p * n + j);
                __m512d b1 = _mm512_loadu_pd(b + p * n + j + 8);
                __m512d a0 = _mm512_set1_pd(a[(i + 0) * k + p]);
                __m512d a1 = _mm512_set1_pd(a[(i + 1) * k + p]);
                __m512d a2 = _mm512_set1_pd(a[(i + 2) * k + p]);
                __m512d a3 = _mm512_set1_pd(a[(i + 3) * k + p]);
                __m512d a4 = _mm512_set1_pd(a[(i + 4) * k + p]);
                __m512d a5 = _mm512_set1_pd(a[(i + 5) * k + p]);
                __m512d a6 = _mm512_set1_pd(a[(i + 6) * k + p]);
                __m512d a7 = _mm512_set1_pd(a[(i + 7) * k + p]);

                c00 = _mm512_fmadd_pd(a0, b0, c00);
                c01 = _mm512_fmadd_pd(a0, b1, c01);
                c10 = _mm512_fmadd_pd(a1, b0, c10);
                c11 = _mm512_fmadd_pd(a1, b1, c11);
                c20 = _mm512_fmadd_pd(a2, b0, c20);
                c21 = _mm512_fmadd_pd(a2, b1, c21);
                c30 = _mm512_fmadd_pd(a3, b0, c30);
                c31 = _mm512_fmadd_pd(a3, b1, c31);
                c40 = _mm512_fmadd_pd(a4, b0, c40);
                c41 = _mm512_fmadd_pd(a4, b1, c41);
                c50 = _mm512_fmadd_pd(a5, b0, c50);
                c51 = _mm512_fmadd_pd(a5, b1, c51);
                c60 = _mm512_fmadd_pd(a6, b0, c60);
                c61 = _mm512_fmadd_pd(a6, b1, c61);
                c70 = _mm512_fmadd_pd(a7, b0, c70);
                c71 = _mm512_fmadd_pd(a7, b1, c71);
            }

#define STORE_ROW(r) \
            _mm512_storeu_pd(c + (i + r) * n + j, c##r##0); \
            _mm512_storeu_pd(c + (i + r) * n + j + 8, c##r##1)
            STORE_ROW(0);
            STORE_ROW(1);
            STORE_ROW(2);
            STORE_ROW(3);
            STORE_ROW(4);
            STORE_ROW(5);
            STORE_ROW(6);
            STORE_ROW(7);
#undef STORE_ROW
        }
        for (; j < n; ++j) {
            for (size_t r = 0; r < 8; ++r) {
                double total = 0.0;
                for (size_t p = 0; p < k; ++p) {
                    total += a[(i + r) * k + p] * b[p * n + j];
                }
                c[(i + r) * n + j] = total;
            }
        }
        j = 0;
    }

    if (i < m) {
        gemm_scalar(m - i, k, n, a + i * k, b, c + i * n);
    }
}

__attribute__((target("avx2,fma")))
static void gemm_avx2(
    size_t m, size_t k, size_t n,
    const double *restrict a,
    const double *restrict b,
    double *restrict c
) {
    size_t i = 0;
    size_t j = 0;

    for (; i + 6 <= m; i += 6) {
        for (j = 0; j + 8 <= n; j += 8) {
            __m256d c00 = _mm256_setzero_pd(), c01 = _mm256_setzero_pd();
            __m256d c10 = _mm256_setzero_pd(), c11 = _mm256_setzero_pd();
            __m256d c20 = _mm256_setzero_pd(), c21 = _mm256_setzero_pd();
            __m256d c30 = _mm256_setzero_pd(), c31 = _mm256_setzero_pd();
            __m256d c40 = _mm256_setzero_pd(), c41 = _mm256_setzero_pd();
            __m256d c50 = _mm256_setzero_pd(), c51 = _mm256_setzero_pd();

            for (size_t p = 0; p < k; ++p) {
                __m256d b0 = _mm256_loadu_pd(b + p * n + j);
                __m256d b1 = _mm256_loadu_pd(b + p * n + j + 4);
                __m256d a0 = _mm256_broadcast_sd(a + (i + 0) * k + p);
                __m256d a1 = _mm256_broadcast_sd(a + (i + 1) * k + p);
                __m256d a2 = _mm256_broadcast_sd(a + (i + 2) * k + p);
                __m256d a3 = _mm256_broadcast_sd(a + (i + 3) * k + p);
                __m256d a4 = _mm256_broadcast_sd(a + (i + 4) * k + p);
                __m256d a5 = _mm256_broadcast_sd(a + (i + 5) * k + p);

                c00 = _mm256_fmadd_pd(a0, b0, c00);
                c01 = _mm256_fmadd_pd(a0, b1, c01);
                c10 = _mm256_fmadd_pd(a1, b0, c10);
                c11 = _mm256_fmadd_pd(a1, b1, c11);
                c20 = _mm256_fmadd_pd(a2, b0, c20);
                c21 = _mm256_fmadd_pd(a2, b1, c21);
                c30 = _mm256_fmadd_pd(a3, b0, c30);
                c31 = _mm256_fmadd_pd(a3, b1, c31);
                c40 = _mm256_fmadd_pd(a4, b0, c40);
                c41 = _mm256_fmadd_pd(a4, b1, c41);
                c50 = _mm256_fmadd_pd(a5, b0, c50);
                c51 = _mm256_fmadd_pd(a5, b1, c51);
            }

#define STORE_ROW(r) \
            _mm256_storeu_pd(c + (i + r) * n + j, c##r##0); \
            _mm256_storeu_pd(c + (i + r) * n + j + 4, c##r##1)
            STORE_ROW(0);
            STORE_ROW(1);
            STORE_ROW(2);
            STORE_ROW(3);
            STORE_ROW(4);
            STORE_ROW(5);
#undef STORE_ROW
        }
        for (; j < n; ++j) {
            for (size_t r = 0; r < 6; ++r) {
                double total = 0.0;
                for (size_t p = 0; p < k; ++p) {
                    total += a[(i + r) * k + p] * b[p * n + j];
                }
                c[(i + r) * n + j] = total;
            }
        }
        j = 0;
    }

    if (i < m) {
        gemm_scalar(m - i, k, n, a + i * k, b, c + i * n);
    }
}
#endif

void uray_matmul(
    size_t m, size_t k, size_t n,
    const double *a, const double *b, double *c
) {
    if (m == 0 || k == 0 || n == 0 || a == NULL || b == NULL || c == NULL) {
        return;
    }
#if URAY_X86_GNU
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("fma")) {
        gemm_avx512(m, k, n, a, b, c);
        return;
    }
    if (__builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma")) {
        gemm_avx2(m, k, n, a, b, c);
        return;
    }
#endif
    gemm_scalar(m, k, n, a, b, c);
}

static size_t sum_weights(const int *dims, size_t layers) {
    size_t total = 0;
    for (size_t i = 0; i < layers; ++i) {
        total += (size_t)dims[i] * (size_t)dims[i + 1];
    }
    return total;
}

static size_t sum_biases(const int *dims, size_t layers) {
    size_t total = 0;
    for (size_t i = 0; i < layers; ++i) {
        total += (size_t)dims[i + 1];
    }
    return total;
}

static double sigmoid(double value) {
    if (value >= 0.0) {
        return 1.0 / (1.0 + exp(-value));
    }
    double e = exp(value);
    return e / (1.0 + e);
}

static double clamp_probability(double value) {
    if (value < 1e-12) return 1e-12;
    if (value > 1.0 - 1e-12) return 1.0 - 1e-12;
    return value;
}

static void fill_batch(
    size_t batch_count, size_t input_dim, size_t output_dim,
    const size_t *indices, size_t offset,
    const double *x, const double *y,
    const unsigned char *mnist_images,
    const unsigned char *mnist_labels,
    int mnist_mode, double *batch_x, double *batch_y
) {
    for (size_t row = 0; row < batch_count; ++row) {
        size_t index = indices[offset + row];
        double *xrow = batch_x + row * input_dim;
        double *yrow = batch_y ? batch_y + row * output_dim : NULL;
        if (mnist_mode) {
            size_t image_start = index * input_dim;
            for (size_t col = 0; col < input_dim; ++col) {
                xrow[col] = (double)mnist_images[image_start + col] / 255.0;
            }
            if (yrow) {
                memset(yrow, 0, output_dim * sizeof(double));
                if (mnist_labels && (size_t)mnist_labels[index] < output_dim) {
                    yrow[mnist_labels[index]] = 1.0;
                }
            }
        } else {
            memcpy(xrow, x + index * input_dim, input_dim * sizeof(double));
            if (yrow) {
                memcpy(yrow, y + index * output_dim, output_dim * sizeof(double));
            }
        }
    }
}

static void forward_batch(
    size_t batch_count, size_t layers, const int *dims,
    const double *weights, const double *biases, double *weight_transposed,
    double **activations, double **pre_activations,
    int activation, int output_activation
) {
    size_t weight_offset = 0;
    size_t bias_offset = 0;
    for (size_t layer = 0; layer < layers; ++layer) {
        size_t input_dim = (size_t)dims[layer];
        size_t output_dim = (size_t)dims[layer + 1];
        const double *w = weights + weight_offset;
        double *wt = weight_transposed + weight_offset;

        for (size_t out = 0; out < output_dim; ++out) {
            for (size_t in = 0; in < input_dim; ++in) {
                wt[in * output_dim + out] = w[out * input_dim + in];
            }
        }

        uray_matmul(
            batch_count, input_dim, output_dim,
            activations[layer], wt, pre_activations[layer]
        );

        int is_last = layer + 1 == layers;
        for (size_t row = 0; row < batch_count; ++row) {
            double *z = pre_activations[layer] + row * output_dim;
            double *a = activations[layer + 1] + row * output_dim;
            const double *b = biases + bias_offset;
            double max_value = -INFINITY;

            for (size_t out = 0; out < output_dim; ++out) {
                z[out] += b[out];
                if (is_last && output_activation == 2 && z[out] > max_value) {
                    max_value = z[out];
                }
            }

            if (is_last && output_activation == 2) {
                double total = 0.0;
                for (size_t out = 0; out < output_dim; ++out) {
                    a[out] = exp(z[out] - max_value);
                    total += a[out];
                }
                for (size_t out = 0; out < output_dim; ++out) {
                    a[out] /= total;
                }
            } else {
                for (size_t out = 0; out < output_dim; ++out) {
                    if (is_last) {
                        if (output_activation == 1) a[out] = sigmoid(z[out]);
                        else a[out] = z[out];
                    } else if (activation == 0) {
                        a[out] = z[out] > 0.0 ? z[out] : 0.0;
                    } else if (activation == 1) {
                        a[out] = z[out] > 0.0 ? z[out] : 0.1 * z[out];
                    } else {
                        a[out] = sigmoid(z[out]);
                    }
                }
            }
        }
        weight_offset += input_dim * output_dim;
        bias_offset += output_dim;
    }
}

int uray_mlp_train_epoch(
    size_t sample_count, const double *x, const double *y,
    const unsigned char *mnist_images, const unsigned char *mnist_labels,
    int mnist_mode, const int *dims, size_t layers,
    double *weights, double *biases,
    double *state_mw, double *state_vw,
    double *state_mb, double *state_vb, int *step,
    int optimizer, int activation, int output_activation, int loss,
    double learning_rate, const size_t *indices, size_t batch_size,
    double *mean_loss_out
) {
    if (!dims || !weights || !biases || !indices || !mean_loss_out ||
        sample_count == 0 || layers == 0 || batch_size == 0 ||
        (mnist_mode && (!mnist_images || !mnist_labels)) ||
        (!mnist_mode && (!x || !y))) {
        return 1;
    }

    size_t input_dim = (size_t)dims[0];
    size_t output_dim = (size_t)dims[layers];
    size_t total_weights = sum_weights(dims, layers);
    size_t total_biases = sum_biases(dims, layers);
    size_t max_width = input_dim;
    for (size_t i = 1; i <= layers; ++i) {
        if ((size_t)dims[i] > max_width) max_width = (size_t)dims[i];
    }

    double **activations = (double **)calloc(layers + 1, sizeof(double *));
    double **pre_activations = (double **)calloc(layers, sizeof(double *));
    double **deltas = (double **)calloc(layers, sizeof(double *));
    double *weight_transposed = (double *)malloc(total_weights * sizeof(double));
    double *grad_weights = (double *)malloc(total_weights * sizeof(double));
    double *grad_biases = (double *)malloc(total_biases * sizeof(double));
    double *targets = (double *)malloc(batch_size * output_dim * sizeof(double));
    double *delta_transposed =
        (double *)malloc(batch_size * max_width * sizeof(double));
    if (!activations || !pre_activations || !deltas ||
        !weight_transposed || !grad_weights || !grad_biases ||
        !targets || !delta_transposed) {
        free(activations); free(pre_activations); free(deltas);
        free(weight_transposed); free(grad_weights); free(grad_biases);
        free(targets); free(delta_transposed);
        return 2;
    }

    int allocation_failed = 0;
    for (size_t layer = 0; layer <= layers; ++layer) {
        activations[layer] = (double *)malloc(
            batch_size * (size_t)dims[layer] * sizeof(double)
        );
        if (!activations[layer]) allocation_failed = 1;
        if (layer < layers) {
            pre_activations[layer] = (double *)malloc(
                batch_size * (size_t)dims[layer + 1] * sizeof(double)
            );
            deltas[layer] = (double *)malloc(
                batch_size * (size_t)dims[layer + 1] * sizeof(double)
            );
            if (!pre_activations[layer] || !deltas[layer]) {
                allocation_failed = 1;
            }
        }
    }
    if (allocation_failed) {
        for (size_t layer = 0; layer <= layers; ++layer) free(activations[layer]);
        for (size_t layer = 0; layer < layers; ++layer) {
            free(pre_activations[layer]); free(deltas[layer]);
        }
        free(activations); free(pre_activations); free(deltas);
        free(weight_transposed); free(grad_weights); free(grad_biases);
        free(targets); free(delta_transposed);
        return 2;
    }

    double total_loss = 0.0;
    size_t bias_offsets[128];
    size_t weight_offsets[128];
    if (layers > 128) {
        for (size_t layer = 0; layer <= layers; ++layer) free(activations[layer]);
        for (size_t layer = 0; layer < layers; ++layer) {
            free(pre_activations[layer]); free(deltas[layer]);
        }
        free(activations); free(pre_activations); free(deltas);
        free(weight_transposed); free(grad_weights); free(grad_biases);
        free(targets); free(delta_transposed);
        return 3;
    }
    size_t weight_offset = 0, bias_offset = 0;
    for (size_t layer = 0; layer < layers; ++layer) {
        weight_offsets[layer] = weight_offset;
        bias_offsets[layer] = bias_offset;
        weight_offset += (size_t)dims[layer] * (size_t)dims[layer + 1];
        bias_offset += (size_t)dims[layer + 1];
    }

    for (size_t start = 0; start < sample_count; start += batch_size) {
        size_t current_batch = sample_count - start;
        if (current_batch > batch_size) current_batch = batch_size;
        fill_batch(
            current_batch, input_dim, output_dim, indices, start,
            x, y, mnist_images, mnist_labels, mnist_mode,
            activations[0], targets
        );
        forward_batch(
            current_batch, layers, dims, weights, biases, weight_transposed,
            activations, pre_activations, activation, output_activation
        );

        double *prediction = activations[layers];
        for (size_t row = 0; row < current_batch; ++row) {
            const double *target = targets + row * output_dim;
            const double *pred = prediction + row * output_dim;
            if (loss == 0) {
                for (size_t out = 0; out < output_dim; ++out) {
                    double d = pred[out] - target[out];
                    total_loss += d * d / (double)output_dim;
                }
            } else if (loss == 1) {
                for (size_t out = 0; out < output_dim; ++out) {
                    if (target[out] != 0.0) {
                        double p = pred[out] < 1e-12 ? 1e-12 : pred[out];
                        total_loss -= target[out] * log(p) / (double)output_dim;
                    }
                }
            } else {
                for (size_t out = 0; out < output_dim; ++out) {
                    double p = clamp_probability(pred[out]);
                    total_loss -= (
                        target[out] * log(p) +
                        (1.0 - target[out]) * log(1.0 - p)
                    ) / (double)output_dim;
                }
            }
        }

        size_t last = layers - 1;
        size_t last_width = (size_t)dims[layers];
        for (size_t row = 0; row < current_batch; ++row) {
            double *delta = deltas[last] + row * last_width;
            const double *pred = prediction + row * last_width;
            const double *target = targets + row * last_width;
            if (loss == 1 || loss == 2) {
                double scale = 1.0 / ((double)current_batch * (double)last_width);
                for (size_t out = 0; out < last_width; ++out) {
                    delta[out] = (pred[out] - target[out]) * scale;
                }
            } else {
                double scale = 2.0 / ((double)current_batch * (double)last_width);
                for (size_t out = 0; out < last_width; ++out) {
                    delta[out] = (pred[out] - target[out]) * scale;
                }
                if (output_activation == 1) {
                    for (size_t out = 0; out < last_width; ++out) {
                        delta[out] *= pred[out] * (1.0 - pred[out]);
                    }
                } else if (output_activation == 2) {
                    double projection = 0.0;
                    for (size_t out = 0; out < last_width; ++out) {
                        projection += delta[out] * pred[out];
                    }
                    for (size_t out = 0; out < last_width; ++out) {
                        delta[out] = pred[out] * (delta[out] - projection);
                    }
                }
            }
        }

        for (size_t rev = layers; rev-- > 0;) {
            size_t in_dim = (size_t)dims[rev];
            size_t out_dim = (size_t)dims[rev + 1];
            double *delta = deltas[rev];
            double *grad_w = grad_weights + weight_offsets[rev];
            double *grad_b = grad_biases + bias_offsets[rev];

            for (size_t out = 0; out < out_dim; ++out) {
                for (size_t row = 0; row < current_batch; ++row) {
                    delta_transposed[out * current_batch + row] =
                        delta[row * out_dim + out];
                }
                double sum = 0.0;
                for (size_t row = 0; row < current_batch; ++row) {
                    sum += delta[row * out_dim + out];
                }
                grad_b[out] = sum;
            }
            uray_matmul(
                out_dim, current_batch, in_dim,
                delta_transposed, activations[rev], grad_w
            );

            if (rev > 0) {
                double *previous_delta = deltas[rev - 1];
                const double *w = weights + weight_offsets[rev];
                uray_matmul(current_batch, out_dim, in_dim, delta, w, previous_delta);
                const double *previous_z = pre_activations[rev - 1];
                const double *previous_a = activations[rev];
                size_t previous_width = in_dim;
                for (size_t row = 0; row < current_batch; ++row) {
                    for (size_t in = 0; in < previous_width; ++in) {
                        double deriv;
                        if (activation == 0) {
                            deriv = previous_z[row * previous_width + in] > 0.0
                                ? 1.0 : 0.0;
                        } else if (activation == 1) {
                            deriv = previous_z[row * previous_width + in] > 0.0
                                ? 1.0 : 0.1;
                        } else {
                            double value = previous_a[row * previous_width + in];
                            deriv = value * (1.0 - value);
                        }
                        previous_delta[row * previous_width + in] *= deriv;
                    }
                }
            }
        }

        if (optimizer == 2 && step != NULL) {
            ++(*step);
        }
        double beta1_correction = 1.0;
        double beta2_correction = 1.0;
        if (optimizer == 2 && step != NULL) {
            beta1_correction = 1.0 - pow(0.9, (double)*step);
            beta2_correction = 1.0 - pow(0.999, (double)*step);
        }

        for (size_t i = 0; i < total_weights; ++i) {
            double grad = grad_weights[i];
            if (optimizer == 2) {
                state_mw[i] = 0.9 * state_mw[i] + 0.1 * grad;
                state_vw[i] = 0.999 * state_vw[i] + 0.001 * grad * grad;
                double mhat = state_mw[i] / beta1_correction;
                double vhat = state_vw[i] / beta2_correction;
                weights[i] -= learning_rate * mhat / (sqrt(vhat) + 1e-8);
            } else if (optimizer == 1) {
                state_mw[i] = 0.9 * state_mw[i] - learning_rate * grad;
                weights[i] += state_mw[i];
            } else {
                weights[i] -= learning_rate * grad;
            }
        }
        for (size_t i = 0; i < total_biases; ++i) {
            double grad = grad_biases[i];
            if (optimizer == 2) {
                state_mb[i] = 0.9 * state_mb[i] + 0.1 * grad;
                state_vb[i] = 0.999 * state_vb[i] + 0.001 * grad * grad;
                double mhat = state_mb[i] / beta1_correction;
                double vhat = state_vb[i] / beta2_correction;
                biases[i] -= learning_rate * mhat / (sqrt(vhat) + 1e-8);
            } else if (optimizer == 1) {
                state_mb[i] = 0.9 * state_mb[i] - learning_rate * grad;
                biases[i] += state_mb[i];
            } else {
                biases[i] -= learning_rate * grad;
            }
        }
    }

    *mean_loss_out = total_loss / (double)sample_count;
    for (size_t layer = 0; layer <= layers; ++layer) free(activations[layer]);
    for (size_t layer = 0; layer < layers; ++layer) {
        free(pre_activations[layer]);
        free(deltas[layer]);
    }
    free(activations); free(pre_activations); free(deltas);
    free(weight_transposed); free(grad_weights); free(grad_biases);
    free(targets); free(delta_transposed);
    return 0;
}

int uray_mlp_evaluate(
    size_t sample_count, const double *x, const double *y,
    const unsigned char *mnist_images, const unsigned char *mnist_labels,
    int mnist_mode, const int *dims, size_t layers,
    const double *weights, const double *biases,
    int activation, int output_activation, int loss,
    size_t batch_size, double *predictions, double *metrics
) {
    if (!dims || !weights || !biases || sample_count == 0 ||
        layers == 0 || batch_size == 0 ||
        (mnist_mode && (!mnist_images || (metrics && !mnist_labels))) ||
        (!mnist_mode && (!x || (metrics && !y)))) {
        return 1;
    }

    size_t input_dim = (size_t)dims[0];
    size_t output_dim = (size_t)dims[layers];
    size_t total_weights = sum_weights(dims, layers);
    size_t max_width = input_dim;
    for (size_t i = 1; i <= layers; ++i) {
        if ((size_t)dims[i] > max_width) max_width = (size_t)dims[i];
    }
    double **activations = (double **)calloc(layers + 1, sizeof(double *));
    double **pre_activations = (double **)calloc(layers, sizeof(double *));
    double *weight_transposed = (double *)malloc(total_weights * sizeof(double));
    double *targets = metrics
        ? (double *)malloc(batch_size * output_dim * sizeof(double))
        : NULL;
    if (!activations || !pre_activations || !weight_transposed ||
        (metrics && !targets)) {
        free(activations); free(pre_activations); free(weight_transposed);
        free(targets);
        return 2;
    }

    int allocation_failed = 0;
    for (size_t layer = 0; layer <= layers; ++layer) {
        activations[layer] = (double *)malloc(
            batch_size * (size_t)dims[layer] * sizeof(double)
        );
        if (!activations[layer]) allocation_failed = 1;
        if (layer < layers) {
            pre_activations[layer] = (double *)malloc(
                batch_size * (size_t)dims[layer + 1] * sizeof(double)
            );
            if (!pre_activations[layer]) allocation_failed = 1;
        }
    }
    if (allocation_failed) {
        for (size_t layer = 0; layer <= layers; ++layer) free(activations[layer]);
        for (size_t layer = 0; layer < layers; ++layer) free(pre_activations[layer]);
        free(activations); free(pre_activations); free(weight_transposed);
        free(targets);
        return 2;
    }

    double *batch_x = activations[0];
    double *batch_y = targets;
    double total_loss = 0.0;
    size_t correct = 0;
    if (layers > 128) {
        for (size_t layer = 0; layer <= layers; ++layer) free(activations[layer]);
        for (size_t layer = 0; layer < layers; ++layer) free(pre_activations[layer]);
        free(activations); free(pre_activations); free(weight_transposed);
        free(targets);
        return 3;
    }
    size_t *indices = (size_t *)malloc(batch_size * sizeof(size_t));
    if (!indices) {
        for (size_t layer = 0; layer <= layers; ++layer) free(activations[layer]);
        for (size_t layer = 0; layer < layers; ++layer) free(pre_activations[layer]);
        free(activations); free(pre_activations); free(weight_transposed);
        free(targets);
        return 2;
    }

    for (size_t start = 0; start < sample_count; start += batch_size) {
        size_t current_batch = sample_count - start;
        if (current_batch > batch_size) current_batch = batch_size;
        for (size_t row = 0; row < current_batch; ++row) {
            indices[row] = start + row;
        }
        fill_batch(
            current_batch, input_dim, output_dim, indices, 0,
            x, y, mnist_images, mnist_labels, mnist_mode,
            batch_x, batch_y
        );
        forward_batch(
            current_batch, layers, dims, weights, biases, weight_transposed,
            activations, pre_activations, activation, output_activation
        );
        double *batch_prediction = activations[layers];
        if (predictions) {
            memcpy(
                predictions + start * output_dim,
                batch_prediction,
                current_batch * output_dim * sizeof(double)
            );
        }
        if (metrics) {
            for (size_t row = 0; row < current_batch; ++row) {
                const double *target = batch_y + row * output_dim;
                const double *pred = batch_prediction + row * output_dim;
                if (loss == 0) {
                    for (size_t out = 0; out < output_dim; ++out) {
                        double d = pred[out] - target[out];
                        total_loss += d * d / (double)output_dim;
                    }
                } else if (loss == 1) {
                    for (size_t out = 0; out < output_dim; ++out) {
                        if (target[out] != 0.0) {
                            double p = pred[out] < 1e-12 ? 1e-12 : pred[out];
                            total_loss -= target[out] * log(p) / (double)output_dim;
                        }
                    }
                } else {
                    for (size_t out = 0; out < output_dim; ++out) {
                        double p = clamp_probability(pred[out]);
                        total_loss -= (
                            target[out] * log(p) +
                            (1.0 - target[out]) * log(1.0 - p)
                        ) / (double)output_dim;
                    }
                }

                size_t predicted = 0, actual = 0;
                if (output_dim == 1) {
                    predicted = pred[0] >= 0.5 ? 1 : 0;
                    actual = target[0] >= 0.5 ? 1 : 0;
                } else {
                    for (size_t out = 1; out < output_dim; ++out) {
                        if (pred[out] > pred[predicted]) predicted = out;
                        if (target[out] > target[actual]) actual = out;
                    }
                }
                if (predicted == actual) ++correct;
            }
        }
    }

    if (metrics) {
        metrics[0] = total_loss / (double)sample_count;
        metrics[1] = (double)correct / (double)sample_count;
    }
    free(indices);
    for (size_t layer = 0; layer <= layers; ++layer) free(activations[layer]);
    for (size_t layer = 0; layer < layers; ++layer) free(pre_activations[layer]);
    free(activations); free(pre_activations); free(weight_transposed);
    free(targets);
    return 0;
}
