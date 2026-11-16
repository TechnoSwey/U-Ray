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
    if (n >= 16 && __builtin_cpu_supports("avx512f") &&
        __builtin_cpu_supports("fma")) {
        gemm_avx512(m, k, n, a, b, c);
        return;
    }
    if (n >= 8 && __builtin_cpu_supports("avx2") &&
        __builtin_cpu_supports("fma")) {
        gemm_avx2(m, k, n, a, b, c);
        return;
    }
#endif
    gemm_scalar(m, k, n, a, b, c);
}

int uray_all_finite(size_t count, const double *values) {
    if (count != 0 && values == NULL) {
        return 0;
    }
    for (size_t index = 0; index < count; ++index) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
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

static int checked_product(size_t a, size_t b, size_t *result) {
    if (a != 0 && b > SIZE_MAX / a) return 0;
    *result = a * b;
    return 1;
}

static int conv2d_dimensions(
    size_t batches, size_t in_channels, size_t height, size_t width,
    size_t out_channels, size_t kernel_h, size_t kernel_w,
    size_t stride, size_t padding, size_t *out_h, size_t *out_w,
    size_t *rows, size_t *reduction
) {
    if (batches == 0 || in_channels == 0 || height == 0 || width == 0 ||
        out_channels == 0 || kernel_h == 0 || kernel_w == 0 || stride == 0 ||
        padding > (SIZE_MAX - height) / 2 ||
        padding > (SIZE_MAX - width) / 2) {
        return 0;
    }
    size_t padded_h = height + 2 * padding;
    size_t padded_w = width + 2 * padding;
    if (kernel_h > padded_h || kernel_w > padded_w) return 0;
    *out_h = (padded_h - kernel_h) / stride + 1;
    *out_w = (padded_w - kernel_w) / stride + 1;
    size_t spatial, kernel_size;
    if (!checked_product(*out_h, *out_w, &spatial) ||
        !checked_product(batches, spatial, rows) ||
        !checked_product(in_channels, kernel_h, &kernel_size) ||
        !checked_product(kernel_size, kernel_w, reduction)) {
        return 0;
    }
    size_t im2col_count, weights_col_count;
    size_t input_count, output_count, weight_count;
    return checked_product(*rows, *reduction, &im2col_count) &&
        checked_product(*reduction, out_channels, &weights_col_count) &&
        checked_product(batches, in_channels, &input_count) &&
        checked_product(input_count, height, &input_count) &&
        checked_product(input_count, width, &input_count) &&
        checked_product(batches, out_channels, &output_count) &&
        checked_product(output_count, *out_h, &output_count) &&
        checked_product(output_count, *out_w, &output_count) &&
        checked_product(out_channels, *reduction, &weight_count) &&
        im2col_count <= SIZE_MAX / sizeof(double) &&
        weights_col_count <= SIZE_MAX / sizeof(double) &&
        input_count <= SIZE_MAX / sizeof(double) &&
        output_count <= SIZE_MAX / sizeof(double) &&
        weight_count <= SIZE_MAX / sizeof(double);
}

int uray_conv2d_forward(
    size_t batches, size_t in_channels, size_t height, size_t width,
    size_t out_channels, size_t kernel_h, size_t kernel_w,
    size_t stride, size_t padding,
    const double *input, const double *weights, const double *bias,
    double *output_nchw, double *im2col, double *weights_col
) {
    size_t out_h, out_w, rows, reduction;
    if (!conv2d_dimensions(
            batches, in_channels, height, width, out_channels,
            kernel_h, kernel_w, stride, padding,
            &out_h, &out_w, &rows, &reduction) ||
        input == NULL || weights == NULL || output_nchw == NULL ||
        im2col == NULL || weights_col == NULL) {
        return 1;
    }

    for (size_t row = 0; row < rows; ++row) {
        size_t batch = row / (out_h * out_w);
        size_t position = row % (out_h * out_w);
        size_t out_y = position / out_w;
        size_t out_x = position % out_w;
        size_t column = 0;
        for (size_t channel = 0; channel < in_channels; ++channel) {
            for (size_t ky = 0; ky < kernel_h; ++ky) {
                size_t in_y = out_y * stride + ky;
                for (size_t kx = 0; kx < kernel_w; ++kx, ++column) {
                    size_t in_x = out_x * stride + kx;
                    if (in_y < padding || in_x < padding ||
                        in_y - padding >= height ||
                        in_x - padding >= width) {
                        im2col[row * reduction + column] = 0.0;
                    } else {
                        size_t input_index =
                            ((batch * in_channels + channel) * height +
                             (in_y - padding)) * width + (in_x - padding);
                        im2col[row * reduction + column] = input[input_index];
                    }
                }
            }
        }
    }

    for (size_t channel = 0; channel < in_channels; ++channel) {
        for (size_t ky = 0; ky < kernel_h; ++ky) {
            for (size_t kx = 0; kx < kernel_w; ++kx) {
                size_t column =
                    (channel * kernel_h + ky) * kernel_w + kx;
                for (size_t out = 0; out < out_channels; ++out) {
                    size_t weight_index =
                        ((out * in_channels + channel) * kernel_h + ky) *
                            kernel_w + kx;
                    weights_col[column * out_channels + out] =
                        weights[weight_index];
                }
            }
        }
    }

    size_t matrix_size;
    if (!checked_product(rows, out_channels, &matrix_size) ||
        matrix_size > SIZE_MAX / sizeof(double)) {
        return 1;
    }
    double *matrix = (double *)malloc(matrix_size * sizeof(double));
    if (matrix == NULL) return 2;
    uray_matmul(rows, reduction, out_channels, im2col, weights_col, matrix);
    for (size_t row = 0; row < rows; ++row) {
        size_t batch = row / (out_h * out_w);
        size_t position = row % (out_h * out_w);
        size_t out_y = position / out_w;
        size_t out_x = position % out_w;
        for (size_t out = 0; out < out_channels; ++out) {
            double value = matrix[row * out_channels + out];
            if (bias != NULL) value += bias[out];
            size_t output_index =
                ((batch * out_channels + out) * out_h + out_y) * out_w + out_x;
            output_nchw[output_index] = value;
        }
    }
    free(matrix);
    return 0;
}

int uray_conv2d_backward(
    size_t batches, size_t in_channels, size_t height, size_t width,
    size_t out_channels, size_t kernel_h, size_t kernel_w,
    size_t stride, size_t padding,
    const double *grad_nchw, const double *im2col, const double *weights_col,
    double *grad_input, double *grad_weights, double *grad_bias
) {
    size_t out_h, out_w, rows, reduction;
    if (!conv2d_dimensions(
            batches, in_channels, height, width, out_channels,
            kernel_h, kernel_w, stride, padding,
            &out_h, &out_w, &rows, &reduction) ||
        grad_nchw == NULL || im2col == NULL || weights_col == NULL ||
        grad_input == NULL || grad_weights == NULL || grad_bias == NULL) {
        return 1;
    }

    size_t grad_size, transposed_size, input_size, weight_size;
    if (!checked_product(rows, out_channels, &grad_size) ||
        !checked_product(out_channels, rows, &transposed_size) ||
        !checked_product(batches, in_channels, &input_size) ||
        !checked_product(input_size, height, &input_size) ||
        !checked_product(input_size, width, &input_size) ||
        !checked_product(out_channels, reduction, &weight_size) ||
        grad_size > SIZE_MAX / sizeof(double) ||
        transposed_size > SIZE_MAX / sizeof(double) ||
        input_size > SIZE_MAX / sizeof(double) ||
        weight_size > SIZE_MAX / sizeof(double) ||
        rows > SIZE_MAX / reduction ||
        rows * reduction > SIZE_MAX / sizeof(double)) {
        return 1;
    }

    double *grad_rows = (double *)malloc(grad_size * sizeof(double));
    double *grad_transposed =
        (double *)malloc(transposed_size * sizeof(double));
    double *weights_transposed =
        (double *)malloc(weight_size * sizeof(double));
    double *weight_rows = (double *)malloc(weight_size * sizeof(double));
    double *grad_columns =
        (double *)malloc(rows * reduction * sizeof(double));
    if (grad_rows == NULL || grad_transposed == NULL ||
        weights_transposed == NULL || weight_rows == NULL ||
        grad_columns == NULL) {
        free(grad_rows);
        free(grad_transposed);
        free(weights_transposed);
        free(weight_rows);
        free(grad_columns);
        return 2;
    }

    memset(grad_input, 0, input_size * sizeof(double));
    memset(grad_weights, 0, weight_size * sizeof(double));
    memset(grad_bias, 0, out_channels * sizeof(double));
    memset(grad_transposed, 0, transposed_size * sizeof(double));
    size_t spatial = out_h * out_w;
    for (size_t row = 0; row < rows; ++row) {
        size_t batch = row / spatial;
        size_t position = row % spatial;
        size_t out_y = position / out_w;
        size_t out_x = position % out_w;
        for (size_t out = 0; out < out_channels; ++out) {
            size_t nchw_index =
                ((batch * out_channels + out) * out_h + out_y) * out_w + out_x;
            double value = grad_nchw[nchw_index];
            grad_rows[row * out_channels + out] = value;
            grad_transposed[out * rows + row] = value;
            grad_bias[out] += value;
        }
    }

    for (size_t column = 0; column < reduction; ++column) {
        for (size_t out = 0; out < out_channels; ++out) {
            weights_transposed[out * reduction + column] =
                weights_col[column * out_channels + out];
        }
    }
    uray_matmul(
        out_channels, rows, reduction,
        grad_transposed, im2col, weight_rows
    );
    uray_matmul(
        rows, out_channels, reduction,
        grad_rows, weights_transposed, grad_columns
    );

    for (size_t out = 0; out < out_channels; ++out) {
        for (size_t column = 0; column < reduction; ++column) {
            size_t channel = column / (kernel_h * kernel_w);
            size_t kernel_offset = column % (kernel_h * kernel_w);
            size_t ky = kernel_offset / kernel_w;
            size_t kx = kernel_offset % kernel_w;
            size_t weight_index =
                ((out * in_channels + channel) * kernel_h + ky) *
                    kernel_w + kx;
            grad_weights[weight_index] =
                weight_rows[out * reduction + column];
        }
    }

    for (size_t row = 0; row < rows; ++row) {
        size_t batch = row / spatial;
        size_t position = row % spatial;
        size_t out_y = position / out_w;
        size_t out_x = position % out_w;
        size_t column = 0;
        for (size_t channel = 0; channel < in_channels; ++channel) {
            for (size_t ky = 0; ky < kernel_h; ++ky) {
                size_t in_y = out_y * stride + ky;
                for (size_t kx = 0; kx < kernel_w; ++kx, ++column) {
                    size_t in_x = out_x * stride + kx;
                    if (in_y < padding || in_x < padding ||
                        in_y - padding >= height ||
                        in_x - padding >= width) {
                        continue;
                    }
                    size_t input_index =
                        ((batch * in_channels + channel) * height +
                         (in_y - padding)) * width + (in_x - padding);
                    grad_input[input_index] +=
                        grad_columns[row * reduction + column];
                }
            }
        }
    }

    free(grad_rows);
    free(grad_transposed);
    free(weights_transposed);
    free(weight_rows);
    free(grad_columns);
    return 0;
}

int uray_max_pool2d_forward(
    size_t batches, size_t channels, size_t height, size_t width,
    size_t kernel, size_t stride, const double *input, double *output,
    size_t *indices
) {
    if (batches == 0 || channels == 0 || height == 0 || width == 0 ||
        kernel == 0 || stride == 0 || kernel > height || kernel > width ||
        input == NULL || output == NULL || indices == NULL) {
        return 1;
    }
    size_t out_h = (height - kernel) / stride + 1;
    size_t out_w = (width - kernel) / stride + 1;
    size_t input_count, output_count;
    if (!checked_product(batches, channels, &input_count) ||
        !checked_product(input_count, height, &input_count) ||
        !checked_product(input_count, width, &input_count) ||
        !checked_product(batches, channels, &output_count) ||
        !checked_product(output_count, out_h, &output_count) ||
        !checked_product(output_count, out_w, &output_count) ||
        input_count > SIZE_MAX / sizeof(double) ||
        output_count > SIZE_MAX / sizeof(double) ||
        output_count > SIZE_MAX / sizeof(size_t) ||
        kernel > SIZE_MAX / kernel) {
        return 1;
    }
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t channel = 0; channel < channels; ++channel) {
            for (size_t out_y = 0; out_y < out_h; ++out_y) {
                for (size_t out_x = 0; out_x < out_w; ++out_x) {
                    size_t best_index = 0;
                    size_t in_y = out_y * stride;
                    size_t in_x = out_x * stride;
                    double best = input[
                        ((batch * channels + channel) * height + in_y) *
                            width + in_x
                    ];
                    for (size_t ky = 0; ky < kernel; ++ky) {
                        for (size_t kx = 0; kx < kernel; ++kx) {
                            size_t index = ky * kernel + kx;
                            double value = input[
                                ((batch * channels + channel) * height +
                                 in_y + ky) * width + in_x + kx
                            ];
                            if (value > best) {
                                best = value;
                                best_index = index;
                            }
                        }
                    }
                    size_t output_index =
                        ((batch * channels + channel) * out_h + out_y) *
                            out_w + out_x;
                    output[output_index] = best;
                    indices[output_index] = best_index;
                }
            }
        }
    }
    return 0;
}

int uray_max_pool2d_backward(
    size_t batches, size_t channels, size_t height, size_t width,
    size_t kernel, size_t stride, const double *grad_output,
    const size_t *indices, double *grad_input
) {
    if (batches == 0 || channels == 0 || height == 0 || width == 0 ||
        kernel == 0 || stride == 0 || kernel > height || kernel > width ||
        grad_output == NULL || indices == NULL || grad_input == NULL) {
        return 1;
    }
    size_t out_h = (height - kernel) / stride + 1;
    size_t out_w = (width - kernel) / stride + 1;
    size_t input_count, output_count;
    if (!checked_product(batches, channels, &input_count) ||
        !checked_product(input_count, height, &input_count) ||
        !checked_product(input_count, width, &input_count) ||
        !checked_product(batches, channels, &output_count) ||
        !checked_product(output_count, out_h, &output_count) ||
        !checked_product(output_count, out_w, &output_count) ||
        input_count > SIZE_MAX / sizeof(double) ||
        output_count > SIZE_MAX / sizeof(double) ||
        output_count > SIZE_MAX / sizeof(size_t) ||
        kernel > SIZE_MAX / kernel) {
        return 1;
    }
    memset(grad_input, 0, input_count * sizeof(double));
    for (size_t batch = 0; batch < batches; ++batch) {
        for (size_t channel = 0; channel < channels; ++channel) {
            for (size_t out_y = 0; out_y < out_h; ++out_y) {
                for (size_t out_x = 0; out_x < out_w; ++out_x) {
                    size_t output_index =
                        ((batch * channels + channel) * out_h + out_y) *
                            out_w + out_x;
                    size_t best_index = indices[output_index];
                    if (best_index >= kernel * kernel) return 1;
                    size_t in_y = out_y * stride + best_index / kernel;
                    size_t in_x = out_x * stride + best_index % kernel;
                    size_t input_index =
                        ((batch * channels + channel) * height + in_y) *
                            width + in_x;
                    grad_input[input_index] += grad_output[output_index];
                }
            }
        }
    }
    return 0;
}

void uray_relu_forward(
    size_t count, const double *input, double *output
) {
    if (count == 0 || input == NULL || output == NULL) return;
    for (size_t index = 0; index < count; ++index) {
        output[index] = input[index] > 0.0 ? input[index] : 0.0;
    }
}

void uray_relu_backward(
    size_t count, const double *input, const double *grad_output,
    double *grad_input
) {
    if (count == 0 || input == NULL || grad_output == NULL ||
        grad_input == NULL) {
        return;
    }
    for (size_t index = 0; index < count; ++index) {
        grad_input[index] = input[index] > 0.0 ? grad_output[index] : 0.0;
    }
}
