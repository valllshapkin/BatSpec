#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace py = pybind11;

// Структура Dual-Heap (Mediator) Ричарда Хартера
struct Mediator {
    std::vector<int> pos;
    std::vector<int> heap_base;
    int* heap;
    std::vector<float> data;
    int N, idx, minCt, maxCt;

    Mediator(int nItems) : pos(nItems), heap_base(nItems), data(nItems, 0.0f) {
        N = nItems;
        int rank = N / 2;
        heap = heap_base.data() + rank;
        reset();
    }

    void reset() {
        int rank = N / 2;
        idx = 0;
        minCt = N - rank - 1;
        maxCt = rank;
        std::fill(data.begin(), data.end(), 0.0f);
        for (int i = 0; i < N; i++) {
            pos[i] = i - rank;
            heap[pos[i]] = i;
        }
    }

    inline bool mmless(int i, int j) { return data[heap[i]] < data[heap[j]]; }

    inline bool mmexchange(int i, int j) {
        int t = heap[i];
        heap[i] = heap[j];
        heap[j] = t;
        pos[heap[i]] = i;
        pos[heap[j]] = j;
        return true;
    }

    inline bool mmCmpExch(int i, int j) {
        return (mmless(i, j) && mmexchange(i, j));
    }

    void minSortDown(int i) {
        for (i *= 2; i <= minCt; i *= 2) {
            if (i < minCt && mmless(i + 1, i)) ++i;
            if (!mmCmpExch(i, i / 2)) break;
        }
    }

    void maxSortDown(int i) {
        for (i *= 2; i >= -maxCt; i *= 2) {
            if (i > -maxCt && mmless(i, i - 1)) --i;
            if (!mmCmpExch(i / 2, i)) break;
        }
    }

    inline bool minSortUp(int i) {
        while (i > 0 && mmCmpExch(i, i / 2)) i /= 2;
        return (i == 0);
    }

    inline bool maxSortUp(int i) {
        while (i < 0 && mmCmpExch(i / 2, i)) i /= 2;
        return (i == 0);
    }

    void insert(float v) {
        int p = pos[idx];
        float old = data[idx];
        data[idx] = v;
        idx++;
        if (idx == N) idx = 0;

        if (p > 0) {
            if (v > old) { minSortDown(p); return; }
            if (minSortUp(p) && mmCmpExch(0, -1)) maxSortDown(-1);
        } else if (p < 0) {
            if (v < old) { maxSortDown(p); return; }
            if (maxSortUp(p) && mmCmpExch(1, 0)) minSortDown(1);
        } else {
            if (maxSortUp(-1)) maxSortDown(-1);
            if (minSortUp(1)) minSortDown(1);
        }
    }

    float get_median() {
        return data[heap[0]];
    }
};

py::array_t<float> fast_median_2d(py::array_t<float, py::array::c_style | py::array::forcecast> input_padded, int h_orig, int w_orig, int kh, int kw) {
    // 1. Извлекаем указатели ДО освобождения GIL (вызовы PyBind11 требуют GIL)
    py::buffer_info buf_info = input_padded.request();
    const float* in_ptr = static_cast<float*>(buf_info.ptr);
    int stride = buf_info.shape[1]; 

    py::array_t<float> output({h_orig, w_orig});
    float* out_ptr = static_cast<float*>(output.request().ptr);

    // 2. ОСВОБОЖДАЕМ GIL! Теперь Python (UI поток) может работать параллельно с C++
    py::gil_scoped_release release;

    // 3. Резервируем 1 ядро для UI, чтобы OpenMP не задушил операционную систему
    int num_threads = omp_get_max_threads();
    if (num_threads > 1) {
        omp_set_num_threads(num_threads - 1); 
    }

    #pragma omp parallel
    {
        Mediator med(kh * kw);

        #pragma omp for schedule(dynamic)
        for (int y = 0; y < h_orig; ++y) {
            med.reset();

            for (int dx = 0; dx < kw; ++dx) {
                for (int dy = 0; dy < kh; ++dy) {
                    med.insert(in_ptr[(y + dy) * stride + dx]);
                }
            }
            out_ptr[y * w_orig + 0] = med.get_median();

            for (int x = 1; x < w_orig; ++x) {
                int dx = x + kw - 1;
                for (int dy = 0; dy < kh; ++dy) {
                    med.insert(in_ptr[(y + dy) * stride + dx]);
                }
                out_ptr[y * w_orig + x] = med.get_median();
            }
        }
    }
    // GIL автоматически захватывается обратно при выходе из функции
    return output;
}

PYBIND11_MODULE(FastZScore, m) {
    m.doc() = "BatSpec Spectral Statistics: Fast 2D Median Module";
    m.def("fast_median_2d", &fast_median_2d, 
          "Супероптимизированный расчет 2D медианы на C++ (Dual-Heap + OpenMP)",
          py::arg("input_padded"), py::arg("h_orig"), py::arg("w_orig"), py::arg("kh"), py::arg("kw"));
}
