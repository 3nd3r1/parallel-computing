#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

int partition(data_t *data, int l, int r) {
    data_t pivot = data[r];
    int i = l - 1;

    for (int j = l; j <= r; j++) {
        if (data[j] < pivot) {
            i++;
            std::swap(data[i], data[j]);
        }
    }

    std::swap(data[i + 1], data[r]);
    return i + 1;
}

void quickSort(data_t *data, int l, int r) {
    if (r - l < 1000000) {
        std::sort(data + l, data + r + 1);
        return;
    }

    int pi;
#pragma omp task
    pi = partition(data, l, r);
#pragma omp task
    quickSort(data, l, pi - 1);
#pragma omp taskwait
    quickSort(data, pi + 1, r);
}

void psort(int n, data_t *data) {
#pragma omp parallel
#pragma omp single
    quickSort(data, 0, n - 1);
}
