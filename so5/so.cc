#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

int partition(data_t *data, int l, int r) {
    int mid = l + (r - l) / 2;
    if (data[mid] < data[l])
        std::swap(data[mid], data[l]);
    if (data[r] < data[l])
        std::swap(data[r], data[l]);
    if (data[mid] < data[r])
        std::swap(data[mid], data[r]);

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

void quickSort(data_t *data, int l, int r, int depth = 0) {
    if (r - l < 10000 || depth > 40) {
        std::sort(data + l, data + r + 1);
        return;
    }

    int pi = partition(data, l, r);
    if (depth < 8) {
#pragma omp task
        quickSort(data, l, pi - 1, depth + 1);
#pragma omp task
        quickSort(data, pi + 1, r, depth + 1);
#pragma omp taskwait
    } else {
        quickSort(data, l, pi - 1, depth + 1);
        quickSort(data, pi + 1, r, depth + 1);
    }
}

void psort(int n, data_t *data) {
#pragma omp parallel
#pragma omp single
    quickSort(data, 0, n - 1);
}
