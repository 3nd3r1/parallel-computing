#include <algorithm>
#include <omp.h>

typedef unsigned long long data_t;

int partition(data_t *data, int l, int r) {
    int mid = l + (r - l) / 2;
    if (data[mid] < data[l])
        std::swap(data[mid], data[l]);
    if (data[r] < data[l])
        std::swap(data[r], data[l]);
    if (data[r] < data[mid])
        std::swap(data[r], data[mid]);
    data_t pivot = data[mid];
    std::swap(data[mid], data[r - 1]);

    int i = l, j = r - 1;
    while (true) {
        while (data[++i] < pivot)
            ;
        while (data[--j] > pivot)
            ;
        if (i >= j)
            break;
        std::swap(data[i], data[j]);
    }
    std::swap(data[i], data[r - 1]);
    return i;
}

void quickSort(data_t *data, int l, int r, int depth = 0) {
    if (r - l < 10000 || depth > 40) {
        std::sort(data + l, data + r + 1);
        return;
    }

    int pi = partition(data, l, r);
    if (pi - l <= r - pi) {
#pragma omp task
        quickSort(data, l, pi - 1, depth + 1);
        quickSort(data, pi + 1, r, depth + 1);
    } else {
#pragma omp task
        quickSort(data, pi + 1, r, depth + 1);
        quickSort(data, l, pi - 1, depth + 1);
    }
}

void psort(int n, data_t *data) {
#pragma omp parallel
#pragma omp single
    quickSort(data, 0, n - 1);
}
