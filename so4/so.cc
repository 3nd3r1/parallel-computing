#include <algorithm>
#include <cstring>
#include <omp.h>

typedef unsigned long long data_t;

void merge(data_t *data, data_t *tmp, int l, int m, int r) {
    int nl = m - l + 1;
    int nr = r - m;

    std::memcpy(tmp + l, data + l, nl * sizeof(data_t));
    std::memcpy(tmp + m + 1, data + m + 1, nr * sizeof(data_t));

    int i = l, j = m + 1;
    int k = l;
    while (i <= m && j <= r) {
        if (tmp[i] <= tmp[j]) {
            data[k++] = tmp[i++];
        } else {
            data[k++] = tmp[j++];
        }
    }

    while (i <= m)
        data[k++] = tmp[i++];

    while (j <= r)
        data[k++] = tmp[j++];
}

void mergeSort(data_t *data, data_t *tmp, int l, int r) {
    if (r - l < 1000000) {
        std::sort(data + l, data + r + 1);
        return;
    }

    int m = l + (r - l) / 2;
#pragma omp task
    mergeSort(data, tmp, l, m);
#pragma omp task
    mergeSort(data, tmp, m + 1, r);
#pragma omp taskwait
    merge(data, tmp, l, m, r);
}

void psort(int n, data_t *data) {
    data_t *tmp = new data_t[n];
#pragma omp parallel
#pragma omp single
    mergeSort(data, tmp, 0, n - 1);
    delete[] tmp;
}
