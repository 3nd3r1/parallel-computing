#include <algorithm>
#include <omp.h>
#include <vector>

using std::vector;

typedef unsigned long long data_t;

void merge(data_t *data, int l, int m, int r) {
    int nl = m - l + 1;
    int nr = r - m;

    vector<data_t> L(nl), R(nr);

    for (int i = 0; i < nl; i++)
        L[i] = data[l + i];
    for (int j = 0; j < nr; j++)
        R[j] = data[m + 1 + j];

    int i = 0, j = 0;
    int k = l;
    while (i < nl && j < nr) {
        if (L[i] <= R[j]) {
            data[k] = L[i];
            i++;
        } else {
            data[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < nl) {
        data[k] = L[i];
        i++;
        k++;
    }
    while (j < nr) {
        data[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(data_t *data, int l, int r) {
    if (r - l < 100000) {
        std::sort(data + l, data + r + 1);
        return;
    }

    int m = l + (r - l) / 2;
#pragma omp task
    mergeSort(data, l, m);
#pragma omp task
    mergeSort(data, m + 1, r);
#pragma omp taskwait
    merge(data, l, m, r);
}

void psort(int n, data_t *data) {
#pragma omp parallel
#pragma omp single
    {
        mergeSort(data, 0, n - 1);
    }
}
