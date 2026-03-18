#include <algorithm>
#include <omp.h>
#include <stdlib.h>
#include <vector>

using std::vector, std::max, std::min;

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/
void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
    #pragma omp parallel for
    for (int y = 0; y < ny; y++) {
        for (int x = 0; x < nx; x++) {
            vector<float> vals;
            vals.reserve(2 * hx * 2 * hy);

            for (int i = max(0, y - hy); i < min(ny, y + hy + 1); i++) {
                for (int j = max(0, x - hx); j < min(nx, x + hx + 1); j++) {
                    vals.push_back(in[j + i * nx]);
                }
            }

            std::nth_element(vals.begin(), vals.begin() + vals.size() / 2,
                             vals.end());
            if (vals.size() % 2 == 0) {
                float a = *std::max_element(vals.begin(),
                                            vals.begin() + vals.size() / 2);
                out[x + y * nx] = (a + vals[vals.size() / 2]) / 2;
            } else {
                out[x + y * nx] = vals[vals.size() / 2];
            }
        }
    }
}
