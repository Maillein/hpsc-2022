#include <cstdio>
#include <cstdlib>
#include <cmath>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    // for(int j=0; j<N; j++) {
    //   if(i != j) {
    //     float rx = x[i] - x[j];
    //     float ry = y[i] - y[j];
    //     float r = std::sqrt(rx * rx + ry * ry);
    //     fx[i] -= rx * m[j] / (r * r * r);
    //     fy[i] -= ry * m[j] / (r * r * r);
    //   }
    // }

    __m256 xivec = _mm_load_ps(x[i]);
    __m256 yivec = _mm_load_ps(y[i]);
    __m256 xjvec = _mm_load_ps(x);
    __m256 yjvec = _mm_load_ps(y);

    __m256 mask = _mm_cmp_ps(xivec, xjvec, _CMP_EQ_OQ);

    __m256 rxvec = _mm_sub_ps(xivec, xjvec);
    __m256 ryvec = _mm_sub_ps(yivec, yjvec);

    __m256 rxvec_pow2 = _mm_mul_ps(rxvec, rxvec);
    __m256 ryvec_pow2 = _mm_mul_ps(ryvec, ryvec);
    __m256 rvec_pow2 = _mm_add_ps(rxvec_pow2, ryvec_pow2);
    __m256 rvec_rev = _mm_rsqrt_ps(rvec_pow2);

    __m256 mvec = _mm_load_ps(m);

    _m256 fxvec = _mm256_mul_ps(rxvec, mvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
