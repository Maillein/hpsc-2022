#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <immintrin.h>

__m256 vec_pow(__m256 vec, int p) {
    __m256 ret = vec;
    for (int i = 1; i < p; i++) {
        ret = _mm256_mul_ps(ret, vec);
    }
    return ret;
}

__m256 vec_reduction(__m256 avec) {
    __m256 bvec = _mm256_permute2f128_ps(avec, avec, 1);
    bvec = _mm256_add_ps(bvec, avec);
    bvec = _mm256_hadd_ps(bvec, bvec);
    bvec = _mm256_hadd_ps(bvec, bvec);
    return bvec;
}

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], index[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    index[i] = i;
  }

  __m256 jvec    = _mm256_load_ps(index);
  __m256 xjvec   = _mm256_load_ps(x);
  __m256 yjvec   = _mm256_load_ps(y);
  __m256 mvec    = _mm256_load_ps(m);

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
    
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(ivec, jvec, _CMP_EQ_OQ);

    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);

    __m256 rxvec = _mm256_sub_ps(xivec, xjvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yjvec);
    __m256 rxvec_pow2 = vec_pow(rxvec, 2);
    __m256 ryvec_pow2 = vec_pow(ryvec, 2);
    __m256 rvec_pow2 = _mm256_add_ps(rxvec_pow2, ryvec_pow2);
    __m256 rrvec = _mm256_rsqrt_ps(rvec_pow2);
    __m256 rrvec_pow3 = vec_pow(rrvec, 3);

    __m256 fxvec = _mm256_load_ps(fx);
    __m256 gxvec = _mm256_mul_ps(rxvec, mvec);
    gxvec = _mm256_mul_ps(gxvec, rrvec_pow3);
    gxvec = _mm256_blendv_ps(gxvec, _mm256_setzero_ps(), mask);
    __m256 hxvec = vec_reduction(gxvec);
    hxvec = _mm256_blendv_ps(_mm256_setzero_ps(), hxvec, mask);
    fxvec = _mm256_sub_ps(fxvec, hxvec);
    _mm256_store_ps(fx, fxvec);

    __m256 fyvec = _mm256_load_ps(fy);
    __m256 gyvec = _mm256_mul_ps(ryvec, mvec);
    gyvec = _mm256_mul_ps(gyvec, rrvec_pow3);
    gyvec = _mm256_blendv_ps(gyvec, _mm256_setzero_ps(), mask);
    __m256 hyvec = vec_reduction(gyvec);
    hyvec = _mm256_blendv_ps(_mm256_setzero_ps(), hyvec, mask);
    fyvec = _mm256_sub_ps(fyvec, hyvec);
    _mm256_store_ps(fy, fyvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
