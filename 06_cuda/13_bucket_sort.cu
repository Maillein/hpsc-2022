#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void scan(int *a, int N) {
  extern __shared__ int b[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for(int j=1; j<N; j<<=1) {
    b[i] = a[i];
    __syncthreads();
    if(i>=j) a[i] += b[i-j];
    __syncthreads();
  }
}

__global__ void init_bucket(int *key, int *bucket, int n, int range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < range) bucket[idx] = 0;
  if (idx < n) atomicAdd(&bucket[key[idx]], 1);
}

__global__ void reset_key(int *key, int *bucket, int n, int range) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) key[idx] = 0;
  if (idx < range) atomicAdd(&key[bucket[idx]], 1);
}

void bucket_sort(int *key, int n, int range) {
  int *bucket;
  cudaMallocManaged(&bucket, range * sizeof(int));

  init_bucket<<<1, n>>>(key, bucket, n, range);
  scan<<<1, range, range * sizeof(int)>>>(bucket, range);
  reset_key<<<1, n>>>(key, bucket, n, range);
  scan<<<1, n, n * sizeof(int)>>>(key, n);
  cudaDeviceSynchronize();
}

int main() {
  int n = 50;
  int range = 5;
  // std::vector<int> key(n);
  int *key;
  cudaMallocManaged(&key, n * sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // std::vector<int> bucket(range); 
  // for (int i=0; i<range; i++) {
  //   bucket[i] = 0;
  // }
  // for (int i=0; i<n; i++) {
  //   bucket[key[i]]++;
  // }

  // for (int i=0, j=0; i<range; i++) {
  //   for (; bucket[i]>0; bucket[i]--) {
  //     key[j++] = i;
  //   }
  // }

  bucket_sort(key, n, range);

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
