#include<cuda_runtime.h>
#include<bits/stdc++.h>

#define N 128 // 数组大小

__global__ void smem(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  a[tid] =smem[tid];
}

__global__ void smem_3(uint32_t *a) {
  __shared__ uint32_t smem[128];
  uint32_t tid = threadIdx.x;
  for (int i = 0; i < 4; i++) {
    smem[i * 32 + tid] = tid;
  }
  __syncthreads();
  reinterpret_cast<uint2 *>(a)[tid] =
      reinterpret_cast<const uint2 *>(smem)[tid/2];
}


int main() {
    // 在主机上声明一个数组
    uint32_t a[N];

    // 在设备上声明一个数组
    uint32_t *d_a;
    cudaMalloc((void**)&d_a, N * sizeof(uint32_t));

    // 调用 CUDA 核函数
    // smem<<<1, 32>>>(d_a);
    smem_3<<<1, 32>>>(d_a);

    // 将结果从设备复制回主机
    cudaMemcpy(a, d_a, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Result:\n");
    for (int i = 0; i < 4; i++) {
        for(int j = 0;j<32;j++){
            printf("%d ", a[i * 32 + j]);
        }
        printf("\n");
    }
    printf("\n");

    // 释放设备上的数组
    cudaFree(d_a);

    return 0;
}
