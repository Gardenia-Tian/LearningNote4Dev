#include<cuda_runtime.h>
#include<bits/stdc++.h>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])


// CUDA 核函数，执行矩阵乘法
__global__ void gemm_v0(float *a, float *b, float *c,  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;

    if (row < N && col < N) {
        for (int i = 0; i < K; ++i) {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}


// smem + tile
// A : M * K   B: K * N  C: M * N
__global__ void sgemm_v1(float* a, float* b, float* c, int M, int N, int K){
    // constexpr 表示常量语义
    // const 表示只读语义
    // 定义每一块的大小, 相当于tile_width
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    __shared__ float s_a[BM][BK];
    __shared__ float s_b[BK][BN];

    int ty = threadIdx.y, tx = threadIdx.x;
    int by = blockIdx.y, bx = threadIdx.x; 
    // 在结果矩阵C中, 所计算的单元的位置
    int row = by * BM + ty;
    int col = bx * BN + tx;

    float tempVal = 0;

    for(int ph = 0; ph < (K + BK - 1)/ BK ;ph++){
        if(row < M && ph * BK + tx < K){
            s_a[ty][tx] = a[row * K + (ph * BK + tx)];
        }else{
            s_a[ty][tx] = 0;
        }
        if(ph * BK + ty<K && col < N){
            s_b[ty][tx] = b[(ph * BK + ty) * M + col];
        }else{
            s_b[ty][tx] = 0;
        }
        __syncthreads();

        for(int k = 0;k < BK ;k++){
            tempVal += s_a[ty][k] * s_b[k][tx];
        }
        __syncthreads();
    }
    if(row < M && col < N){
        c[row * N + col] = tempVal;
    }
}


/*
shared memo 大幅度提升访存效率, 进而提高性能, 但是shared memory会存在band conflict
现象, 如果线程访问的数据在同一个bank内, 就要串行访问了.
在A*B的过程中, 每个线程访问矩阵A的一行, 所有的线程访问的是矩阵的A的列. 
而矩阵A在shared memory中是按行存储的.
同一个warp的不同线程的同一条load指令的访存地址是被间隔开的, 存在band conflict
优化点:
1. 矩阵A的shared memory按列存储
2. 让矩阵A的load数据, 使用一个load指令就可以完成
*/

/*
通过向量化内存访问提高性能:
a的share mem按列存储
*/
__global__ void sgemm_v2(float* a, float* b, float* c, int M, int N, int K){
    constexpr int BM = 32;
    constexpr int BK = 32;
    constexpr int BN = 32;
    // a -> BM*BK
    // b -> BK*BN
    // c -> BM*BN

    __shared__ float sa[BK][BM];
    __shared__ float sb[BK][BN];


    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x, by = blockIdx.y;
    int row = by * BM + ty;
    int col = bx * BN + tx; 
    int tid = ty * blockDim.x + tx;

    float tempVal = 0;

    for(int ph = 0; ph < (BK + K - 1) / BK ; ph++){
        reinterpret_cast<float4*>(sa)[tx][ty] = reinterpret_cast<float4*>(a)[row * K + ph * BK + tx];
        sb[ty][tx] = b[(ph * BK + ty) * K + col];
        __syncthreads();
    }

}

bool checkVal(float* a, float* b, int m){
    for(int i = 0;i<m;i++){
        if(a[i] != b[i]) return false;
    }
    return true;
}



int main(){
    constexpr int M = 2048;
    constexpr int K = 1024;
    constexpr int N = 2048;
    float* a = (float*)malloc(sizeof(float) * M * K);
    float* b = (float*)malloc(sizeof(float) * K * N);
    float* c = (float*)malloc(sizeof(float) * M * N);
    float* test = (float*)malloc(sizeof(float) * M * N);
    float *da, *db, *dc;
    float *dtest;
    cudaMalloc((void**)&da, sizeof(float) * M * K);
    cudaMalloc((void**)&db, sizeof(float) * K * N);
    cudaMalloc((void**)&dc, sizeof(float) * M * N);
    cudaMalloc((void**)&dtest, sizeof(float) * M * N);
    
    cudaMemcpy(da, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);
    

    
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(M/32, N/32);
    
    
    


    sgemm_v1<<<numBlocks, threadsPerBlock>>>(da, db, dc, M, N, K);
    gemm_v0<<<numBlocks, threadsPerBlock>>>(da, db, dtest, M, N, K);
    
    cudaMemcpy(c, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(test, dtest, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    bool resRight = checkVal(c, test, M*N);
    printf("%d\n", resRight);

    free(a);
    free(b);
    free(c);
    free(test);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dtest);

    return 0;
}