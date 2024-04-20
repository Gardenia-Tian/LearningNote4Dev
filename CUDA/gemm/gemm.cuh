#include<cuda_runtime.h>
#include<bits/stdc++.h>
#include<curand_kernel.h>

// a : M * K    b: K * N   c : M * N
template<typename T>
__global__ void gemm_v0(T *a, T *b, T *c, int M , int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    T sum = 0;
    if (row < M && col < N) {
        for (int i = 0; i < K; ++i) {
            sum += a[row * K + i] * b[i * N + col];
        }
        c[row * N + col] = sum;
    }
}

template<int M_tile, int N_tile, int K_tile,typename T>
__global__ void gemm_shared_tile(T *a, T *b, T *c, int M , int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // constexpr int M_tile = 32;
    // constexpr int N_tile = 32;
    // constexpr int K_tile = 32;

    __shared__ T s_a[M_tile][K_tile];
    __shared__ T s_b[K_tile][N_tile];

    float val = 0;
    for(int ph = 0; ph < (K + K_tile - 1)/K_tile;ph++){
        if(row < M && (ph * K_tile + tx) < K){
            s_a[ty][tx] = a[row * K + (ph * K_tile + tx)];
        }else{
            s_a[ty][tx] = 0;
        }
        if((ph * K_tile + ty) < K && col < N){
            s_b[ty][tx] = b[(ph * K_tile + ty) * N + col];
        }else{
            s_b[ty][tx] = 0;
        }
        __syncthreads();
        for(int i = 0;i<K_tile;i++){
            val += s_a[ty][i] * s_b[i][tx];
        }
        __syncthreads();
    }
    if(row < M && col < N){
        c[row * N + col] = val;
    }

}

template<int M_tile, int N_tile, int K_tile,typename T>
__global__ void gemm_shared_transposeA_tile(T *a, T *b, T *c, int M , int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    __shared__ T s_a[M_tile][K_tile];
    __shared__ T s_b[K_tile][N_tile];

    float val = 0;
    for(int ph = 0; ph < (K + K_tile - 1)/K_tile;ph++){
        if(row < M && (ph * K_tile + tx) < K){
            s_a[tx][ty] = a[row * K + (ph * K_tile + tx)];
        }else{
            s_a[ty][tx] = 0;
        }
        if((ph * K_tile + ty) < K && col < N){
            s_b[ty][tx] = b[(ph * K_tile + ty) * N + col];
        }else{
            s_b[ty][tx] = 0;
        }
        __syncthreads();
        for(int i = 0;i<K_tile;i++){
            val += s_a[i][ty] * s_b[i][tx];
        }
        __syncthreads();
    }
    if(row < M && col < N){
        c[row * N + col] = val;
    }

}


template<typename T>
__global__ void randomInit(T* matrix, int rows, int cols, unsigned int seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int index = row * cols + col;

    if (row < rows && col < cols) {
        // 使用 cuRAND 来生成随机数
        curandState state;
        curand_init(seed, index, 0, &state);
        matrix[index] = static_cast<T>(curand_uniform(&state));
    }
}


template<typename T>
bool checkVal(T* a, T* b, int m){
    for(int i = 0;i<m;i++){
        if(a[i] != b[i]) return false;
    }
    return true;
}

template<typename T>
void logMatrix(T* matrix, int row, int col){
    for(int i = 0;i< row;i++){
        for(int j = 0;j<col;j++){
            printf("%.2f ", matrix[i * col + j]);
        }
        printf("\n");
    }
}

template<typename T>
__global__ void logDevice(T* matrix, int row, int col){
    for(int i = 0;i< row;i++){
        for(int j = 0;j<col;j++){
            printf("%.2f ", matrix[i * col + j]);
        }
        printf("\n");
    }
}