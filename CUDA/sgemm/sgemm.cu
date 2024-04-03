#include<cuda_runtime.h>
#include<bits/stdc++.h>


#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4*>(&(value))[0])


// smem + tile
// A : M * K   B: K * N  C: M * N
__global__ void sgemm(float* a, float* b, float* c, int M, int N, int K){
    // constexpr 表示常量语义
    // const 表示只读语义
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



