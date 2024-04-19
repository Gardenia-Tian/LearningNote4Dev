#include "gemm.cuh"



int main(){
    constexpr int M = 4;
    constexpr int K = 2;
    constexpr int N = 4;

    // float* a = (float*)malloc(sizeof(float) * M * K);
    // float* b = (float*)malloc(sizeof(float) * K * N);
    float* c = (float*)malloc(sizeof(float) * M * N);
    float* test = (float*)malloc(sizeof(float) * M * N);

    float *da, *db, *dc;
    float *dtest;
    cudaMalloc((void**)&da, sizeof(float) * M * K);
    cudaMalloc((void**)&db, sizeof(float) * K * N);
    cudaMalloc((void**)&dc, sizeof(float) * M * N);
    cudaMalloc((void**)&dtest, sizeof(float) * M * N);
    
    dim3 blockSize(32, 32);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    randomInit<float><<<gridSize, blockSize>>>(da, M, K, time(NULL));

    gridSize.x = (N + blockSize.x - 1) / blockSize.x;
    gridSize.y = (K + blockSize.y - 1) / blockSize.y;
    randomInit<float><<<gridSize, blockSize>>>(db, K, N, time(NULL));

    gridSize.x = (N + blockSize.x - 1) / blockSize.x;
    gridSize.y = (M + blockSize.y - 1) / blockSize.y;

    // sgemm_v1<<<numBlocks, threadsPerBlock>>>(da, db, dc, M, N, K);
    
    gemm_v0<float><<<gridSize, blockSize>>>(da, db, dtest, M, N, K);


    cudaMemcpy(c, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(test, dtest, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
 

    cudaDeviceSynchronize();

    // bool resRight = checkVal(c, test, M*N);
    // printf("%d\n", resRight);


    free(c);
    free(test);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dtest);

    return 0;
}
