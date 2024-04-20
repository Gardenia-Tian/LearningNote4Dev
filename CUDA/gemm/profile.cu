#include "gemm.cuh"



int main(int argc, char* argv[]) {
    // constexpr int M = 40960;
    // constexpr int K = 20480;
    // constexpr int N = 40960;
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    // float* a = (float*)malloc(sizeof(float) * M * K);
    // float* b = (float*)malloc(sizeof(float) * K * N);
    float* c = (float*)malloc(sizeof(float) * M * N);
    float* test = (float*)malloc(sizeof(float) * M * N);

    float* da, * db, * dc;
    float* dtest;
    cudaMalloc((void**)&da, sizeof(float) * M * K);
    cudaMalloc((void**)&db, sizeof(float) * K * N);
    cudaMalloc((void**)&dc, sizeof(float) * M * N);
    cudaMalloc((void**)&dtest, sizeof(float) * M * N);

    dim3 blockSize(32, 32);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    randomInit<float> << <gridSize, blockSize >> > (da, M, K, time(NULL));

    gridSize.x = (N + blockSize.x - 1) / blockSize.x;
    gridSize.y = (K + blockSize.y - 1) / blockSize.y;
    randomInit<float> << <gridSize, blockSize >> > (db, K, N, time(NULL));

    gridSize.x = (N + blockSize.x - 1) / blockSize.x;
    gridSize.y = (M + blockSize.y - 1) / blockSize.y;


    gemm_v0<float> << <gridSize, blockSize >> > (da, db, dtest, M, N, K);
    // gemm_v1<float><<<gridSize, blockSize>>>(da, db, dtest, M, N, K);
    // gemm_shared_tile<32, 32, 32, float> << <gridSize, blockSize >> > (da, db, dc, M, N, K);
    gemm_shared_transposeA_tile<32, 32, 32, float> << <gridSize, blockSize >> > (da, db, dc, M, N, K);

    cudaMemcpy(c, dc, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(test, dtest, sizeof(float) * M * N, cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();

    bool resRight = checkVal(c, test, M*N);
    printf("%d\n", resRight);


    free(c);
    free(test);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cudaFree(dtest);

    return 0;
}
