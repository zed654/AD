#include <cuda_runtime.h>
#include <iostream>

__global__ void gpu_kernel(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] *= 2;  // GPU 연산
    }
}

void cpu_compute(int *data, int N) {
    for (int i = 0; i < N; i++) {
        data[i] += 1;  // CPU 연산
    }
}

int main() {
    const int N = 1000000;
    int *data;

    // Unified Memory 할당
    cudaMallocManaged(&data, N * sizeof(int));

    // CPU에서 데이터를 주로 사용할 것으로 설정
    cudaMemAdvise(data, N * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
    
    // CPU에서 초기화
    cpu_compute(data, N);

    // GPU에서 사용할 것이므로, 미리 GPU로 데이터를 전송
    cudaMemPrefetchAsync(data, N * sizeof(int), 0);  // 0 = GPU Device
    cudaDeviceSynchronize();

    // GPU에서 연산
    gpu_kernel<<<(N + 255) / 256, 256>>>(data, N);
    cudaDeviceSynchronize();

    // 다시 CPU에서 사용할 것이므로, 미리 CPU로 전송
    cudaMemPrefetchAsync(data, N * sizeof(int), cudaCpuDeviceId);
    cudaDeviceSynchronize();

    // CPU에서 추가 연산
    cpu_compute(data, N);

    // 결과 출력
    std::cout << "data[0]: " << data[0] << std::endl;
    
    // 메모리 해제
    cudaFree(data);

    return 0;
}
