#include <cuda_runtime.h>
#include <iostream>

// 🚀 GPU에서 실행되는 커널 함수 (병렬 연산)
__global__ void gpu_compute(int *data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] *= 2;  // GPU에서 연산
    }
}

// 🚀 CPU에서 실행되는 함수 (단일 스레드 연산)
void cpu_compute(int *data, int N) {
    for (int i = 0; i < N; i++) {
        data[i] += 1;  // CPU에서 연산
    }
}

int main() {
    const int N = 1000000;  // 데이터 크기
    int *data;  // Unified Memory를 사용할 포인터

    // ✅ 1. Unified Memory(UMA) 할당 (CPU & GPU 공유 메모리)
    cudaMallocManaged(&data, N * sizeof(int));

    // ✅ 2. CPU에서 데이터를 주로 사용할 것이라고 CUDA에 힌트 제공
    cudaMemAdvise(data, N * sizeof(int), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

    // ✅ 3. CPU에서 데이터 초기화
    cpu_compute(data, N);

    // ✅ 4. GPU에서 데이터를 사용할 것이므로, 미리 GPU로 데이터 전송 (페이지 폴트 방지)
    cudaMemPrefetchAsync(data, N * sizeof(int), 0);  // 0 = GPU
    cudaDeviceSynchronize();  // GPU 연산 전 동기화

    // ✅ 5. GPU에서 데이터 연산 실행 (병렬 연산)
    gpu_compute<<<(N + 255) / 256, 256>>>(data, N);
    cudaDeviceSynchronize();  // GPU 연산이 끝날 때까지 대기

    // ✅ 6. 다시 CPU에서 데이터를 사용할 예정이므로 미리 CPU로 데이터 전송
    cudaMemPrefetchAsync(data, N * sizeof(int), cudaCpuDeviceId);
    cudaDeviceSynchronize();  // CPU 연산 전 동기화

    // ✅ 7. CPU에서 추가 연산
    cpu_compute(data, N);

    // ✅ 8. 연산 결과 출력 (첫 번째 데이터 확인)
    std::cout << "Result: " << data[0] << std::endl;

    // ✅ 9. 메모리 해제 (Unified Memory 반환)
    cudaFree(data);

    return 0;
}
