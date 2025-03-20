#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

#define WIDTH 8  // 행렬 가로 크기
#define HEIGHT 8 // 행렬 세로 크기
#define PATCH_SIZE 3 // 3x3 Convolution

// ✅ 3x3 Convolution을 수행하는 GPU 커널
__global__ void gpu_convolution(const float *input, float *output, const float *kernel, int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;  // X 좌표
    int y = threadIdx.y + blockIdx.y * blockDim.y;  // Y 좌표

    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {  // 경계 제외
        float sum = 0.0f;

        // ✅ 3x3 Convolution 적용
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int input_idx = (y + ky) * width + (x + kx);
                int kernel_idx = (ky + 1) * PATCH_SIZE + (kx + 1);
                sum += input[input_idx] * kernel[kernel_idx];
            }
        }

        output[y * width + x] = sum;  // 결과 저장
    }
}

// ✅ CPU에서 3x3 Convolution 수행
void cpu_convolution(const float *input, float *output, const float *kernel, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum = 0.0f;

            // ✅ 3x3 Convolution 적용
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int input_idx = (y + ky) * width + (x + kx);
                    int kernel_idx = (ky + 1) * PATCH_SIZE + (kx + 1);
                    sum += input[input_idx] * kernel[kernel_idx];
                }
            }

            output[y * width + x] = sum;  // 결과 저장
        }
    }
}

// ✅ 행렬 초기화 (테스트용 랜덤 값)
void initialize_matrix(float *matrix, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        matrix[i] = static_cast<float>(rand() % 10);  // 0~9 랜덤 값
    }
}

// ✅ 행렬 출력 (디버깅용)
void print_matrix(const float *matrix, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << matrix[y * width + x] << "\t";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    // ✅ 1. Unified Memory(UMA) 할당
    float *input, *output_cpu, *output_gpu, *kernel;
    cudaMallocManaged(&input, WIDTH * HEIGHT * sizeof(float));
    cudaMallocManaged(&output_cpu, WIDTH * HEIGHT * sizeof(float));
    cudaMallocManaged(&output_gpu, WIDTH * HEIGHT * sizeof(float));
    cudaMallocManaged(&kernel, PATCH_SIZE * PATCH_SIZE * sizeof(float));

    // ✅ 2. CPU에서 데이터 초기화
    initialize_matrix(input, WIDTH, HEIGHT);
    
    // ✅ 3. 3x3 Convolution 필터 초기화 (Edge Detection 필터 예제)
    float filter[PATCH_SIZE * PATCH_SIZE] = { 
        -1, -1, -1, 
        -1,  8, -1, 
        -1, -1, -1 
    };
    for (int i = 0; i < PATCH_SIZE * PATCH_SIZE; i++) {
        kernel[i] = filter[i];
    }

    std::cout << "🔹 Input Matrix:\n";
    print_matrix(input, WIDTH, HEIGHT);

    // ✅ 4. CPU에서 연산 수행
    auto start_time = std::chrono::high_resolution_clock::now();
    cpu_convolution(input, output_cpu, kernel, WIDTH, HEIGHT);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "🔹 CPU Convolution Time: " << cpu_duration.count() << " microseconds\n";
    std::cout << "🔹 CPU Convolution Result:\n";
    print_matrix(output_cpu, WIDTH, HEIGHT);

    // ✅ 5. GPU에서 연산 수행 (페이지 폴트 방지)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    auto gpu_start_time = std::chrono::high_resolution_clock::now();
    cudaMemPrefetchAsync(input, WIDTH * HEIGHT * sizeof(float), 0);  // GPU로 데이터 전송
    cudaMemPrefetchAsync(kernel, PATCH_SIZE * PATCH_SIZE * sizeof(float), 0);
    cudaDeviceSynchronize();  // GPU 연산 전 동기화
    dim3 threadsPerBlock(8, 8);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);
    gpu_convolution<<<numBlocks, threadsPerBlock>>>(input, output_gpu, kernel, WIDTH, HEIGHT);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "🔹 GPU Convolution Time: " << milliseconds << " milliseconds\n";
    cudaDeviceSynchronize();  // GPU 연산이 끝날 때까지 대기
    auto gpu_end_time = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end_time - gpu_start_time);
    std::cout << "🔹 GPU Convolution Time: " << gpu_duration.count() << " microseconds\n";


    // ✅ 6. GPU 결과를 CPU로 가져오기
    cudaMemPrefetchAsync(output_gpu, WIDTH * HEIGHT * sizeof(float), cudaCpuDeviceId);
    cudaDeviceSynchronize();

    std::cout << "🔹 GPU Convolution Result:\n";
    print_matrix(output_gpu, WIDTH, HEIGHT);

    // ✅ 7. 메모리 해제
    cudaFree(input);
    cudaFree(output_cpu);
    cudaFree(output_gpu);
    cudaFree(kernel);

    return 0;
}
