#include <cuda_runtime.h>

// 一个 CUDA block 完成整个数组的排序。
// 前提：n 是 2 的幂且 n <= 1024，因此一个 block 可以为每个元素分配一个线程。
__global__ void bitonicSortKernel(const float* input, float* output, int n) {
    // 使用共享内存保存待排序的数据。
    // 共享内存由 block 内线程共同访问，比反复访问全局内存更合适。
    __shared__ float values[1024];

    // 每个线程负责数组中一个位置。
    const unsigned int tid = threadIdx.x;

    // 将输入元素从全局内存读入共享内存。
    values[tid] = input[tid];

    // 等待所有线程完成数据加载，确保后续比较交换可以读取完整数组。
    __syncthreads();

    // k 表示当前 bitonic 排序阶段处理的区间长度：
    // 依次为 2、4、8、……、n。
    for (unsigned int k = 2; k <= static_cast<unsigned int>(n); k <<= 1) {
        // j 表示当前比较的距离：
        // 在每个长度为 k 的阶段中，依次为 k/2、k/4、……、1。
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            // XOR 可以快速找到当前线程的比较伙伴。
            // 例如 j=1 时比较相邻元素；j=2 时比较相距 2 的元素。
            const unsigned int partner = tid ^ j;

            // 一对元素由其中一个线程负责即可。
            // partner > tid 能避免两个线程重复处理同一对元素。
            if (partner > tid) {
                // 根据 tid 的第 k 位决定当前区间的排序方向：
                // 第 k 位为 0 时升序，为 1 时降序。
                // 最后一个阶段 k=n 时，所有 tid<n 的第 n 位都是 0，
                // 所以最终结果整体升序。
                const bool ascending = (tid & k) == 0;

                // 先读取两个元素，再按当前方向决定是否交换。
                const float a = values[tid];
                const float b = values[partner];

                // 升序时把较小值放在较小下标处；
                // 降序时把较大值放在较小下标处。
                if ((ascending && a > b) || (!ascending && a < b)) {
                    values[tid] = b;
                    values[partner] = a;
                }
            }

            // 当前距离 j 的所有比较交换完成后再继续下一轮。
            // 这样下一轮读取到的就是本轮更新后的共享内存数据。
            __syncthreads();
        }
    }

    // 排序完成，将共享内存中的结果写回全局内存。
    output[tid] = values[tid];
}

void solve(const float* input, float* output, int n) {
    // 在 GPU 全局内存中为输入和输出分别分配空间。
    float* deviceInput;
    float* deviceOutput;

    cudaMalloc(&deviceInput, n * sizeof(float));
    cudaMalloc(&deviceOutput, n * sizeof(float));

    // 将主机（CPU）输入复制到 GPU。
    cudaMemcpy(
        deviceInput,
        input,
        n * sizeof(float),
        cudaMemcpyHostToDevice
    );

    // n 是 2 的幂且不超过 1024，因此使用一个包含 n 个线程的 block。
    // 每个线程对应一个数组元素。
    bitonicSortKernel<<<1, n>>>(deviceInput, deviceOutput, n);

    // 将排序结果从 GPU 复制回主机。
    // 该复制会等待前面的 kernel 完成。
    cudaMemcpy(
        output,
        deviceOutput,
        n * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    // 释放 GPU 上分配的内存。
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
}