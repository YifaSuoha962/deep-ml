#include <cuda_runtime.h>  // 引入 CUDA 运行时 API

// 对数组进行分块求和：每个线程块计算一个部分和，再累加到 out。
__global__ void kernel(const float* in, float* out, int N) {
    extern __shared__ float sdata[];  // 声明动态共享内存，大小由 kernel 启动参数指定

    const unsigned int tid = threadIdx.x;  // 当前线程在线程块内的索引
    const unsigned int index = blockIdx.x * blockDim.x + tid;  // 当前线程对应的全局数组索引

    // 如果索引在数组范围内，就读取对应元素；越界线程写入加法单位元 0。
    sdata[tid] = (index < static_cast<unsigned int>(N)) ? in[index] : 0.0f;

    __syncthreads();  // 等待线程块中的所有线程完成共享内存写入

    // 每轮把参与归约的线程数减半，直到只剩线程 0 保存块内总和。
    // 这里要求 blockDim.x 是 2 的幂；下面的 solve 使用 256 个线程。
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // 当前线程负责将自己的值与相隔 stride 的值相加。
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }

        __syncthreads();  // 确保本轮加法完成后，再开始下一轮
    }

    // 线程 0 持有当前线程块的部分和，将其累加到设备端总和。
    if (tid == 0) {
        atomicAdd(out, sdata[0]);  // 原子操作，避免多个线程块同时更新时发生冲突
    }
}

// 主机端函数：分配设备内存、复制数据、启动 kernel，并将结果复制回主机。
void solve(const float* input, float* output, int N) {
    // 空数组的总和为 0，无需启动 kernel。
    if (N <= 0) {
        *output = 0.0f;  // 直接将结果写入主机端输出指针
        return;          // 提前结束函数
    }

    constexpr int threads = 256;  // 每个线程块使用 256 个线程；这是 2 的幂
    const int blocks = (N + threads - 1) / threads;  // 计算覆盖 N 个元素所需的线程块数量

    float* d_input = nullptr;   // 声明设备端输入指针
    float* d_output = nullptr;  // 声明设备端总和指针

    // 为输入数组在 GPU 上分配空间。
    cudaMalloc(&d_input, static_cast<size_t>(N) * sizeof(float));

    // 为单个总和结果在 GPU 上分配空间。
    cudaMalloc(&d_output, sizeof(float));

    // 将主机端输入数组复制到设备端。
    cudaMemcpy(
        d_input,                              // 目标：设备端输入数组
        input,                                // 来源：主机端输入数组
        static_cast<size_t>(N) * sizeof(float), // 复制 N 个 float 所需的字节数
        cudaMemcpyHostToDevice                 // 指定复制方向：主机到设备
    );

    // 将设备端总和初始化为 0，供各线程块通过 atomicAdd 累加。
    cudaMemset(d_output, 0, sizeof(float));

    // 启动 kernel：blocks 个线程块，每块 threads 个线程；
    // 第三个参数指定每个线程块所需的动态共享内存大小。
    kernel<<<blocks, threads, threads * sizeof(float)>>>(
        d_input,   // 设备端输入数组
        d_output,  // 设备端总和
        N          // 输入元素个数
    );

    // 将设备端计算结果复制回主机端 output 指向的位置。
    cudaMemcpy(
        output,                   // 目标：主机端输出
        d_output,                 // 来源：设备端总和
        sizeof(float),            // 只复制一个 float
        cudaMemcpyDeviceToHost    // 指定复制方向：设备到主机
    );

    // 释放输入数组的设备端内存。
    cudaFree(d_input);

    // 释放总和结果的设备端内存。
    cudaFree(d_output);
}