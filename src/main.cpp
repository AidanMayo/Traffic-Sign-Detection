#include "tensor.hpp"
#include <iostream>
#include <chrono>
#include <cuda_runtime_api.h>

void testTensorOps(Tensor& t1, Tensor& t2, const std::string& deviceName) {
    auto start = std::chrono::high_resolution_clock::now();
    t1.fill(1.0f);
    t2.fill(2.0f);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Fill Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.addTensor(t2);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Add Tensor Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.addScalar(3.0f);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Add Scalar Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.subtractTensor(t2);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Subtract Tensor Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.subtractScalar(1.0f);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Subtract Scalar Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.multiplyTensor(t2);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Multiply Tensor Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.multiplyScalar(2.0f);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Multiply Scalar Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.divideTensor(t2);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Divide Tensor Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.divideScalar(2.0f);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Divide Scalar Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.negate();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Negate Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.ReLU();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " ReLU Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.LReLU(0.1f);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " LReLU Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.ELU(1.0f);
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " ELU Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.sigmoid();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Sigmoid Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.tanh();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Tanh Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.square();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Square Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.sqrt();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Sqrt Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.exp();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Exp Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.log();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " Log Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    t1.zeroGrad();
    end = std::chrono::high_resolution_clock::now();
    std::cout << deviceName << " ZeroGrad Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " us" << std::endl;

    t1.printShape();
    t1.printImageTensor();
}

int main() {
    const std::vector<int> shape = {4, 3, 512, 512};

    Tensor inputCpu(shape, Device::CPU);

    inputCpu.fill(1.0f);

    Tensor bias({3}, Device::CPU);
    bias.edit({0}, 2);
    bias.edit({1}, 3);
    bias.edit({2}, 4);

    auto start = std::chrono::high_resolution_clock::now();
    inputCpu.addBias(bias);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "CPU Bias Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << " microseconds" << std::endl;

    // inputCpu.printImageTensor();

    // CPU tensors
    // Tensor cpuT1(shape, Device::CPU);
    // Tensor cpuT2(shape, Device::CPU);
    // testTensorOps(cpuT1, cpuT2, "CPU");

#ifdef USE_CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        Tensor inputGpu(shape, Device::GPU);

        inputGpu.fill(1.0f);

        Tensor biasGpu({3}, Device::GPU);
        biasGpu.edit({0}, 2);
        biasGpu.edit({1}, 3);
        biasGpu.edit({2}, 4);

        start = std::chrono::high_resolution_clock::now();
        inputGpu.addBias(biasGpu);
        end = std::chrono::high_resolution_clock::now();
        std::cout << "GPU Bias Time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                  << " microseconds" << std::endl;


        // inputGpu.printImageTensor();

        // Tensor gpuT1(shape, Device::GPU);
        // Tensor gpuT2(shape, Device::GPU);
        // testTensorOps(gpuT1, gpuT2, "GPU");
    } else {
        std::cout << "No CUDA device found. Skipping GPU tests." << std::endl;
    }
#endif

    return 0;
}
