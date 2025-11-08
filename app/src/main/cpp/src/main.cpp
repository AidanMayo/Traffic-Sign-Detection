#include "tensor.hpp"
#include "logger.hpp"
#include <iostream>
#include <chrono>
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

void testTensorOps(Tensor& t1, Tensor& t2, const std::string& deviceName) {
    LOG_INFO("Starting tensor operations test on " + deviceName);

    auto start = std::chrono::high_resolution_clock::now();
    t1.fill(1.0f);
    t2.fill(2.0f);
    auto end = std::chrono::high_resolution_clock::now();
    long long fillTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << deviceName << " Fill Time: " << fillTime << " us" << std::endl;
    LOG_PERFORMANCE("Fill Operations", deviceName, fillTime, "t1.fill(1.0f) + t2.fill(2.0f)");

    start = std::chrono::high_resolution_clock::now();
    t1.addTensor(t2);
    end = std::chrono::high_resolution_clock::now();
    long long addTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << deviceName << " Add Tensor Time: " << addTime << " us" << std::endl;
    LOG_PERFORMANCE("Add Tensor", deviceName, addTime, "");

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
    // t1.printImageTensor(); // Commented out to avoid massive output
}

int main() {
    // Initialize logger
    Logger::getInstance().initialize("traffic_sign_app.log", LogLevel::INFO, true);
    LOG_INFO("=== Traffic Sign Recognition Application Starting ===");

    const std::vector<int> shape = {4, 3, 512, 512};
    LOG_INFO("Creating tensors with shape: [4, 3, 512, 512]");

    Tensor inputCpu(shape, Device::CPU);
    LOG_INFO("Created CPU input tensor");

    inputCpu.fill(1.0f);
    LOG_DEBUG("Filled CPU tensor with 1.0");

    Tensor bias({3}, Device::CPU);
    bias.edit({0}, 2);
    bias.edit({1}, 3);
    bias.edit({2}, 4);
    LOG_INFO("Created bias tensor with values [2, 3, 4]");

    auto start = std::chrono::high_resolution_clock::now();
    inputCpu.addBias(bias);
    auto end = std::chrono::high_resolution_clock::now();
    long long cpuBiasTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "CPU Bias Time: " << cpuBiasTime << " microseconds" << std::endl;
    LOG_PERFORMANCE("CPU Bias Addition", "CPU", cpuBiasTime, "4x3x512x512 tensor with 3-channel bias");

    std::cout << "\n=== CPU Tensor Operations Test ===" << std::endl;

    // CPU tensors for testing
    Tensor cpuT1(shape, Device::CPU);
    Tensor cpuT2(shape, Device::CPU);
    testTensorOps(cpuT1, cpuT2, "CPU");

#ifdef USE_CUDA
    std::cout << "\n=== GPU/CUDA Operations Test ===" << std::endl;
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "CUDA device count: " << deviceCount << std::endl;
    LOG_INFO("CUDA device count: " + std::to_string(deviceCount));

    if (deviceCount > 0) {
        LOG_INFO("Creating GPU tensors");
        Tensor inputGpu(shape, Device::GPU);
        LOG_INFO("Created GPU input tensor");

        inputGpu.fill(1.0f);
        LOG_DEBUG("Filled GPU tensor with 1.0");

        Tensor biasGpu({3}, Device::GPU);
        biasGpu.edit({0}, 2);
        biasGpu.edit({1}, 3);
        biasGpu.edit({2}, 4);
        LOG_INFO("Created GPU bias tensor with values [2, 3, 4]");

        start = std::chrono::high_resolution_clock::now();
        inputGpu.addBias(biasGpu);
        end = std::chrono::high_resolution_clock::now();
        long long gpuBiasTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "GPU Bias Time: " << gpuBiasTime << " microseconds" << std::endl;
        LOG_PERFORMANCE("GPU Bias Addition", "GPU", gpuBiasTime, "4x3x512x512 tensor with 3-channel bias");

        // Log performance comparison
        double speedup = (double)cpuBiasTime / gpuBiasTime;
        LOG_INFO("GPU speedup over CPU: " + std::to_string(speedup) + "x");

        // inputGpu.printImageTensor();

        // Tensor gpuT1(shape, Device::GPU);
        // Tensor gpuT2(shape, Device::GPU);
        // testTensorOps(gpuT1, gpuT2, "GPU");
    } else {
        std::cout << "No CUDA device found. Skipping GPU tests." << std::endl;
        LOG_WARNING("No CUDA device found. Skipping GPU tests.");
    }
#else
    std::cout << "\n=== GPU/CUDA Operations Test ===" << std::endl;
    std::cout << "CUDA support not compiled in this build." << std::endl;
    std::cout << "To enable GPU operations, compile with proper CUDA toolkit setup." << std::endl;
    LOG_INFO("CUDA support not compiled in this build");
#endif

    std::cout << "\n=== Application Summary ===" << std::endl;
    std::cout << "✓ CPU tensor operations completed successfully" << std::endl;
    std::cout << "✓ Memory allocations tracked" << std::endl;
    std::cout << "✓ Performance metrics logged" << std::endl;
    std::cout << "✓ Comprehensive logging system active" << std::endl;

    LOG_INFO("=== Traffic Sign Recognition Application Completed Successfully ===");
    Logger::getInstance().close();
    return 0;
}
