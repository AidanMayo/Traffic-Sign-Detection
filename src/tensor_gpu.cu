#include "../include/tensor.hpp"
#include <cuda_runtime.h>
#include <iostream>

__global__ void fillKernel(float* data, int size, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = val;
    }
}

void Tensor::fillGpu(float val) {
    fillKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize, val);
    cudaDeviceSynchronize();
}

__global__ void addTensorKernel(float*a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] += b[idx];
}

void Tensor::addTensorGpu(const Tensor& other) {
    addTensorKernel<<<(totalSize + 255)/256, 256>>>(gpuData, other.gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void addScalarKernel(float*a, const float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] += val;
}

void Tensor::addScalarGpu(const float val) {
    addScalarKernel<<<(totalSize + 255)/256, 256>>>(gpuData, val, totalSize);
    cudaDeviceSynchronize();
}

__global__ void addBiasKernel(float* a, const float* bias, int channels, int size, int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalSize) {
        int c = (idx / size) % channels;
        a[idx] += bias[c];
    }
}

void Tensor::addBiasGpu(const Tensor& bias) {
    addBiasKernel<<<(totalSize + 255)/256, 256>>>(gpuData, bias.gpuData, shape[1], shape[2] * shape[3], totalSize);
    cudaDeviceSynchronize();
}

__global__ void subtractTensorKernel(float*a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] -= b[idx];
}

void Tensor::subtractTensorGpu(const Tensor& other) {
    subtractTensorKernel<<<(totalSize + 255)/256, 256>>>(gpuData, other.gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void subtractScalarKernel(float*a, const float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] -= val;
}

void Tensor::subtractScalarGpu(const float val) {
    subtractScalarKernel<<<(totalSize + 255)/256, 256>>>(gpuData, val, totalSize);
    cudaDeviceSynchronize();
}

__global__ void multiplyTensorKernel(float*a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] *= b[idx];
}

void Tensor::multiplyTensorGpu(const Tensor& other) {
    multiplyTensorKernel<<<(totalSize + 255)/256, 256>>>(gpuData, other.gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void multiplyScalarKernel(float*a, const float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] *= val;
}

void Tensor::multiplyScalarGpu(const float val) {
    multiplyScalarKernel<<<(totalSize + 255)/256, 256>>>(gpuData, val, totalSize);
    cudaDeviceSynchronize();
}

__global__ void divideTensorKernel(float*a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] /= b[idx];
}

void Tensor::divideTensorGpu(const Tensor& other) {
    divideTensorKernel<<<(totalSize + 255)/256, 256>>>(gpuData, other.gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void divideScalarKernel(float*a, const float val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] /= val;
}

void Tensor::divideScalarGpu(const float val) {
    divideScalarKernel<<<(totalSize + 255)/256, 256>>>(gpuData, val, totalSize);
    cudaDeviceSynchronize();
}

__global__ void negateKernel(float*a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = -a[idx];
}

void Tensor::negateGpu() {
    negateKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void ReLUKernel(float*a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = fmaxf(a[idx], 0);
}

void Tensor::ReLUGpu() {
    ReLUKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void sigmoidKernel(float*a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = 1.0 / (1.0 + expf(-1.0 * a[idx]));
}

void Tensor::sigmoidGpu() {
    sigmoidKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void tanhKernel(float*a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = tanhf(a[idx]);
}

void Tensor::tanhGpu() {
    tanhKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void LReLUKernel(float*a, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = a[idx] < 0 ? alpha * a[idx] : a[idx];
}

void Tensor::LReLUGpu(float alpha) {
    LReLUKernel<<<(totalSize + 255)/256, 256>>>(gpuData, alpha, totalSize);
    cudaDeviceSynchronize();
}

__global__ void ELUKernel(float*a, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = a[idx] < 0 ? alpha * (expf(a[idx]) - 1) : a[idx];
}

void Tensor::ELUGpu(float alpha) {
    ELUKernel<<<(totalSize + 255)/256, 256>>>(gpuData, alpha, totalSize);
    cudaDeviceSynchronize();
}

__global__ void squareKernel(float*a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = a[idx] * a[idx];
}

void Tensor::squareGpu() {
    squareKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void sqrtKernel(float*a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = sqrtf(a[idx]);
}

void Tensor::sqrtGpu() {
    sqrtKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void expKernel(float*a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = expf(a[idx]);
}

void Tensor::expGpu() {
    expKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void logKernel(float*a, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) a[idx] = logf(a[idx]);
}

void Tensor::logGpu() {
    logKernel<<<(totalSize + 255)/256, 256>>>(gpuData, totalSize);
    cudaDeviceSynchronize();
}

__global__ void zeroGradKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0;
    }
}

void Tensor::zeroGradGpu() {
    int threadsPerBlock = 256;
    int blocks = (totalSize + threadsPerBlock - 1) / threadsPerBlock;
    zeroGradKernel<<<blocks, threadsPerBlock>>>(gpuGrad, totalSize);
    cudaDeviceSynchronize();
}

void Tensor::freeGpuMemory() {
    if (gpuData) {
        cudaFree(gpuData);
        gpuData = nullptr;
    }

    if (gpuGrad) {
        cudaFree(gpuGrad);
        gpuGrad = nullptr;
    }
}


void Tensor::toGpu() {
    if (gpuData != nullptr) return;
    cudaMalloc(&gpuData, totalSize * sizeof(float));
    cudaMalloc(&gpuGrad, totalSize * sizeof(float));
    cudaMemcpy(gpuData, cpuData.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);
    device = Device::GPU;
}

void Tensor::toCpu() {
    if (gpuData == nullptr) return;
    cudaMemcpy(cpuData.data(), gpuData, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
    device = Device::CPU;
}

void Tensor::copyCpu() {
    if (gpuData == nullptr) return;
    cudaMemcpy(cpuData.data(), gpuData, totalSize * sizeof(float), cudaMemcpyDeviceToHost);
}

void Tensor::copyGpu() {
    if (gpuData == nullptr) return;
    cudaMemcpy(gpuData, cpuData.data(), totalSize * sizeof(float), cudaMemcpyHostToDevice);
}

