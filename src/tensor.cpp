#include "tensor.hpp"
#include <algorithm>
#include <iostream>
#include <ostream>

Tensor::Tensor() : device(Device::CPU) {}

Tensor::Tensor(const std::vector<int>& shape_, Device dev) : shape(shape_), device(dev) {
    totalSize = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
	cpuData.resize(totalSize, 0.0f);
	cpuGrad.resize(totalSize, 0.0f);
	computeStrides();

#ifdef USE_CUDA
	if (device == Device::GPU) {
		toGpu();
	}
#endif
}

Tensor::~Tensor() {
#ifdef USE_CUDA
  freeGpuMemory();
#endif
}

void Tensor::computeStrides() {
 	strides.resize(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      	strides[i] = stride;
        stride *= shape[i];
    }
}


// Convenience accessors
float& Tensor::at(const int i) {
  	assert(shape.size() == 1 && device == Device::CPU);
    return cpuData[i * strides[0]];
}

float& Tensor::at(const int i, const int j) {
  	assert(shape.size() == 2 && device == Device::CPU);
    return cpuData[i * strides[0] + j * strides[1]];
}

float& Tensor::at(const int i, const int j, const int k) {
  	assert(shape.size() == 3 && device == Device::CPU);
    return cpuData[i * strides[0] + j * strides[1] + k * strides[2]];
}

float& Tensor::at(const std::vector<int>& index) {
  	assert(index.size() == shape.size());
#ifdef USE_CUDA
    if (device == Device::GPU) copyCpu();
#endif
  	int local = 0;
	for (std::size_t i = 0; i < shape.size(); ++i) {
        assert(index[i] >= 0 && index[i] < shape[i]);
        local += index[i] * strides[i];
	}
	return cpuData[local];
}

void Tensor::fillCpu(float val) {
    std::fill(cpuData.begin(), cpuData.end(), val);
}

void Tensor::addTensorCpu(const Tensor &other) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] += other.cpuData[i];
    }
}

void Tensor::addScalarCpu(const float val) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] += val;
    }
}

void Tensor::subtractTensorCpu(const Tensor &other) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] -= other.cpuData[i];
    }
}

void Tensor::subtractScalarCpu(const float val) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] -= val;
    }
}

void Tensor::multiplyTensorCpu(const Tensor &other) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] *= other.cpuData[i];
    }
}

void Tensor::multiplyScalarCpu(const float val) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] *= val;
    }
}

void Tensor::divideTensorCpu(const Tensor &other) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] /= other.cpuData[i];
    }
}

void Tensor::divideScalarCpu(const float val) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] /= val;
    }
}

void Tensor::negateCpu() { // bitwise hacking not allowed
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = -cpuData[i];
    }
}

void Tensor::ReLUCpu() {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = std::fmaxf(cpuData[i], 0.0f);
    }
}

void Tensor::sigmoidCpu() {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = 1.0f / (1.0f + std::expf(-cpuData[i]));
    }
}

void Tensor::tanhCpu() {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = std::tanhf(cpuData[i]);
    }
}

void Tensor::LReLUCpu(const float alpha) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = cpuData[i] < 0 ? alpha * cpuData[i] : cpuData[i];
    }
}

void Tensor::ELUCpu(const float alpha) {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = cpuData[i] < 0 ? alpha * (std::expf(cpuData[i]) - 1) : cpuData[i];
    }
}

void Tensor::squareCpu() {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = cpuData[i] * cpuData[i];
    }
}

void Tensor::sqrtCpu() {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = std::sqrtf(cpuData[i]);
    }
}

void Tensor::expCpu() {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = std::expf(cpuData[i]);
    }
}

void Tensor::logCpu() {
    for (std::size_t i = 0; i < cpuData.size(); ++i) {
        cpuData[i] = std::logf(cpuData[i]);
    }
}

void Tensor::zeroGradCpu() {
    std::fill(cpuGrad.begin(), cpuGrad.end(), 0.0f);
}


void Tensor::fill(const float val) {
  	if (device == Device::CPU) {
  	    fillCpu(val);
  	}
#ifdef USE_CUDA
    else {
        fillGpu(val);
  	}
#endif
}

void Tensor::addTensor(const Tensor& other) {
  	assert(shape == other.shape);
    if (device == Device::CPU) {
        addTensorCpu(other);
    }
#ifdef USE_CUDA
    else {
        addTensorGpu(other);
    }
#endif
}

void Tensor::addScalar(const float val) {
    if (device == Device::CPU) {
        addScalarCpu(val);
    }
#ifdef USE_CUDA
    else {
        addScalarGpu(val);
    }
#endif
}

void Tensor::subtractTensor(const Tensor& other) {
    assert(shape == other.shape);
    if (device == Device::CPU) {
        subtractTensorCpu(other);
    }
#ifdef USE_CUDA
    else {
        subtractTensorGpu(other);
    }
#endif
}

void Tensor::subtractScalar(const float val) {
    if (device == Device::CPU) {
        subtractScalarCpu(val);
    }
#ifdef USE_CUDA
    else {
        subtractScalarGpu(val);
    }
#endif
}

void Tensor::multiplyTensor(const Tensor& other) {
    assert(shape == other.shape);
    if (device == Device::CPU) {
        multiplyTensorCpu(other);
    }
#ifdef USE_CUDA
    else {
        multiplyTensorGpu(other);
    }
#endif
}

void Tensor::multiplyScalar(const float val) {
    if (device == Device::CPU) {
        multiplyScalarCpu(val);
    }
#ifdef USE_CUDA
    else {
        multiplyScalarGpu(val);
    }
#endif
}

void Tensor::divideTensor(const Tensor& other) {
    assert(shape == other.shape);
    if (device == Device::CPU) {
        divideTensorCpu(other);
    }
#ifdef USE_CUDA
    else {
        divideTensorGpu(other);
    }
#endif
}

void Tensor::divideScalar(const float val) {
    if (device == Device::CPU) {
        divideScalarCpu(val);
    }
#ifdef USE_CUDA
    else {
        divideScalarGpu(val);
    }
#endif
}

void Tensor::negate() {
    if (device == Device::CPU) {
        negateCpu();
    }
#ifdef USE_CUDA
    else {
        negateGpu();
    }
#endif
}

void Tensor::ReLU() {
    if (device == Device::CPU) {
        ReLUCpu();
    }
#ifdef USE_CUDA
    else {
        ReLUGpu();
    }
#endif
}

void Tensor::sigmoid() {
    if (device == Device::CPU) {
        sigmoidCpu();
    }
#ifdef USE_CUDA
    else {
        sigmoidGpu();
    }
#endif
}

void Tensor::tanh() {
    if (device == Device::CPU) {
        tanhCpu();
    }
#ifdef USE_CUDA
    else {
        tanhGpu();
    }
#endif
}

void Tensor::LReLU(const float alpha) {
    if (device == Device::CPU) {
        LReLUCpu(alpha);
    }
#ifdef USE_CUDA
    else {
        LReLUGpu(alpha);
    }
#endif
}

void Tensor::ELU(const float alpha) {
    if (device == Device::CPU) {
        ELUCpu(alpha);
    }
#ifdef USE_CUDA
    else {
        ELUGpu(alpha);
    }
#endif
}

void Tensor::square() {
    if (device == Device::CPU) {
        squareCpu();
    }
#ifdef USE_CUDA
    else {
        squareGpu();
    }
#endif
}

void Tensor::sqrt() {
    if (device == Device::CPU) {
        sqrtCpu();
    }
#ifdef USE_CUDA
    else {
        sqrtGpu();
    }
#endif
}

void Tensor::exp() {
    if (device == Device::CPU) {
        expCpu();
    }
#ifdef USE_CUDA
    else {
        expGpu();
    }
#endif
}

void Tensor::log() {
    if (device == Device::CPU) {
        logCpu();
    }
#ifdef USE_CUDA
    else {
        logGpu();
    }
#endif
}

void Tensor::zeroGrad() {
    if (device == Device::CPU) {
        zeroGradCpu();
    }
#ifdef USE_CUDA
    else {
        zeroGradGpu();
    }
#endif
}