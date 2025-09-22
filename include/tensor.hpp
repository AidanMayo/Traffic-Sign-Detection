#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cassert>
#include <numeric>
#include <functional>

enum class Device { CPU, GPU };

class Tensor {
public:
    Tensor();
    Tensor(const std::vector<int>& shape_, Device dev = Device::CPU);
    ~Tensor();

    float& at(int i);
    float& at(int i, int j);
    float& at(int i, int j, int k);
    float& at(const std::vector<int>& index);

    // Utility
    void fill(float val);

    void reshape(const std::vector<int>& shape_);
    void flatten();

    // Arithmetic
	void addTensor(const Tensor& other);
    void addScalar(float val);

    void subtractTensor(const Tensor& other);
    void subtractScalar(float val);

    void multiplyTensor(const Tensor& other);
    void multiplyScalar(float val);

    void divideTensor(const Tensor& other);
    void divideScalar(float val);

    void negate();

    void ReLU();
    void sigmoid();
    void tanh();
    void LReLU(const float alpha);
    void ELU(const float alpha);

    void square();
    void sqrt();
    void exp();
    void log();

    // Gradiant
	void zeroGrad();

    // testing util
    void printShape() const;
    void printData() const;

#ifdef USE_CUDA
    void toGpu();
    void toCpu();
#endif

private:
	Device device;

    int totalSize;

    std::vector<int> shape;
    std::vector<int> strides;

    // CPU DATA
    std::vector<float> cpuData;
    std::vector<float> cpuGrad;

    void computeStrides();

    void fillCpu(float val);
    void addTensorCpu(const Tensor& other);
    void addScalarCpu(float val);

    void subtractTensorCpu(const Tensor& other);
    void subtractScalarCpu(float val);

    void multiplyTensorCpu(const Tensor& other);
    void multiplyScalarCpu(float val);

    void divideTensorCpu(const Tensor& other);
    void divideScalarCpu(float val);

    void negateCpu();

    void ReLUCpu();
    void sigmoidCpu();
    void tanhCpu();
    void LReLUCpu(const float alpha);
    void ELUCpu(const float alpha);

    void squareCpu();
    void sqrtCpu();
    void expCpu();
    void logCpu();

    void zeroGradCpu();

#ifdef USE_CUDA
    float* gpuData = nullptr;
    float* gpuGrad = nullptr;

    void freeGpuMemory();

    void fillGpu(float val);
    void addTensorGpu(const Tensor& other);
    void addScalarGpu(float val);

    void subtractTensorGpu(const Tensor& other);
    void subtractScalarGpu(float val);

    void multiplyTensorGpu(const Tensor& other);
    void multiplyScalarGpu(float val);

    void divideTensorGpu(const Tensor& other);
    void divideScalarGpu(float val);

    void negateGpu();

    void ReLUGpu();
    void sigmoidGpu();
    void tanhGpu();
    void LReLUGpu(const float alpha);
    void ELUGpu(const float alpha);

    void squareGpu();
    void sqrtGpu();
    void expGpu();
    void logGpu();

    void zeroGradGpu();

    void copyCpu();

#endif
};

#endif //TENSOR_H
