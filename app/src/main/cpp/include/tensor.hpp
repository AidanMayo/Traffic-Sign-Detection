#ifndef TRAFFIC_SIGN_DETECTION_TENSOR_HPP
#define TRAFFIC_SIGN_DETECTION_TENSOR_HPP

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

    void edit(const std::vector<int>& index, float val);

    // Utility
    void fill(float val);

    void reshape(const std::vector<int>& shape_);
    void flatten();

    void transpose(const std::vector<int>& order);

    Tensor broadcast(const std::vector<int>& newShape);

    void makeContiguous(); // for now our function will mutate the current tensor to be contiguous, if needed a function that returns a new contiguous tensor will be made

    // Arithmetic
    void addTensor(const Tensor& other);
    void addScalar(float val);
    void addBias(const Tensor& bias);

    void subtractTensor(const Tensor& other);
    void subtractScalar(float val);

    void multiplyTensor(const Tensor& other);
    void multiplyScalar(float val);
    void multiplyBias(const Tensor& bias);

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
    void printShape();
    void printData();
    void printImageTensor();

#ifdef USE_CUDA
    void toGpu();
    void toCpu();
#endif

private:
    Device device;

    int totalSize;

    bool contiguous;

    std::vector<int> shape;
    std::vector<int> strides;

    // CPU DATA
    std::vector<float> cpuData;
    std::vector<float> cpuGrad;

    void computeStrides();

    void makeContiguousCpu();

    void fillCpu(float val);
    void addTensorCpu(const Tensor& other);
    void addScalarCpu(float val);
    void addBiasCpu(const Tensor& bias);

    void subtractTensorCpu(const Tensor& other);
    void subtractScalarCpu(float val);

    void multiplyTensorCpu(const Tensor& other);
    void multiplyScalarCpu(float val);
    void multiplyBiasCpu(const Tensor& bias);

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

    void makeContiguousGpu();

    void fillGpu(float val);
    void addTensorGpu(const Tensor& other);
    void addScalarGpu(float val);
    void addBiasGpu(const Tensor& bias);

    void subtractTensorGpu(const Tensor& other);
    void subtractScalarGpu(float val);

    void multiplyTensorGpu(const Tensor& other);
    void multiplyScalarGpu(float val);
    void multiplyBiasGpu(const Tensor& bias);

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
    void copyGpu();

#endif
};

#endif //TRAFFIC_SIGN_DETECTION_TENSOR_HPP
