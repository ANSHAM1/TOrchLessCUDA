//#pragma once
//
//#include "tensor.hpp"
//
//
//
//class Layer {
//protected:
//	Tensor InputTensor;
//	std::vector<size_t> OutputShape;
//
//public:
//	virtual void forward(const Tensor<T>& Input, Tensor<T>& Output) = 0;
//
//	//virtual void backward(const Tensor<T>& UpstreamGrad, Tensor<T>& DownstreamGrad) = 0;
//
//	virtual const std::vector<size_t>& getOutputShape() const {
//		return OutputShape;
//	}
//
//	virtual ~Layer() {}
//};