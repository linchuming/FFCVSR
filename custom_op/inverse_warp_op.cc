#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/types.h"
#include <stdio.h>
#include "inverse_warp.h"

using namespace tensorflow;

REGISTER_OP("InverseWarp")
.Attr("T: {float, double} = DT_FLOAT")
.Input("input: T")
.Input("flow: T")
.Output("output: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	return Status::OK();
});

REGISTER_OP("InverseWarpGrad")
.Attr("T: {float, double} = DT_FLOAT")
.Input("input: T")
.Input("flow: T")
.Input("output_grad: T")
.Output("input_grad: T")
.Output("flow_grad: T")
.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
	c->set_output(0, c->input(0));
	c->set_output(1, c->input(1));
	return Status::OK();
});

namespace tensorflow {
	typedef Eigen::ThreadPoolDevice CPUDevice;
	typedef Eigen::GpuDevice GPUDevice;

	// Define class InverseWarpOp
	template <typename Device, typename T>
	class InverseWarpOp : public OpKernel {
	public:
		explicit InverseWarpOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor = context->input(0);
			const Tensor& flow_tensor = context->input(1);
			OP_REQUIRES(context, input_tensor.dims() >= 3, 
				errors::InvalidArgument("input must be at least 3-D, got shape", input_tensor.shape().DebugString()));
			//OP_REQUIRES(context, flow_tensor.dims() >= 3,
			//	errors::InvalidArgument("flow data must be at least 3-D, got shape", flow_tensor.shape().DebugString()));
			OP_REQUIRES(context, input_tensor.dims() == flow_tensor.dims(),
				errors::InvalidArgument("input dims must be equal to flow dims"));

			Tensor* output_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
			OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
				errors::InvalidArgument("Too many elements in tensor"));

			// Only consider data_format == FORMAT_NHWC
			const int dims = input_tensor.dims();
			const int height = static_cast<int>(input_tensor.dim_size(dims - 3));
			const int width = static_cast<int>(input_tensor.dim_size(dims - 2));
			const int channels = static_cast<int>(input_tensor.dim_size(dims - 1));
			OP_REQUIRES(context, height == static_cast<int>(flow_tensor.dim_size(dims - 3)),
				errors::InvalidArgument("the height of input must be equal to the height of flow data"));
			OP_REQUIRES(context, width == static_cast<int>(flow_tensor.dim_size(dims - 2)),
				errors::InvalidArgument("the width of input must be equal to the width of flow data"));
			OP_REQUIRES(context, 2 == static_cast<int>(flow_tensor.dim_size(dims - 1)),
				errors::InvalidArgument("the channels of flow data must be 2"));

			if (input_tensor.NumElements() > 0) {
				const int batch = static_cast<int>(input_tensor.NumElements() / (height * width * channels));
				functor::InverseWarpFunctor<Device, T>()(
					context->eigen_device<Device>(),
					batch, height, width, channels,
					input_tensor.flat<T>().data(),
					flow_tensor.flat<T>().data(),
					output_tensor->flat<T>().data()
					);
			}
		}
	};	// class InverseWarpOp

	// Define class InverseWarpGradOp
	template <typename Device, typename T>
	class InverseWarpGradOp : public OpKernel {
	public:
		explicit InverseWarpGradOp(OpKernelConstruction* context) : OpKernel(context) {}

		void Compute(OpKernelContext* context) override {
			const Tensor& input_tensor = context->input(0);
			const Tensor& flow_tensor = context->input(1);
			const Tensor& grad_tensor = context->input(2);
			OP_REQUIRES(context, input_tensor.dims() >= 3,
				errors::InvalidArgument("input must be at least 3-D, got shape", input_tensor.shape().DebugString()));
			OP_REQUIRES(context, input_tensor.dims() == flow_tensor.dims(),
				errors::InvalidArgument("input dims must be equal to flow dims"));

			Tensor* input_grad_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &input_grad_tensor));
			Tensor* flow_grad_tensor = nullptr;
			OP_REQUIRES_OK(context, context->allocate_output(1, flow_tensor.shape(), &flow_grad_tensor));
			OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
				errors::InvalidArgument("Too many elements in tensor"));

			// Only consider data_format == FORMAT_NHWC
			const int dims = input_tensor.dims();
			const int height = static_cast<int>(input_tensor.dim_size(dims - 3));
			const int width = static_cast<int>(input_tensor.dim_size(dims - 2));
			const int channels = static_cast<int>(input_tensor.dim_size(dims - 1));
			OP_REQUIRES(context, height == static_cast<int>(flow_tensor.dim_size(dims - 3)),
				errors::InvalidArgument("the height of input must be equal to the height of flow data"));
			OP_REQUIRES(context, width == static_cast<int>(flow_tensor.dim_size(dims - 2)),
				errors::InvalidArgument("the width of input must be equal to the width of flow data"));
			OP_REQUIRES(context, 2 == static_cast<int>(flow_tensor.dim_size(dims - 1)),
				errors::InvalidArgument("the channels of flow data must be 2"));
			OP_REQUIRES(context, input_tensor.NumElements() == grad_tensor.NumElements(),
				errors::InvalidArgument("the number of input data must be equal to the number of output grad"));

			if (input_tensor.NumElements() > 0) {
				const int batch = static_cast<int>(input_tensor.NumElements() / (height * width * channels));
				functor::InverseWarpGradFunctor<Device, T>()(
					context->eigen_device<Device>(),
					batch, height, width, channels,
					input_tensor.flat<T>().data(),
					flow_tensor.flat<T>().data(),
					grad_tensor.flat<T>().data(),
					input_grad_tensor->flat<T>().data(),
					flow_grad_tensor->flat<T>().data());
			}
		}
	};	//	class InverseWarpGradOp
	
	namespace functor {
		//	Define class InverseWarpFunctor with CPUDevice.
		template <typename T>
		struct InverseWarpFunctor<CPUDevice, T> {
			void operator()(const CPUDevice& d,
			int batch, int height, int width, int channels,
			const T* in, const T* flow, T* out) {
				//printf("Run CPU Functor.\n");
				for (int n = 0; n < batch; n++) {
					for (int h = 0; h < height; h++) {
						for (int w = 0; w < width; w++) {
							// Get flow value.
							const T f_w = flow[n * height * width * 2 + h * width * 2 + w * 2 + 0];
							const T f_h = flow[n * height * width * 2 + h * width * 2 + w * 2 + 1];
							// Get target coordinate.
							const int f_h0 = floorf(f_h);
							const int f_w0 = floorf(f_w);
							const int f_h1 = f_h0 + 1;
							const int f_w1 = f_w0 + 1;

							const int _h0 = std::min(std::max(h + f_h0, 0), height - 1);
							const int _w0 = std::min(std::max(w + f_w0, 0), width - 1);
							const int _h1 = std::min(std::max(h + f_h1, 0), height - 1);
							const int _w1 = std::min(std::max(w + f_w1, 0), width - 1);

							const T w00 = (f_h1 - f_h) * (f_w1 - f_w);
							const T w01 = (f_h1 - f_h) * (f_w - f_w0);
							const T w10 = (f_h - f_h0) * (f_w1 - f_w);
							const T w11 = (f_h - f_h0) * (f_w - f_w0);

							const int feature_size = height * width * channels;
							const int row_size = width * channels;
							for (int c = 0; c < channels; c++) {
								// Calculate interpolation value.
								const T v00 = in[n * feature_size + _h0 * row_size + _w0 * channels + c];
								const T v01 = in[n * feature_size + _h0 * row_size + _w1 * channels + c];
								const T v10 = in[n * feature_size + _h1 * row_size + _w0 * channels + c];
								const T v11 = in[n * feature_size + _h1 * row_size + _w1 * channels + c];

								const int t_index = n * feature_size + h * row_size + w * channels + c;
								out[t_index] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
							}
						}
					}
				}
			}
		};	// class InverseWarpFunctor

		// Define class InverseWarpGradFunctor with CPUDevice.
		template <typename T>
		struct InverseWarpGradFunctor<CPUDevice, T> {
			void operator()(const CPUDevice& d,
			int batch, int height, int width, int channels,
			const T* in, const T* flow, const T* output_grad,
			T* in_grad, T* flow_grad) {

				for (int i = 0; i < batch * height * width * channels; i++) {
					in_grad[i] = 0;
				}
				for (int n = 0; n < batch; n++) {
					for (int h = 0; h < height; h++) {
						for (int w = 0; w < width; w++) {
							// Get flow value.
							const T f_w = flow[n * height * width * 2 + h * width * 2 + w * 2 + 0];
							const T f_h = flow[n * height * width * 2 + h * width * 2 + w * 2 + 1];
							// Get target coordinate.
							const int f_h0 = floorf(f_h);
							const int f_w0 = floorf(f_w);
							const int f_h1 = f_h0 + 1;
							const int f_w1 = f_w0 + 1;

							const int _h0 = std::min(std::max(h + f_h0, 0), height - 1);
							const int _w0 = std::min(std::max(w + f_w0, 0), width - 1);
							const int _h1 = std::min(std::max(h + f_h1, 0), height - 1);
							const int _w1 = std::min(std::max(w + f_w1, 0), width - 1);

							const T w00 = (f_h1 - f_h) * (f_w1 - f_w);
							const T w01 = (f_h1 - f_h) * (f_w - f_w0);
							const T w10 = (f_h - f_h0) * (f_w1 - f_w);
							const T w11 = (f_h - f_h0) * (f_w - f_w0);

							const int feature_size = height * width * channels;
							const int row_size = width * channels;
							T fw_grad_sum = 0, fh_grad_sum = 0;
							for (int c = 0; c < channels; c++) {
								// Calculate input grad.
								const int t_index = n * feature_size + h * row_size + w * channels + c;
								const T grad = output_grad[t_index];
								in_grad[n * feature_size + _h0 * row_size + _w0 * channels + c] +=
									static_cast<T>(w00 * grad);
								in_grad[n * feature_size + _h0 * row_size + _w1 * channels + c] +=
									static_cast<T>(w01 * grad);
								in_grad[n * feature_size + _h1 * row_size + _w0 * channels + c] +=
									static_cast<T>(w10 * grad);
								in_grad[n * feature_size + _h1 * row_size + _w1 * channels + c] +=
									static_cast<T>(w11 * grad);
								// Calculate flow grad.
								const T v00 = in[n * feature_size + _h0 * row_size + _w0 * channels + c];
								const T v01 = in[n * feature_size + _h0 * row_size + _w1 * channels + c];
								const T v10 = in[n * feature_size + _h1 * row_size + _w0 * channels + c];
								const T v11 = in[n * feature_size + _h1 * row_size + _w1 * channels + c];

								T fw_grad = grad, fh_grad = grad;
								fw_grad *= (f_h1 - f_h) * (v01 - v00) + (f_h - f_h0) * (v11 - v10);
								fh_grad *= (f_w1 - f_w) * (v10 - v00) + (f_w - f_w0) * (v11 - v01);
								fw_grad_sum += fw_grad;
								fh_grad_sum += fh_grad;
							}
							// Save flow grad.
							flow_grad[n * height * width * 2 + h * width * 2 + w * 2 + 0] = fw_grad_sum;
							flow_grad[n * height * width * 2 + h * width * 2 + w * 2 + 1] = fh_grad_sum;
						}
					}
				}

			}
		};	// class InverseWarpGradFunctor

	}	//	namespace functor

	REGISTER_KERNEL_BUILDER(Name("InverseWarp").Device(DEVICE_CPU).TypeConstraint<float>("T"),
		InverseWarpOp<CPUDevice, float>);
	REGISTER_KERNEL_BUILDER(Name("InverseWarpGrad").Device(DEVICE_CPU).TypeConstraint<float>("T"),
		InverseWarpGradOp<CPUDevice, float>);
#if GOOGLE_CUDA
	namespace functor {
		extern template struct InverseWarpFunctor<GPUDevice, float>;
	}
	REGISTER_KERNEL_BUILDER(Name("InverseWarp").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
		InverseWarpOp<GPUDevice, float>);

	namespace functor {
		extern template struct InverseWarpGradFunctor<GPUDevice, float>;
	}
	REGISTER_KERNEL_BUILDER(Name("InverseWarpGrad").Device(DEVICE_GPU).TypeConstraint<float>("T"), 
		InverseWarpGradOp<GPUDevice, float>);
#endif	// GOOGLE_CUDA


}	// namespace tensorflow