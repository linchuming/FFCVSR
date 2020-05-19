#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "inverse_warp.h"
#include <stdio.h>

namespace tensorflow {
	typedef Eigen::GpuDevice GPUDevice;
	namespace functor {
		template <typename T>
		__global__ void InverseWarpNHWC(const int nthreads,
										const int batch, const int height,
										const int width, const int channels,
										const T* in, const T* flow,
										T* out) {
			CUDA_1D_KERNEL_LOOP(index, nthreads) {
				const int c = index % channels;
				const int w = (index / channels) % width;
				const int h = (index / channels / width) % height;
				const int n = index / channels / width / height;
				// Get flow value.
				const T f_w = ldg(flow + n * height * width * 2 + h * width * 2 + w * 2 + 0);
				const T f_h = ldg(flow + n * height * width * 2 + h * width * 2 + w * 2 + 1);
				// Get target coordinate.
				const int f_h0 = floorf(f_h);
				const int f_w0 = floorf(f_w);
				const int f_h1 = f_h0 + 1;
				const int f_w1 = f_w0 + 1;
			
				const int _h0 = tf_min(tf_max(h + f_h0, 0), height - 1);
				const int _w0 = tf_min(tf_max(w + f_w0, 0), width - 1);
				const int _h1 = tf_min(tf_max(h + f_h1, 0), height - 1);
				const int _w1 = tf_min(tf_max(w + f_w1, 0), width - 1);

				const T w00 = (f_h1 - f_h) * (f_w1 - f_w);
				const T w01 = (f_h1 - f_h) * (f_w - f_w0);
				const T w10 = (f_h - f_h0) * (f_w1 - f_w);
				const T w11 = (f_h - f_h0) * (f_w - f_w0);
			
				const int feature_size = height * width * channels;
				const int row_size = width * channels;
				
				// Calculate interpolation value.
				const T v00 = ldg(in + n * feature_size + _h0 * row_size + _w0 * channels + c);
				const T v01 = ldg(in + n * feature_size + _h0 * row_size + _w1 * channels + c);
				const T v10 = ldg(in + n * feature_size + _h1 * row_size + _w0 * channels + c);
				const T v11 = ldg(in + n * feature_size + _h1 * row_size + _w1 * channels + c);

				out[index] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
				
			}
		}

		template <typename T>
		__global__ void InverseWarpGradNHWC(const int nthreads,
											const int batch, const int height,
											const int width, const int channels,
											const T* in, const T* flow, const T* output_grad,
											T* in_grad, T* flow_grad) {
			CUDA_1D_KERNEL_LOOP(index, nthreads) {
				const int c = index % channels;
				const int w = (index / channels) % width;
				const int h = (index / channels / width) % height;
				const int n = index / channels / width / height;
				// Get flow value.
				const T f_w = ldg(flow + n * height * width * 2 + h * width * 2 + w * 2 + 0);
				const T f_h = ldg(flow + n * height * width * 2 + h * width * 2 + w * 2 + 1);
				// Get target coordinate.
				const int f_h0 = floorf(f_h);
				const int f_w0 = floorf(f_w);
				const int f_h1 = f_h0 + 1;
				const int f_w1 = f_w0 + 1;

				const int _h0 = tf_min(tf_max(h + f_h0, 0), height - 1);
				const int _w0 = tf_min(tf_max(w + f_w0, 0), width - 1);
				const int _h1 = tf_min(tf_max(h + f_h1, 0), height - 1);
				const int _w1 = tf_min(tf_max(w + f_w1, 0), width - 1);

				const T w00 = (f_h1 - f_h) * (f_w1 - f_w);
				const T w01 = (f_h1 - f_h) * (f_w - f_w0);
				const T w10 = (f_h - f_h0) * (f_w1 - f_w);
				const T w11 = (f_h - f_h0) * (f_w - f_w0);

				const int feature_size = height * width * channels;
				const int row_size = width * channels;
				
				
				// Calculate input grad.
				const T grad = ldg(output_grad + index);
				CudaAtomicAdd(in_grad + n * feature_size + _h0 * row_size + _w0 * channels + c,
					static_cast<T>(w00 * grad));
				CudaAtomicAdd(in_grad + n * feature_size + _h0 * row_size + _w1 * channels + c,
					static_cast<T>(w01 * grad));
				CudaAtomicAdd(in_grad + n * feature_size + _h1 * row_size + _w0 * channels + c,
					static_cast<T>(w10 * grad));
				CudaAtomicAdd(in_grad + n * feature_size + _h1 * row_size + _w1 * channels + c,
					static_cast<T>(w11 * grad));
				// Calculate flow grad.
				const T v00 = ldg(in + n * feature_size + _h0 * row_size + _w0 * channels + c);
				const T v01 = ldg(in + n * feature_size + _h0 * row_size + _w1 * channels + c);
				const T v10 = ldg(in + n * feature_size + _h1 * row_size + _w0 * channels + c);
				const T v11 = ldg(in + n * feature_size + _h1 * row_size + _w1 * channels + c);

				T fw_grad = grad, fh_grad = grad;
				fw_grad *= (f_h1 - f_h) * (v01 - v00) + (f_h - f_h0) * (v11 - v10);
				fh_grad *= (f_w1 - f_w) * (v10 - v00) + (f_w - f_w0) * (v11 - v01);
				
				// Add flow grad.
				CudaAtomicAdd(flow_grad + n * height * width * 2 + h * width * 2 + w * 2 + 0,
					fw_grad);
				CudaAtomicAdd(flow_grad + n * height * width * 2 + h * width * 2 + w * 2 + 1,
					fh_grad);
			}
		}

		template <typename T>													
		struct InverseWarpFunctor<Eigen::GpuDevice, T> {
			void operator()(const Eigen::GpuDevice& d,
			int batch, int height, int width, int channels,
			const T* in, const T* flow, T* out) {
				// Launch the CUDA kernel.
				const int size = batch * height * width * channels;
				CudaLaunchConfig config = GetCudaLaunchConfig(size, d);
				InverseWarpNHWC<T>
					<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
					config.virtual_thread_count, 
					batch, height, width, channels,
					in, flow, out);
			}
		};

		template <typename T>
		struct InverseWarpGradFunctor<Eigen::GpuDevice, T> {
			void operator()(const Eigen::GpuDevice& d,
			int batch, int height, int width, int channels,
			const T* in, const T* flow, const T* output_grad,
			T* in_grad, T* flow_grad) {

				CudaLaunchConfig config;
				int size;
				// Set flow_grad to 0.
				size = batch * height * width * 2;
				config = GetCudaLaunchConfig(size, d);
				SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
					config.virtual_thread_count, flow_grad);
				// Set in_grad to 0.
				size = batch * height * width * channels;
				config = GetCudaLaunchConfig(size, d);
				SetZero<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
					config.virtual_thread_count, in_grad);
				// Launch the CUDA kernel.
				InverseWarpGradNHWC<T>
					<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
					config.virtual_thread_count,
					batch, height, width, channels,
					in, flow, output_grad,
					in_grad, flow_grad);
			}
		};
		// Specify the kernels
		template struct InverseWarpFunctor<GPUDevice, float>;
		template struct InverseWarpGradFunctor<GPUDevice, float>;
	}	// namespace functor

}	// namespace tensorflow

#endif	// GOOGLE_CUDA