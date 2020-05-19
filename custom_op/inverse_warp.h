#ifndef OP_INVERSE_WARP_H_
#define OP_INVERSE_WARP_H_

namespace tensorflow {

	namespace functor {

		template <typename Device, typename T>
		struct InverseWarpFunctor {
			void operator()(const Device& d, 
							int batch, int height, int width, int channels, 
							const T* in, const T* flow, T* out);
		};

		template <typename Device, typename T>
		struct InverseWarpGradFunctor {
			void operator()(const Device& d,
							int batch, int height, int width, int channels,
							const T* in, const T* flow, const T* output_grad,
							T* in_grad, T* flow_grad);
		};

	}	// namespace functor

}	// namespace tensorflow

#endif	// OP_INVERSE_WARP_H_