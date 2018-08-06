#include "stdafx.h"
#include "gmat.h"

__global__ void _cuda_c_identity(gmatc_e * res, int len)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < len) {
		res[idx + idx * len].real = 1.0f;
		res[idx + idx * len].imag = 0.0f;
	}
}

GMAT_API gmat_error_t gmatc_create_identity(gmatc * mat, size_t size, int device)
{
	CHECK_NULL(mat);
	CHECK_CSET(device);

	gmat_error_t rtn;
	rtn = gmatc_create(mat, size, size, device);
	CHECK_RTN(rtn, {});

	THR_GRID(device, size);

	CHECK_CUDA(cudaMemset(mat->data, 0, size * size * sizeof(gmatc_e)), gmatc_destroy(mat));

	_cuda_c_identity <<< grids, threads >>> (mat->data, size);

	CHECK_CUDA(cudaGetLastError(), gmatc_destroy(mat));

	return gmat_success;
}

__device__ gmatc_e _cuda_gmatc_e_add(const gmatc_e l, const gmatc_e r) {
	return { l.real + r.real, l.imag + r.imag };
}

__device__ gmatc_e _cuda_gmatc_e_sub(const gmatc_e l, const gmatc_e r) {
	return { l.real - r.real, l.imag - r.imag };
}

__device__ gmatc_e _cuda_gmatc_e_mul(const gmatc_e l, const gmatc_e r) {
	return { (l.real * r.real) - (l.imag * r.imag), (l.real * r.imag) + (l.imag * r.real) };
}

__device__ gmatc_e _cuda_gmatc_e_conj(const gmatc_e l) {
	return { l.real, -l.imag };
}

__device__ gmatc_e _cuda_gmatc_e_div(const gmatc_e l, const gmatc_e r) {
	gmatc_e rtn = _cuda_gmatc_e_mul(l, _cuda_gmatc_e_conj(r));
	float div = r.real * r.real + r.imag * r.imag;
	rtn.real /= div;
	rtn.imag /= div;
	return rtn;
}

#define CUDA_1DIM(name, op) \
__global__ void name(gmatc_e * res, gmatc_e * arg0, gmatc_e * arg1, int len) \
{ \
	int idx = threadIdx.x + blockIdx.x * blockDim.x; \
	if(idx < len) { \
		res[idx] = op(arg0[idx], arg1[idx]); \
	} \
}

#define CUDA_1DIM_FLOAT(name, op) \
__global__ void name(gmatc_e * res, gmatc_e * arg0, gmatc_e arg1, int len) \
{ \
	int idx = threadIdx.x + blockIdx.x * blockDim.x; \
	if (idx < len) \
		res[idx] = op(arg0[idx], arg1); \
}

#define CUDA_1DIM_FLOAT_R(name, op) \
__global__ void name(gmatc_e * res, gmatc_e arg0, gmatc_e * arg1, int len) \
{ \
	int idx = threadIdx.x + blockIdx.x * blockDim.x; \
	if (idx < len) \
		res[idx] = op(arg0, arg1[idx]); \
}

CUDA_1DIM			(_cuda_c_add,			_cuda_gmatc_e_add);
CUDA_1DIM_FLOAT		(_cuda_c_add_gmatc_e,	_cuda_gmatc_e_add);
CUDA_1DIM			(_cuda_c_sub,			_cuda_gmatc_e_sub);
CUDA_1DIM_FLOAT		(_cuda_c_sub_gmatc_e,	_cuda_gmatc_e_sub);
CUDA_1DIM_FLOAT_R	(_cuda_c_sub_gmatc_e_r,	_cuda_gmatc_e_sub);
CUDA_1DIM			(_cuda_c_mul,			_cuda_gmatc_e_mul);
CUDA_1DIM_FLOAT		(_cuda_c_mul_gmatc_e,	_cuda_gmatc_e_mul);
CUDA_1DIM			(_cuda_c_div,			_cuda_gmatc_e_div);
CUDA_1DIM_FLOAT		(_cuda_c_div_gmatc_e,	_cuda_gmatc_e_div);
CUDA_1DIM_FLOAT_R	(_cuda_c_div_gmatc_e_r,	_cuda_gmatc_e_div);

__global__ void _cuda_c_product_mul(gmatc_e * buf, gmatc_e * arg0, gmatc_e * arg1, int l, int m, int n, int len)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < len)
	{
		int _idx = idx;
		int _n = idx % n;
		idx /= n;
		int _m = idx % m;
		idx /= m;
		int _l = idx;
		buf[_idx] = _cuda_gmatc_e_mul(arg0[_l * n + _n], arg1[_n * m + _m]);
	}
}

__global__ void _cuda_c_transpose(gmatc_e * res, gmatc_e * arg0, int arg0_height, int arg0_width)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < arg0_height * arg0_width)
	{
		int h = idx / arg0_width;
		int w = idx % arg0_width;
		int new_idx = w * arg0_height + h;
		res[new_idx] = arg0[idx];
	}
}

__global__ void _cuda_c_sum(gmatc_e * res, gmatc_e * buf, int bundle_len, int len)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < len)
	{
		int _i = idx / bundle_len;
		int _j = idx % bundle_len;
		for (
			int i = bundle_len / 2,
			j = bundle_len % 2;
			i > 0;
			j = i % 2, i /= 2
			)
		{
			if (j && i * 2 == _j)
				buf[idx - i] = _cuda_gmatc_e_add(buf[idx - i], buf[idx]);
			__syncthreads();
			if (_j >= i && _j < i * 2)
				buf[idx - i] = _cuda_gmatc_e_add(buf[idx - i], buf[idx]);
			__syncthreads();
		}
		if (!_j)
			res[_i] = buf[idx];
	}
}

__global__ void _cuda_c_conjugate(gmatc_e * res, gmatc_e * mat, int len)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < len)
		res[idx] = _cuda_gmatc_e_conj(mat[idx]);
}

template <class TArg0, class TArg1>
inline cudaError_t _cuda_c_1_invoke(
	gmatc_e * res, TArg0 arg0, TArg1 arg1, size_t length, int device,
	void(*_cuda_c_func)(gmatc_e *, TArg0, TArg1, int)
)
{
	cudaSetDevice(device);

	THR_GRID(device, length);
	_cuda_c_func <<< grids, threads >>> (res, arg0, arg1, length);

	return cudaGetLastError();
}

#define CUDA_1DIM_INVOKER(invoker, name) \
GMAT_API gmat_error_t invoker(gmatc * res, const gmatc * mat0, const gmatc * mat1) \
{ \
	CHECK_NULL(res); \
	CHECK_NULL(mat0); \
	CHECK_NULL(mat1); \
 \
	CHECK_SIZE(mat0, mat1); \
	CHECK_SIZE(mat0, res); \
 \
	CHECK_DEVC(mat0, mat1); \
	CHECK_DEVC(mat0, res); \
 \
	CHECK_CUDA(_cuda_c_1_invoke( \
		res->data, mat0->data, mat1->data, \
		mat0->frame.height * mat0->frame.width, \
		mat0->frame.device, name \
	), {}); \
 \
	return gmat_success; \
}

#define CUDA_1DIM_INVOKER_FLOAT(invoker, name) \
GMAT_API gmat_error_t invoker(gmatc * res, const gmatc * mat0, const gmatc_e val1) \
{ \
	CHECK_NULL(res); \
	CHECK_NULL(mat0); \
 \
	CHECK_SIZE(mat0, res); \
	CHECK_DEVC(mat0, res); \
 \
	CHECK_CUDA(_cuda_c_1_invoke( \
		res->data, mat0->data, val1, \
		mat0->frame.height * mat0->frame.width, \
		mat0->frame.device, name \
	), {}); \
 \
	return gmat_success; \
}

#define CUDA_1DIM_INVOKER_FLOAT_R(invoker, name) \
GMAT_API gmat_error_t invoker(gmatc * res, const gmatc_e val0, const gmatc * mat1) \
{ \
	CHECK_NULL(res); \
	CHECK_NULL(mat1); \
 \
	CHECK_SIZE(mat1, res); \
	CHECK_DEVC(mat1, res); \
 \
	CHECK_CUDA(_cuda_c_1_invoke( \
		res->data, val0, mat1->data, \
		mat1->frame.height * mat1->frame.width, \
		mat1->frame.device, name \
	), {}); \
 \
	return gmat_success; \
}

CUDA_1DIM_INVOKER			(gmatc_add,				_cuda_c_add);
CUDA_1DIM_INVOKER_FLOAT		(gmatc_add_gmatc_e,		_cuda_c_add_gmatc_e);
CUDA_1DIM_INVOKER			(gmatc_sub,				_cuda_c_sub);
CUDA_1DIM_INVOKER_FLOAT		(gmatc_sub_gmatc_e,		_cuda_c_sub_gmatc_e);
CUDA_1DIM_INVOKER_FLOAT_R	(gmatc_sub_gmatc_e_r,	_cuda_c_sub_gmatc_e_r);
CUDA_1DIM_INVOKER			(gmatc_mul,				_cuda_c_mul);
CUDA_1DIM_INVOKER_FLOAT		(gmatc_mul_gmatc_e,		_cuda_c_mul_gmatc_e);
CUDA_1DIM_INVOKER			(gmatc_div,				_cuda_c_div);
CUDA_1DIM_INVOKER_FLOAT		(gmatc_div_gmatc_e,		_cuda_c_div_gmatc_e);
CUDA_1DIM_INVOKER_FLOAT_R	(gmatc_div_gmatc_e_r,	_cuda_c_div_gmatc_e_r);

GMAT_API gmat_error_t gmatc_product(gmatc * res, const gmatc * mat0, const gmatc * mat1)
{
	CHECK_NULL(res);
	CHECK_NULL(mat0);
	CHECK_NULL(mat1);

	if (res->frame.height != mat0->frame.height || res->frame.width != mat1->frame.width)
		return gmat_invalid_argument;

	CHECK_DEVC(res, mat0);
	CHECK_DEVC(res, mat1);

	int l = res->frame.height;
	int m = res->frame.width;
	int n = mat0->frame.width;
	
	int device = res->frame.device;
	int len = l * m * n;

	CHECK_CSET(device);

	gmatc_e * buf;
	CHECK_CUDA(cudaMalloc(&buf, sizeof(gmatc_e) * len), {});

	THR_GRID(device, len);

	_cuda_c_product_mul <<< grids, threads >>>
		(buf, mat0->data, mat1->data, l, m, n, len);

	_cuda_c_sum <<< grids, threads >>>
		(res->data, buf, n, len);

	CHECK_CUDA(cudaGetLastError(), cudaFree(buf));

	return gmat_success;
}

GMAT_API gmat_error_t gmatc_transpose(gmatc * res, const gmatc * mat)
{
	CHECK_NULL(res);
	CHECK_NULL(mat);
	
	if (res->frame.height != mat->frame.width || res->frame.width != mat->frame.height)
		return gmat_invalid_argument;

	CHECK_DEVC(mat, res);
	
	int device = mat->frame.device;
	int len = mat->frame.height * mat->frame.width;
	THR_GRID(device, len);

	CHECK_CSET(device);

	_cuda_c_transpose <<< grids, threads >>>
		(res->data, mat->data, mat->frame.height, mat->frame.width);

	CHECK_CUDA(cudaGetLastError(), {});
	return gmat_success;
}

GMAT_API gmat_error_t gmatc_conjugate(gmatc * res, const gmatc * mat)
{
	CHECK_NULL(res);
	CHECK_NULL(mat);
	CHECK_SIZE(res, mat);
	CHECK_DEVC(res, mat);

	int device = mat->frame.device;
	int len = mat->frame.height * mat->frame.width;
	THR_GRID(device, len);

	CHECK_CSET(device);

	_cuda_c_conjugate <<< grids, threads >>> (res->data, mat->data, len);

	CHECK_CUDA(cudaGetLastError(), {});
	return gmat_success;
	
}