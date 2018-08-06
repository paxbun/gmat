#include "stdafx.h"
#include "gmat.h"

__global__ void _cuda_identity(float * res, int len)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < len)
		res[idx + idx * len] = 1.0f;
}

GMAT_API gmat_error_t gmat_create_identity(gmat * mat, size_t size, int device)
{
	CHECK_NULL(mat);
	CHECK_CSET(device);

	gmat_error_t rtn;
	rtn = gmat_create(mat, size, size, device);
	CHECK_RTN(rtn, {});

	THR_GRID(device, size);

	CHECK_CUDA(cudaMemset(mat->data, 0, size * size * sizeof(float)), gmat_destroy(mat));

	_cuda_identity <<< grids, threads >>> (mat->data, size);

	CHECK_CUDA(cudaGetLastError(), gmat_destroy(mat));

	return gmat_success;
}

#define OP_ADD(arg0, arg1) ((arg0)+(arg1))
#define OP_SUB(arg0, arg1) ((arg0)-(arg1))
#define OP_MUL(arg0, arg1) ((arg0)*(arg1))
#define OP_DIV(arg0, arg1) ((arg0)/(arg1))
#define OP_MAX(arg0, arg1) fmaxf((arg0), (arg1));
#define OP_MIN(arg0, arg1) fminf((arg0), (arg1));

#define CUDA_1DIM(name, op) \
__global__ void name(float * res, float * arg0, float * arg1, int len) \
{ \
	int idx = threadIdx.x + blockIdx.x * blockDim.x; \
	if(idx < len) \
		res[idx] = op(arg0[idx], arg1[idx]); \
}

#define CUDA_1DIM_FLOAT(name, op) \
__global__ void name(float * res, float * arg0, float arg1, int len) \
{ \
	int idx = threadIdx.x + blockIdx.x * blockDim.x; \
	if (idx < len) \
		res[idx] = op(arg0[idx], arg1); \
}

#define CUDA_1DIM_FLOAT_R(name, op) \
__global__ void name(float * res, float arg0, float * arg1, int len) \
{ \
	int idx = threadIdx.x + blockIdx.x * blockDim.x; \
	if (idx < len) \
		res[idx] = op(arg0, arg1[idx]); \
}

CUDA_1DIM			(_cuda_add,			OP_ADD);
CUDA_1DIM_FLOAT		(_cuda_add_float,	OP_ADD);
CUDA_1DIM			(_cuda_sub,			OP_SUB);
CUDA_1DIM_FLOAT		(_cuda_sub_float,	OP_SUB);
CUDA_1DIM_FLOAT_R	(_cuda_sub_float_r,	OP_SUB);
CUDA_1DIM			(_cuda_mul,			OP_MUL);
CUDA_1DIM_FLOAT		(_cuda_mul_float,	OP_MUL);
CUDA_1DIM			(_cuda_div,			OP_DIV);
CUDA_1DIM_FLOAT		(_cuda_div_float,	OP_DIV);
CUDA_1DIM_FLOAT_R	(_cuda_div_float_r,	OP_DIV);
CUDA_1DIM			(_cuda_max,			OP_MAX);
CUDA_1DIM_FLOAT		(_cuda_max_float,	OP_MAX);
CUDA_1DIM			(_cuda_min,			OP_MIN);
CUDA_1DIM_FLOAT		(_cuda_min_float,	OP_MIN);

__global__ void _cuda_product_mul(float * buf, float * arg0, float * arg1, int l, int m, int n, int len)
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
		buf[_idx] = arg0[_l * n + _n] * arg1[_n * m + _m];
	}
}

__global__ void _cuda_transpose(float * res, float * arg0, int arg0_height, int arg0_width)
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

__global__ void _cuda_sum(float * res, float * buf, int bundle_len, int len)
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
				buf[idx - i] += buf[idx];
			__syncthreads();
			if (_j >= i && _j < i * 2)
				buf[idx - i] += buf[idx];
			__syncthreads();
		}
		if (!_j)
			res[_i] = buf[idx];
	}
}

template <class TArg0, class TArg1>
inline cudaError_t _cuda_1_invoke(
	float * res, TArg0 arg0, TArg1 arg1, size_t length, int device,
	void(*_cuda_func)(float *, TArg0, TArg1, int)
)
{
	cudaSetDevice(device);

	THR_GRID(device, length);
	_cuda_func <<< grids, threads >>> (res, arg0, arg1, length);

	return cudaGetLastError();
}

#define CUDA_1DIM_INVOKER(invoker, name) \
GMAT_API gmat_error_t invoker(gmat * res, const gmat * mat0, const gmat * mat1) \
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
	CHECK_CUDA(_cuda_1_invoke( \
		res->data, mat0->data, mat1->data, \
		mat0->frame.height * mat0->frame.width, \
		mat0->frame.device, name \
	), {}); \
 \
	return gmat_success; \
}

#define CUDA_1DIM_INVOKER_FLOAT(invoker, name) \
GMAT_API gmat_error_t invoker(gmat * res, const gmat * mat0, const float val1) \
{ \
	CHECK_NULL(res); \
	CHECK_NULL(mat0); \
 \
	CHECK_SIZE(mat0, res); \
	CHECK_DEVC(mat0, res); \
 \
	CHECK_CUDA(_cuda_1_invoke( \
		res->data, mat0->data, val1, \
		mat0->frame.height * mat0->frame.width, \
		mat0->frame.device, name \
	), {}); \
 \
	return gmat_success; \
}

#define CUDA_1DIM_INVOKER_FLOAT_R(invoker, name) \
GMAT_API gmat_error_t invoker(gmat * res, const float val0, const gmat * mat1) \
{ \
	CHECK_NULL(res); \
	CHECK_NULL(mat1); \
 \
	CHECK_SIZE(mat1, res); \
	CHECK_DEVC(mat1, res); \
 \
	CHECK_CUDA(_cuda_1_invoke( \
		res->data, val0, mat1->data, \
		mat1->frame.height * mat1->frame.width, \
		mat1->frame.device, name \
	), {}); \
 \
	return gmat_success; \
}

CUDA_1DIM_INVOKER			(gmat_add,			_cuda_add);
CUDA_1DIM_INVOKER_FLOAT		(gmat_add_float,	_cuda_add_float);
CUDA_1DIM_INVOKER			(gmat_sub,			_cuda_sub);
CUDA_1DIM_INVOKER_FLOAT		(gmat_sub_float,	_cuda_sub_float);
CUDA_1DIM_INVOKER_FLOAT_R	(gmat_sub_float_r,	_cuda_sub_float_r);
CUDA_1DIM_INVOKER			(gmat_mul,			_cuda_mul);
CUDA_1DIM_INVOKER_FLOAT		(gmat_mul_float,	_cuda_mul_float);
CUDA_1DIM_INVOKER			(gmat_div,			_cuda_div);
CUDA_1DIM_INVOKER_FLOAT		(gmat_div_float,	_cuda_div_float);
CUDA_1DIM_INVOKER_FLOAT_R	(gmat_div_float_r,	_cuda_div_float_r);
CUDA_1DIM_INVOKER			(gmat_max,			_cuda_max);
CUDA_1DIM_INVOKER_FLOAT		(gmat_max_float,	_cuda_max_float);
CUDA_1DIM_INVOKER			(gmat_min,			_cuda_min);
CUDA_1DIM_INVOKER_FLOAT		(gmat_min_float,	_cuda_min_float);

GMAT_API gmat_error_t gmat_product(gmat * res, const gmat * mat0, const gmat * mat1)
{
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

	float * buf;
	CHECK_CUDA(cudaMalloc(&buf, sizeof(float) * len), {});

	THR_GRID(device, len);

	_cuda_product_mul <<< grids, threads >>>
		(buf, mat0->data, mat1->data, l, m, n, len);

	_cuda_sum <<< grids, threads >>>
		(res->data, buf, n, len);

	CHECK_CUDA(cudaGetLastError(), cudaFree(buf));

	return gmat_success;
}

GMAT_API gmat_error_t gmat_transpose(gmat * res, const gmat * mat)
{
	if (res->frame.height != mat->frame.width || res->frame.width != mat->frame.height)
		return gmat_invalid_argument;

	CHECK_DEVC(mat, res);
	
	int device = mat->frame.device;
	int len = mat->frame.height * mat->frame.width;
	THR_GRID(device, len);

	CHECK_CSET(device);

	_cuda_transpose <<< grids, threads >>>
		(res->data, mat->data, mat->frame.height, mat->frame.width);

	CHECK_CUDA(cudaGetLastError(), {});
	return gmat_success;
}
