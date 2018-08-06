#pragma once

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

EXTERN struct cudaDeviceProp * _device_properties;
EXTERN int _device_number;
EXTERN BOOL _construct();
EXTERN BOOL _destruct();

#define CHECK_NULL(arg)			if((arg) == NULL)						{ return gmat_invalid_argument; }
#define CHECK_RTN(arg, action)	if((arg) != gmat_success)				{ action; return rtn; }
#define CHECK_CUDA(arg, action)	if((arg) != cudaSuccess)				{ action; return gmat_cuda_error; }
#define CHECK_CSET(arg)			if(cudaSetDevice(arg) != cudaSuccess)	{ return gmat_cuda_error; }
#define CHECK_SIZE(arg0, arg1)	if((arg0)->frame.height != (arg1)->frame.height \
	|| (arg0)->frame.width != (arg1)->frame.width) \
	{ return gmat_invalid_argument; }
#define CHECK_DEVC(arg0, arg1) if((arg0)->frame.device != (arg1)->frame.device) { return gmat_invalid_argument; }

#define THR_GRID(device_idx, len)\
int threads = _device_properties[device_idx].maxThreadsPerBlock;\
int grids = (len) / threads + (int)(bool)((len) % threads);