#include "stdafx.h"

struct cudaDeviceProp * _device_properties;

int _device_number;

BOOL _construct()
{
	_device_properties = NULL;
	_device_number = 0;

	if (cudaGetDeviceCount(&_device_number) != cudaSuccess)
		return FALSE;

	if (_device_number == 0)
		return FALSE;

	_device_properties = (struct cudaDeviceProp *)
		malloc(sizeof(struct cudaDeviceProp) * _device_number);

	for (int i = 0; i < _device_number; i++)
	{
		if (cudaGetDeviceProperties(_device_properties + i, i)
			!= cudaSuccess)
			return FALSE;
	}

	return TRUE;
}


BOOL _destruct()
{
	free(_device_properties);

	return TRUE;
}