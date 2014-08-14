/* written by Helen Fan */

#ifndef HF_CUDA_FUNCTIONS_CUH_
#define HF_CUDA_FUNCTIONS_CUH_

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
//#include "CTSystemElements.h"
#include "hf_cuda_kernels.cuh"
#include "CTSystemElements_HF.h"


class timer
{
private:
	cudaEvent_t start, stop;
	void eventCreate();
	void eventDestroy();
public:
	void tic();
	void toc();
	float toc_time();
};

void timer::eventCreate()
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void timer::eventDestroy()
{
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

void timer::tic()
{
	eventCreate();
	cudaEventRecord( start, 0 );
}

void timer::toc()
{
	cudaEventRecord( stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;	
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("tic toc took %3.5f seconds \n", elapsedTime/1000.0f);
	eventDestroy();	
}

float timer::toc_time()
{
	cudaEventRecord( stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;	
	cudaEventElapsedTime(&elapsedTime, start, stop);
	eventDestroy();	
	return(elapsedTime/1000.0f);
}



void rotation_matrix3D(float *d_RotationMatrix, float thetax, float thetay, float thetaz)
{
	float *RotationMatrix = new float [9];

	//rotation matrix defined by von Smekal et al. paper on Geometric misalignment and calibration
	RotationMatrix[0] = cos(thetay)*cos(thetaz)-sin(thetay)*sin(thetax)*sin(thetaz);
	RotationMatrix[1] = -cos(thetax)*sin(thetaz);
	RotationMatrix[2] = -cos(thetaz)*sin(thetay)-cos(thetay)*sin(thetax)*sin(thetaz);
	RotationMatrix[3] = cos(thetaz)*sin(thetay)*sin(thetax)+cos(thetay)*sin(thetaz);
	RotationMatrix[4] = cos(thetax)*cos(thetaz);
	RotationMatrix[5] = cos(thetay)*cos(thetaz)*sin(thetax)-sin(thetay)*sin(thetaz);
	RotationMatrix[6] = cos(thetax)*sin(thetay);
	RotationMatrix[7] = -sin(thetax);
	RotationMatrix[8] = cos(thetay)*cos(thetax);

	//for (unsigned int n = 0; n < 9; n++)
//		printf("RotationMatrix[%d] = %f \n", n, RotationMatrix[n]);

	CUDA_SAFE_CALL( cudaMemcpy(d_RotationMatrix, RotationMatrix, 9*sizeof(float), cudaMemcpyHostToDevice) );

	delete [] RotationMatrix;

}

enum memory_type {host, device};

enum output_direction {first, second};

//this function divides the numerator by the denominator, then put the result back into numerator
template <typename T>
void gpu_divide(T *numerator, T *denominator, int n_elements, dim3 threads, dim3 blocks, memory_type mtype, output_direction direction)
{
	if (mtype == host)
	{
		T *d_numerator, *d_denominator;
		CUDA_SAFE_CALL( cudaMalloc( (void**)&d_numerator, n_elements*sizeof(T) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&d_denominator, n_elements*sizeof(T) ) );
		CUDA_SAFE_CALL( cudaMemcpy(d_numerator, numerator, n_elements*sizeof(T), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy(d_denominator, denominator, n_elements*sizeof(T), cudaMemcpyHostToDevice) );

		if (direction == first)
		{
			divide1<T> <<<blocks, threads>>> (d_numerator, d_denominator);
			CUDA_SAFE_CALL( cudaMemcpy(numerator, d_numerator, n_elements*sizeof(T), cudaMemcpyDeviceToHost) );
		}
		if (direction == second)
		{
			divide2<T> <<<blocks, threads>>> (d_numerator, d_denominator);
			CUDA_SAFE_CALL( cudaMemcpy(denominator, d_denominator, n_elements*sizeof(T), cudaMemcpyDeviceToHost) );
		}
		CUDA_SAFE_CALL( cudaFree(d_numerator) );
		CUDA_SAFE_CALL( cudaFree(d_denominator) );
	}
	else if (mtype == device && direction == first)
	{
		divide1<T><<<blocks, threads>>>(numerator, denominator);
	}
	else if (mtype == device && direction == second)
	{
		divide2<T><<<blocks, threads>>>(numerator, denominator);
	}
	else
	{
		std::cout << "input error in gpu_divide \n" << std::endl;
	}
}

template <typename T>
void gpu_multiply(T *pointer1, T*pointer2, int n_elements, dim3 threads, dim3 blocks, memory_type mtype, output_direction direction)
{
	if (mtype == host)
	{
		T *d_pointer1;
		T *d_pointer2;
		CUDA_SAFE_CALL( cudaMalloc( (void**)&d_pointer1, n_elements*sizeof(T) ) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&d_pointer2, n_elements*sizeof(T) ) );
		CUDA_SAFE_CALL( cudaMemcpy(d_pointer1, pointer1, n_elements*sizeof(T), cudaMemcpyHostToDevice) );
		CUDA_SAFE_CALL( cudaMemcpy(d_pointer2, pointer2, n_elements*sizeof(T), cudaMemcpyHostToDevice) );

		if (direction == first)
		{
			multiply1<T><<<blocks, threads>>>(d_pointer1, d_pointer2);
			CUDA_SAFE_CALL( cudaMemcpy(pointer1, d_pointer1, n_elements*sizeof(T), cudaMemcpyDeviceToHost) );
		}

		if (direction == second)
		{
			multiply2<T><<<blocks, threads>>>(d_pointer1, d_pointer2);
			CUDA_SAFE_CALL( cudaMemcpy(pointer2, d_pointer2, n_elements*sizeof(T), cudaMemcpyDeviceToHost) );
		}
		
		CUDA_SAFE_CALL( cudaFree(d_pointer1) );
		CUDA_SAFE_CALL( cudaFree(d_pointer2) );
	}
	else if (mtype == device && direction == first)
	{
		multiply1<T><<<blocks, threads>>>(pointer1, pointer2);
	}
	else if (mtype == device && direction == second)
	{
		multiply2<T><<<blocks, threads>>>(pointer1, pointer2);
	}
	else
	{
		std::cout << "input error in gpu_multiply \n" << std::endl;
	}
	
}

#endif

////object data is unsigned __int8 if we use binvox
//void prepare_texture3D_uint8(cudaArray *cuArray, unsigned __int8 *objectdata, VoxVolParam &voxparam)
//{
//	//create 3D array
//	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned __int8>();
//	cudaExtent volumeSize = make_cudaExtent(voxparam.Xdim, voxparam.Ydim, voxparam.Zdim);
//	cudaMalloc3DArray(&cuArray, &channelDesc, volumeSize);
//
//	//copy data to 3D array
//	cudaMemcpy3DParms copyParams = {0};
//	copyParams.srcPtr = make_cudaPitchedPtr((void*)objectdata, volumeSize.width*sizeof(unsigned __int8), volumeSize.width, volumeSize.height);
//	copyParams.dstArray = cuArray;
//	copyParams.extent = volumeSize;
//	copyParams.kind = cudaMemcpyHostToDevice;
//	cudaMemcpy3D(&copyParams);
//	
//	// set texture parameters
//	tex3D_uint8.normalized = true;
//	tex3D_uint8.filterMode = cudaFilterModeLinear;
//	tex3D_uint8.addressMode[0] = cudaAddressModeBorder;
//	tex3D_uint8.addressMode[1] = cudaAddressModeBorder;
//	tex3D_uint8.addressMode[2] = cudaAddressModeBorder;
//
//	//bind array to 3D texture
//	cudaBindTextureToArray(tex3D_uint8, cuArray, channelDesc);
//}

//void Set3DTexture(cudaArray *cuArray, float *objdata, VoxVolParam &voxparam, cudaMemcpyKind kind)//passing value because I don't want params to change
//{
//	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
//	cudaExtent extent = make_cudaExtent(voxparam.Xdim, voxparam.Ydim, voxparam.Zdim);
//	cudaMalloc3DArray(&cuArray, &desc, extent, 0);
//
//	cudaMemcpy3DParms params = {0};
//	params.srcPtr = make_cudaPitchedPtr((void*)objdata, voxparam.Xdim*sizeof(float), voxparam.Xdim, voxparam.Ydim);
//	params.dstArray = cuArray;
//	params.extent = extent;
//	params.kind = kind;
//	cudaMemcpy3D(&params);
//	tex3D_float.normalized = true;
//	tex3D_float.filterMode = cudaFilterModeLinear;
//	tex3D_float.addressMode[0] = cudaAddressModeBorder;
//	tex3D_float.addressMode[1] = cudaAddressModeBorder;
//	tex3D_float.addressMode[2] = cudaAddressModeBorder;
//	cudaBindTextureToArray(tex3D_float, cuArray, desc);
//}
//
//void Unbind3DTexture(cudaArray *cuArray)
//{
//	CUDA_SAFE_CALL( cudaUnbindTexture(tex3D_float) );
//	CUDA_SAFE_CALL( cudaFreeArray(cuArray) );
//}
//
//void Set2DTexture(cudaArray *cuArray, float *data2D, int width, int height, cudaMemcpyKind kind)
//{
//
//	//Allocate array and copy image data
//	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//	cudaMallocArray( &cuArray, &desc, width, height);
//	cudaMemcpyToArray(cuArray, 0, 0, data2D, width*height*sizeof(float), kind);
//
//	// set texture parameters
//	tex2.addressMode[0] = cudaAddressModeBorder;
//	tex2.addressMode[1] = cudaAddressModeBorder;
//	tex2.filterMode = cudaFilterModeLinear;
//	tex2.normalized = true;
//
//    // Bind the array to the texture
//	cudaBindTextureToArray( tex2, cuArray, desc);
//
//}

//void Set2DTexture(float *data2D, int width, int height, cudaMemcpyKind kind)
//{
//	cudaArray *cuArray;
//	//Allocate array and copy image data
//	cudaChannelFormatDesc desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
//	cudaMallocArray( &cuArray, &desc, width, height);
//	cudaMemcpyToArray(cuArray, 0, 0, data2D, width*height*sizeof(float), kind);
//
//	// set texture parameters
//	tex2.addressMode[0] = cudaAddressModeBorder;
//	tex2.addressMode[1] = cudaAddressModeBorder;
//	tex2.filterMode = cudaFilterModeLinear;
//	tex2.normalized = true;
//
//    // Bind the array to the texture
//	cudaBindTextureToArray( tex2, cuArray, desc);
//
//}

//void Unbind2DTexture(cudaArray *cuArray)
//{
//	cudaUnbindTexture(tex2);
//	CUDA_SAFE_CALL( cudaFreeArray(cuArray) );
//}


//=========================================================================
//=========================================================================
//=========================================================================
//class cuArray
//{
//public:
//	void start(float &circle, int width, int height, int depth);
//	~cuArray();
//
//private:
//	cudaArray *cuda_array;
//	cudaChannelFormatDesc desc;
//
//	void bindTexture();
//	void allocate(int width, int height, int depth);
//	void copy(float &dataPointer, int width, int height, int depth);
//
//	void unBindTexture();
//	void deallocate();
//};
//
//void cuArray::start(float &circle, int width, int height, int depth)
//{
//	allocate(width, height, depth);
//	copy(circle, width, height, depth);
//	bindTexture();
//}
//
//cuArray::~cuArray()
//{
//	deallocate();
//	unBindTexture();
//}
//
//void cuArray::allocate(int width, int height, int depth)
//{
//	cudaExtent extent;
//	extent.width = width;
//	extent.height = height;
//	extent.depth = depth;	
//	desc = cudaCreateChannelDesc<float>();
//	CUDA_SAFE_CALL( cudaMalloc3DArray(&cuda_array, &desc, extent) );
//}
//
//void cuArray::deallocate()
//{
//	CUDA_SAFE_CALL( cudaFreeArray(cuda_array) );
//}
//
//void cuArray::copy(float &dataPointer, int width, int height, int depth)
//{
//	cudaMemcpy3DParms param = {0};
//	param.srcPtr.ptr = &dataPointer;
//	param.srcPtr.pitch = width * sizeof(float);
//	param.srcPtr.xsize = width;
//	param.srcPtr.ysize = depth;
//	
//	param.dstArray = cuda_array;
//	param.extent.width = width;
//	param.extent.height = height;
//	param.extent.depth = depth;
//	param.kind = cudaMemcpyHostToDevice;
//	CUDA_SAFE_CALL( cudaMemcpy3D(&param) );
//}
//
//void cuArray::bindTexture()
//{
//	CUDA_SAFE_CALL( cudaBindTextureToArray(tex, cuda_array, desc) );
//	tex.normalized = true;
//	tex.filterMode = cudaFilterModeLinear;
//	tex.addressMode[0] = cudaAddressModeBorder;
//	tex.addressMode[1] = cudaAddressModeBorder;
//	tex.addressMode[2] = cudaAddressModeBorder;
//
//}
//
//void cuArray::unBindTexture()
//{
//	CUDA_SAFE_CALL( cudaUnbindTexture(tex) );
//}


//__global__ void calc_temp(float *tempX, float *tempY, float *tempZ, float *temp, 
//						  float *dev_xplane, float *dev_detY, float *dev_detZ, 
//						  int width, float voxsize, float xdet, float xsource)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//		
//	tempX[x] = __fdividef(dev_xplane[x], (float)width*voxsize);
//	tempY[x] = __fdividef(dev_detY[x], (xdet-xsource));
//	tempZ[x] = __fdividef(dev_detZ[x], (xdet-xsource));
//	temp[x] = xsource - dev_xplane[x];
//}
//
//
//__global__ void project3D_temp(float *dev_out3D, float *tempX, float *tempY, 
//						  float *tempZ, float *temp, int width, 
//						  int depth, int height, float voxsize, 
//						  float sin_theta, float cos_theta)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int z = threadIdx.z + blockIdx.z * blockDim.z;
//	int offset = x + y * blockDim.x * gridDim.x +
//				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	float normY = __fdividef(tempY[y] * temp[x], (float)depth*voxsize);
//	float normZ = __fdividef(tempZ[z] * temp[x], (float)height*voxsize);
//	
//	float xplaneR = tempX[x]*cos_theta - normY*sin_theta + 0.5f;
//	float yplaneR = tempX[x]*sin_theta + normY*cos_theta + 0.5f;
//	float zplaneR = normZ + 0.5f;
//	
//	dev_out3D[offset] = tex3D(tex, xplaneR, yplaneR, zplaneR);
//}


//void pre_rotate_volume_to_texture(cudaArray *cuArray, float *objectdata, VoxVolParam &voxparam, float thetax, float thetay, float thetaz)
//{
//	Set3DTexture(cuArray, objectdata, voxparam, cudaMemcpyHostToDevice);
//	float *d_rotationmatrix;
//	CUDA_SAFE_CALL( cudaMalloc((void**)&d_rotationmatrix, 9*sizeof(float)) );
//	rotation_matrix3D(d_rotationmatrix, thetax, thetay, thetaz);
//	
//	float *d_rotatedobject;
//	CUDA_SAFE_CALL( cudaMalloc((void**)&d_rotatedobject, voxparam.Xdim*voxparam.Ydim*voxparam.Zdim*sizeof(float)) );
//	dim3 threads(8,8,8);
//	dim3 blocks(voxparam.Xdim/8, voxparam.Ydim/8, voxparam.Zdim/8);
//	rotate_volume_3D<<< blocks, threads >>>(d_rotatedobject, d_rotationmatrix);
//
//	//------------------------------------------------------------------------------
//	//rebind texture memory
//	Set3DTexture(cuArray, d_rotatedobject, voxparam, cudaMemcpyDeviceToDevice);
//
//	//free memory
//	CUDA_SAFE_CALL( cudaFree(d_rotatedobject) );
//	CUDA_SAFE_CALL( cudaFree(d_rotationmatrix) );
//}


//#define CUDA_SAFE_CALL(call) do 																				\
//	{																											\
//		int err = call;																							\
//		if(cudaSuccess != err)																					\
//		{																										\
//			fprintf(stderr, "Cuda driver error %04Xh in file \"%s\" in line %i.\n", err, __FILE__, __LINE__);	\
//			fprintf(stderr, "CUDA Error message: %s. \n", cudaGetErrorString( (cudaError_t) err) );				\
//			exit(EXIT_FAILURE);																					\
//		}																										\
//	} while(0)