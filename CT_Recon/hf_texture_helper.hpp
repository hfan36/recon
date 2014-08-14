//Written by Helen Fan
//Uses Danny Ruijters cubic b-spline interpolation code
//it helps me bind texture memory, allocate memory for binding and unallocate and unbind
//so I know what's being created and destroyed.  Just something to use as little memory as possible
//and prevent code from allocating and deallocating over and over (hopefully speed things up a bit)



#ifndef TEXTURE_HELPER_HF_HPP_
#define TEXTURE_HELPER_HF_HPP_


#include <memcpy.cu>
#include <cubicPrefilter3D.cu>
#include <cubicPrefilter2D.cu>
#include <cubicTex2D.cu>
#include <cubicTex3D.cu>

texture<float, 2, cudaReadModeElementType> tex2_float;  //2D texture
texture<float, 3, cudaReadModeElementType> tex3_float;  //3D texture

//aid to bind 2d texture memory
//don't use any of the variables inside, juse use the functions in order (1,2,3)
//1 for allocate space necessary for binding
//2 is where you put the arrays you want inside, 
//I use it to stick different arrays in over and over in a loop so I don't have to 
//allocate and deallocate in every single loop
//3 finally destroy what 1 allocated and unbinds tex2_float 
//probably making a class would be better, but I don't feel like it.

struct texture_helper2D
{
	cudaPitchedPtr m_bsplineCoeffs;
	cudaArray *m_coeffArray;
	cudaExtent m_extent;
	cudaChannelFormatDesc m_desc;
	
	int a1_allocate(uint width, uint height)
	{
		this->m_extent = make_cudaExtent(width*sizeof(float), height, 1);
		this->m_desc = cudaCreateChannelDesc<float>();
		CUDA_SAFE_CALL( cudaMalloc3D(&this->m_bsplineCoeffs, this->m_extent) );
		this->m_coeffArray = 0;

		//tex2_float.normalized = true; //I would not change this, since it's what Danny Ruijters uses in his codes; original code;
		tex2_float.normalized = false;

		//Change address mode here to whatever you want. might not be wise
		tex2_float.addressMode[0] = cudaAddressModeClamp; //used to be Clamp
		tex2_float.addressMode[1] = cudaAddressModeClamp; //used to be Clamp

		//tex2_float.filterMode = cudaFilterModeLinear; //I would not change this, since it's what Danny Ruijters uses in his codes; original code;
		tex2_float.filterMode = cudaFilterModePoint; //temporarily changed for siddon's algorithm
		return 0;
	}


	int a2_input(float *data, uint width, uint height, bool pre_filter, cudaMemcpyKind kind)
	{
		cudaMemcpy3DParms p = {0};
		p.srcPtr = make_cudaPitchedPtr( (void*)data, width*sizeof(float), width, height );
		p.dstPtr = this->m_bsplineCoeffs;
		p.extent = this->m_extent;
		p.kind = kind;
		CUDA_SAFE_CALL( cudaMemcpy3D(&p) );

		if (pre_filter)
		{
			CubicBSplinePrefilter2D( (float*)this->m_bsplineCoeffs.ptr, (uint)this->m_bsplineCoeffs.pitch, width, height);
		}
		CUDA_SAFE_CALL( cudaMallocArray( &this->m_coeffArray, &this->m_desc, width, height ) );
		CUDA_SAFE_CALL( cudaMemcpy2DToArray( this->m_coeffArray, 0, 0, this->m_bsplineCoeffs.ptr, this->m_bsplineCoeffs.pitch, width*sizeof(float), height, cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaBindTextureToArray( tex2_float, this->m_coeffArray, this->m_desc ) );
		return 0;
	}

	int a3_destroy()
	{

		CUDA_SAFE_CALL( cudaFree(m_bsplineCoeffs.ptr) );
		CUDA_SAFE_CALL( cudaFreeArray( m_coeffArray ) );
		CUDA_SAFE_CALL( cudaUnbindTexture( tex2_float ) );
		std::cout << "2D texture cleared! " << std::endl;
		return 0;
	}

};


//aids in binding 3d texture memory
//don't use any of the variables inside
// use functions in order, 1,2,3,4
// same as texture_helper2D, except I added 3_output function just coz
// I want to make sure it's done the way Dannny Ruijters did, probably don't really need it,
// if you just use straight cudaMemcpy to copy the memory out but since I wrote it, so deal with it.
// 4 is to destroy everything I've created in 1.
// probably should've also made a class for it? but I don't feel like it

struct texture_helper3D
{
	cudaPitchedPtr m_bsplineCoeffs;
	cudaArray *m_coeffArray;
	cudaExtent m_extent;
	cudaChannelFormatDesc m_desc;
	
	int a1_allocate(uint dimx, uint dimy, uint dimz)
	{
		this->m_extent = make_cudaExtent(dimx*sizeof(float), dimy, dimz);
		this->m_desc = cudaCreateChannelDesc<float>();
		CUDA_SAFE_CALL( cudaMalloc3D(&this->m_bsplineCoeffs, this->m_extent) );
		
		//tex3_float.normalized = true; //I would not change this, since it's what Danny Ruijters uses in his codes
		tex3_float.normalized = false; //temporary for siddon's algorithm, used to true;

		//Change address mode here to whatever you want. might not be wise
		tex3_float.addressMode[0] = cudaAddressModeBorder; //don't use clamp for forward projection! holy crap! no wrap for forward projection, use border for fp
		tex3_float.addressMode[1] = cudaAddressModeBorder;
		tex3_float.addressMode[2] = cudaAddressModeBorder;

		//tex3_float.filterMode = cudaFilterModeLinear; //I would not change this, since it's what Danny Ruijters uses in his codes
		tex3_float.filterMode = cudaFilterModePoint; //temporary for siddons's algorithm, used to be cudaFilterModeLinear;
		return 0;
	}

	int a2_input(float *data, uint dimx, uint dimy, uint dimz, bool pre_filter, cudaMemcpyKind kind)
	{	
		cudaMemcpy3DParms p = {0};
		p.srcPtr = make_cudaPitchedPtr( (void*)data, dimx*sizeof(float), dimx, dimy );
		p.dstPtr = this->m_bsplineCoeffs;
		p.extent = this->m_extent;
		p.kind = kind;
		CUDA_SAFE_CALL( cudaMemcpy3D(&p) );
		
		if (pre_filter)
		{
			CubicBSplinePrefilter3D( (float*)this->m_bsplineCoeffs.ptr, (uint)this->m_bsplineCoeffs.pitch, dimx, dimy, dimz);
		}

		cudaMemcpy3DParms p1 = {0};
		cudaExtent volumeExtent = make_cudaExtent(dimx, dimy, dimz);
		CUDA_SAFE_CALL( cudaMalloc3DArray(&this->m_coeffArray, &this->m_desc, volumeExtent) );
		p1.srcPtr = this->m_bsplineCoeffs;
		p1.dstArray = this->m_coeffArray;
		p1.extent = volumeExtent;
		p1.kind = cudaMemcpyDeviceToDevice;
		CUDA_SAFE_CALL( cudaMemcpy3D(&p1) );
		CUDA_SAFE_CALL( cudaBindTextureToArray( tex3_float, this->m_coeffArray, this->m_desc) );

	
		return 0;
	}

	//dimx_device, dimy_device and dimz_device doesn't have to be the same size as dimx, dimy and dimz since
	//these values are interpolated from array they can be anysize, all depends on your cuda kernel
	//I dont' think i use it????
	int a_optional_output(float *host, float *device, uint dimx_device, uint dimy_device, uint dimz_device)
	{
		cudaMemcpy3DParms p2 = {0};
		
		p2.srcPtr = make_cudaPitchedPtr( (void*)device, dimx_device*sizeof(float), dimx_device, dimy_device );
		p2.dstPtr = make_cudaPitchedPtr( (void*)host, dimx_device*sizeof(float), dimx_device, dimy_device );
		p2.extent = make_cudaExtent(dimx_device*sizeof(float), dimy_device, dimz_device);
		p2.kind = cudaMemcpyDeviceToHost;
		CUDA_SAFE_CALL( cudaMemcpy3D(&p2) );

		return 0;
	}
	
	int a3_destroy()
	{
		CUDA_SAFE_CALL( cudaFree(this->m_bsplineCoeffs.ptr) );
		CUDA_SAFE_CALL( cudaFreeArray(this->m_coeffArray) );
		CUDA_SAFE_CALL( cudaUnbindTexture( tex3_float) );
		std::cout << "3D texture cleared! " << std::endl;

		return 0;
	}
};


#endif