#pragma once


#ifndef HF_SIDDON_CLASS_HPP_
#define HF_SIDDON_CLASS_HPP_


#include <vector>
#include <algorithm>
#include "hf_cuda_functions.cuh"
#include "hf_cpp_functions.hpp"
#include "siddon_kernelsMLEM.cuh"
#include <helper_timer.h>

#include "RawData.hpp"


struct siddon_commons
{
	float *d_alpha_min;
	float *d_alpha_max;
	float *d_d12;
	unsigned int *d_mask;
	float *d_rotation_matrix;
	float *d_current_alpha; //will be reset by siddon_fp_begin or siddon_bp_begin;
	float *d_next_alpha; //will be reset by siddon_sort_alpha
	float3 *d_alpha_i;
	float3 *d_alpha_f;
	float3 *d_alpha; //needs reset by siddon_fp_begin or siddon_bp_begin;
	float3 *d_delta_alpha;
	float3 *p1;
	float3 *p2;
	int3 *d_index;
	unsigned int *d_delta_index;
	unsigned int loopN;
	float angle_degree;
	XraySystem system;
	int N_image_pixels;
	int N_object_voxels;
	std::vector <unsigned int> delta_index;

	int check;
	void a1_initiate(CalibrationParam params, Detector det, VoxVolParam vox)
	{
		Planes P_initial; //voxel volume initial plane position
		Planes P_final; //voxel volume final plane position
		Planes P_delta; //voxel size
		Planes XraySource; // position of the x-ray source
		
		//fill in the Planes
		P_initial.fill( -(float)vox.Xdim/2.0f*(float)vox.XVoxSize, -(float)vox.Ydim/2.0f*(float)vox.YVoxSize, -(float)vox.Zdim/2.0f*(float)vox.ZVoxSize);
		P_final.fill(    (float)vox.Xdim/2.0f*(float)vox.XVoxSize,  (float)vox.Ydim/2.0f*(float)vox.YVoxSize,  (float)vox.Zdim/2.0f*(float)vox.ZVoxSize);
		P_delta.fill( vox.XVoxSize, vox.YVoxSize, vox.ZVoxSize );
		XraySource.fill(0.0f, (-params.Rf), 0.0f);
		system.create(params, det, vox, P_initial, P_final, P_delta, XraySource);	
		
		this->N_image_pixels = det.NumAxPixels*det.NumTAxPixels;
		this->N_object_voxels = system.voxparam.Xdim*system.voxparam.Ydim*system.voxparam.Zdim;

		check = 1;
	}

	void a2_allocate()
	{
		if (check == 1)
		{
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_alpha_min, N_image_pixels*sizeof(float) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_alpha_max, N_image_pixels*sizeof(float) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_d12, N_image_pixels*sizeof(float) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_mask, N_image_pixels*sizeof(unsigned int) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_rotation_matrix, 9*sizeof(float) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_current_alpha, N_image_pixels*sizeof(float) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_next_alpha, N_image_pixels*sizeof(float) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_alpha_i, N_image_pixels*sizeof(float3) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_alpha_f, N_image_pixels*sizeof(float3) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_alpha, N_image_pixels*sizeof(float3) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_delta_alpha, N_image_pixels*sizeof(float3) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_delta_index, N_image_pixels*sizeof(unsigned int) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->p1, N_image_pixels*sizeof(float3) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->p2, N_image_pixels*sizeof(float3) ) );
			CUDA_SAFE_CALL( cudaMalloc( (void**)&this->d_index, N_image_pixels*sizeof(int3) ) );
			rotation_matrix3D(d_rotation_matrix, system.CalParam.xangle, system.CalParam.yangle, system.CalParam.zangle);
			delta_index.resize(N_image_pixels, 0);

			check = 2;
		}
		else
		{
			std::cout << "need to initialize first" << std::endl;
		}
	}

	unsigned int a_calc_initial_limits(float angle_degree, dim3 blocks_image, dim3 threads_image)
	{
		this->angle_degree = angle_degree;
		siddon_alpha_limits<<<blocks_image, threads_image>>>(	this->d_alpha_min, this->d_alpha_max, 
																this->d_mask, this->d_d12,
																this->d_alpha_i, this->d_alpha_f, this->d_delta_alpha, 
																this->d_delta_index,
																this->d_rotation_matrix, this->angle_degree, 
																this->system, this->p1, this->p2);

		CUDA_SAFE_CALL( cudaMemcpy( &this->delta_index[0], this->d_delta_index, N_image_pixels*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		this->loopN = (unsigned int)*std::max_element( this->delta_index.begin(), this->delta_index.end() );
		return(loopN);
	}


	void a3_deallocate()
	{
		if (check == 2)
		{
			delta_index.clear();
			CUDA_SAFE_CALL( cudaFree(this->d_alpha_min) );
			CUDA_SAFE_CALL( cudaFree(this->d_alpha_max) );
			CUDA_SAFE_CALL( cudaFree(this->d_d12) );
			CUDA_SAFE_CALL( cudaFree(this->d_mask) );
			CUDA_SAFE_CALL( cudaFree(this->d_rotation_matrix) );
			CUDA_SAFE_CALL( cudaFree(this->d_current_alpha) );
			CUDA_SAFE_CALL( cudaFree(this->d_next_alpha) );
			CUDA_SAFE_CALL( cudaFree(this->d_alpha_i) );
			CUDA_SAFE_CALL( cudaFree(this->d_alpha_f) );
			CUDA_SAFE_CALL( cudaFree(this->d_alpha) );
			CUDA_SAFE_CALL( cudaFree(this->d_delta_alpha) );
			CUDA_SAFE_CALL( cudaFree(this->d_delta_index) );
			CUDA_SAFE_CALL( cudaFree(this->d_index) );
		}
		else
		{
			std::cout << "cannot deallocate, did you allocate first?" << std::endl;
		}
	}
};


class forward
{
public:
	void a1_Finitiate(int N_image_pixels);
	void a2_fp_per_angle(float *d_object, siddon_commons &siddon_var, dim3 blocks_image, dim3 threads_image);

	float *a_Fdevicepointer();

	~forward();

protected:

	float *md_image_temp;
	float *md_image;

	void m_zero_image_temp();
	void m_zero_image();

private:
	int pN_pixels;
	bool pcheck;

};

//================ protected functions ====================================================
void forward::m_zero_image_temp()
{
	if (this->pcheck)
	{
		CUDA_SAFE_CALL( cudaMemset( this->md_image_temp, 0, pN_pixels*sizeof(float) ) );
	}
	else
	{
		std::cout<<"need to initiate first" << std::endl;
	}
}

void forward::m_zero_image()
{
	if (this->pcheck)
	{
		CUDA_SAFE_CALL( cudaMemset( md_image, 0, pN_pixels*sizeof(float) ) );
	}
	else
	{
		std::cout<<"need to initiate first" << std::endl;
	}
}

//================ public functions ===================================================

void forward::a1_Finitiate(int N_image_pixels)
{
	this->pN_pixels = N_image_pixels;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_image_temp, this->pN_pixels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_image, this->pN_pixels*sizeof(float) ) );
	this->pcheck = true;
}

void forward::a2_fp_per_angle(float *d_object, siddon_commons &siddon_var, dim3 blocks_image, dim3 threads_image)
{
	this->m_zero_image_temp();
	this->m_zero_image();

	siddon_begin<<<blocks_image, threads_image>>>(siddon_var.d_alpha_min, siddon_var.d_alpha, siddon_var.d_alpha_i, 
												  siddon_var.d_mask, siddon_var.d_next_alpha);

	for (unsigned int i = 0; i < siddon_var.loopN; i++) // loopN
	{
		this->m_zero_image_temp();

		siddon_sort_alpha<<<blocks_image, threads_image>>>(	siddon_var.d_alpha, siddon_var.d_alpha_i, siddon_var.d_alpha_f, 
															siddon_var.d_delta_alpha, siddon_var.d_current_alpha, siddon_var.d_next_alpha, 
															siddon_var.d_alpha_max, siddon_var.d_alpha_min);

		siddon_fp_calc_ray_path<<<blocks_image, threads_image>>>(	siddon_var.system, siddon_var.p1, siddon_var.p2, siddon_var.d_index,
																	siddon_var.d_current_alpha, siddon_var.d_next_alpha, siddon_var.d_d12, 
																	siddon_var.d_mask, this->md_image_temp, d_object);

		siddon_add <<<blocks_image, threads_image >>>(this->md_image, this->md_image_temp);
	}

	this->m_zero_image_temp();

	siddon_fp_end <<<blocks_image, threads_image >>>(	siddon_var.d_alpha_max, siddon_var.d_d12, siddon_var.d_mask,
														siddon_var.system, siddon_var.p1, siddon_var.p2, siddon_var.d_index,
														siddon_var.d_current_alpha, this->md_image_temp, d_object	);		

	siddon_add <<<blocks_image, threads_image >>>(this->md_image, this->md_image_temp);
}

float *forward::a_Fdevicepointer()
{
	return(this->md_image);
}

forward::~forward()
{
	if (this->pcheck)
	{
		CUDA_SAFE_CALL( cudaFree(this->md_image) );
		CUDA_SAFE_CALL( cudaFree(this->md_image_temp) );
	}
	else
	{
		std::cout << "need to initiate first" << std::endl;
	}

}


class backward
{
public:
	void a1_Binitiate(int N_object_voxels, int N_image_pixels);
	void a2_bp_per_angle(float *d_image, siddon_commons &siddon_var, 
						dim3 blocks_image, dim3 threads_image, 
						dim3 blocks_object, dim3 threads_object,
						dim3 blocks_atomic, dim3 threads_atomic);
	float *a_bpdevicepointer();

	~backward();

protected:
	float *md_object_temp;
	float *md_object;
	int *md_object_index;
	float *md_image_value;
	
	void m_zero_object_temp();
	void m_zero_object();
	int pN_voxels;
	int pN_voxels_pad;

private:
	bool pcheck;

};

//============= protected functions ================================================
void backward::m_zero_object_temp()
{
	if (this->pcheck)
	{
		CUDA_SAFE_CALL( cudaMemset( md_object_temp, 0, this->pN_voxels_pad*sizeof(float) ) );
	}
	else
	{
		std::cout<<"need to initiate first" << std::endl;
	}
}

void backward::m_zero_object()
{
	if (this->pcheck)
	{
		CUDA_SAFE_CALL( cudaMemset( this->md_object, 0, this->pN_voxels_pad*sizeof(float) ) );
	}
	else
	{
		std::cout<<"need to allocate first" << std::endl;
	}
}

//======================== public functions =========================
void backward::a1_Binitiate(int N_object_voxels, int N_image_pixels)
{
	this->pN_voxels = N_object_voxels;
	this->pN_voxels_pad = N_object_voxels + 1;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_object_temp, this->pN_voxels_pad*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_object, this->pN_voxels_pad*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_object_index, N_image_pixels*sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_image_value, N_image_pixels*sizeof(float) ) );

	this->pcheck = true;
}

void backward::a2_bp_per_angle(	float *d_image, siddon_commons &siddon_var, 
								dim3 blocks_image, dim3 threads_image, 
								dim3 blocks_object, dim3 threads_object, 
								dim3 blocks_atomic, dim3 threads_atomic)
{
	this->m_zero_object();
	this->m_zero_object_temp();
	siddon_begin<<<blocks_image, threads_image>>>(	siddon_var.d_alpha_min, siddon_var.d_alpha, siddon_var.d_alpha_i, 
													siddon_var.d_mask, siddon_var.d_next_alpha);

	for (unsigned int i = 0; i < siddon_var.loopN; i++) // loopN
	{
		this->m_zero_object_temp();
		siddon_sort_alpha<<<blocks_image, threads_image>>>(	siddon_var.d_alpha, siddon_var.d_alpha_i, siddon_var.d_alpha_f, 
															siddon_var.d_delta_alpha, siddon_var.d_current_alpha, siddon_var.d_next_alpha, 
															siddon_var.d_alpha_max, siddon_var.d_alpha_min);

		siddon_bp_calc_ray_path<<<blocks_image, threads_image>>>(	siddon_var.system, siddon_var.p1, siddon_var.p2, siddon_var.d_index,
																	this->md_object_index, siddon_var.d_current_alpha, siddon_var.d_next_alpha, 
																	siddon_var.d_d12, siddon_var.d_mask, d_image, this->md_image_value );
		kernel<<<blocks_atomic, threads_atomic>>>(this->md_object_temp, this->md_object_index, this->md_image_value, siddon_var.N_image_pixels);
		siddon_add_bp<<<blocks_object, threads_object>>>(this->md_object, this->md_object_temp);
	}
	
	this->m_zero_object_temp();

	siddon_bp_end<<<blocks_image, threads_image>>>(	siddon_var.d_alpha_max, siddon_var.d_d12, siddon_var.d_mask,
													siddon_var.system, siddon_var.p1, siddon_var.p2, siddon_var.d_index,
													this->md_object_index, siddon_var.d_current_alpha, d_image, this->md_image_value);		
	
	kernel<<<blocks_atomic, threads_atomic>>>(this->md_object_temp, this->md_object_index, this->md_image_value, siddon_var.N_image_pixels);

	siddon_add_bp<<<blocks_object, threads_object>>>(this->md_object, this->md_object_temp);

}

float *backward::a_bpdevicepointer()
{
	return(this->md_object);
}

backward::~backward()
{
	if(pcheck)
	{
		CUDA_SAFE_CALL( cudaFree(this->md_object_temp) );
		CUDA_SAFE_CALL( cudaFree(this->md_object) );
		CUDA_SAFE_CALL( cudaFree(this->md_image_value) );
		CUDA_SAFE_CALL( cudaFree(this->md_object_index) );
	}
	else
	{
		std::cout<<"need to allocate first" << std::endl;
	}

}


class sensitivity : public backward
{
public:
	void a1_Sinitiate(int N_object_voxels, int N_image_pixels);
	void a3S_calculate(siddon_commons &siddon_var, dim3 blocks_image, dim3 threads_image, dim3 blocks_object, dim3 threads_object, dim3 blocks_atomic, dim3 threads_atomic);
	void a3S_add_per_angle(dim3 blocks_object, dim3 threads_object);
	float *a_dsensitivitypointer();
	~sensitivity();
private:
	float *pd_ones;
	float *pd_sensitivity;
};

void sensitivity::a1_Sinitiate(int N_object_voxels, int N_image_pixels)
{
	this->a1_Binitiate(N_object_voxels, N_image_pixels);
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->pd_ones, N_image_pixels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&pd_sensitivity, N_object_voxels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMemset(pd_sensitivity, 0, N_object_voxels*sizeof(float) ) );

}

void sensitivity::a3S_calculate(siddon_commons &siddon_var, dim3 blocks_image, dim3 threads_image, dim3 blocks_object, dim3 threads_object, dim3 blocks_atomic, dim3 threads_atomic)
{
	set_ones<<<blocks_image, threads_image>>>(this->pd_ones);
	std::cout << "calculating sensitivity... " << std::endl;
	for (int n = 0; n < 360; n++)
	{
		siddon_var.a_calc_initial_limits((float)n, blocks_image, threads_image);
		this->a2_bp_per_angle(pd_ones, siddon_var, blocks_image, threads_image, blocks_object, threads_object, blocks_atomic, threads_atomic);
		siddon_add_bp<<<blocks_object, threads_object>>>(pd_sensitivity, md_object);
		std::cout << "sensitivity at " << n << "angles" << std::endl;
	}
	std::cout << "sensitivity calculated, yepee!" << std::endl;
}


float *sensitivity::a_dsensitivitypointer()
{
	return(pd_sensitivity);
}

sensitivity::~sensitivity()
{
	CUDA_SAFE_CALL( cudaFree(this->pd_ones) );
	CUDA_SAFE_CALL( cudaFree(this->pd_sensitivity) );
}



#endif