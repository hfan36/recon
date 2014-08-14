#pragma once

#ifndef SIDDON_KERNELSMLEM_CUH
#define SIDDON_KERNELSMLEM_CUH



#include "cuda.h"
#include "cuda_runtime.h"
#include "CTSystemElements_HF.h"
#include "hf_cuda_kernels.cuh"
#include "siddon_device_kernelsMLEM.cuh"


uint3 find_max_uint3(uint3 *index, int length)
{
	uint3 max_index;
	max_index.x = 0;
	max_index.y = 0;
	max_index.z = 0;

	for (int i = 0; i < length; i++)
	{
		max_index.x = std::max(max_index.x, index[i].x);
		max_index.y = std::max(max_index.y, index[i].y);
		max_index.z = std::max(max_index.z, index[i].z);
	}
	return max_index;
}


__global__ void initiate_iterator(	float *d_alpha_xi, float *d_alpha_yi, float *d_alpha_zi,
									float *d_alpha_x, float *d_alpha_y, float *d_alpha_z)

{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;
	
	d_alpha_x[n] = d_alpha_xi[n];
	d_alpha_y[n] = d_alpha_yi[n];
	d_alpha_z[n] = d_alpha_zi[n];
}


__global__ void siddon_add(	float *d_object, float *d_temp )
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	d_object[n] = d_object[n] + d_temp[n];
}

__global__ void siddon_add_fp(	float *d_image, float *d_image_temp )
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	d_image[n] = d_image[n] + d_image_temp[n];
}

__global__ void siddon_add_bp(	float *d_object, float *d_object_temp )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	d_object[n] = d_object[n] + d_object_temp[n];
}

__global__ void siddon_add_bp_T(float *d_object, float *d_object_temp )
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	d_object[n] = d_object[n] + d_object_temp[n];
}

__global__ void siddon_calc_d12(XraySystem sys, float *RM, float *d_d12)
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	float3 p1, p2;
	calc_points12(p1, p2, sys.CalParam, sys.Det, sys.voxparam, RM, sys.XraySource, depth, height, 0);

	d_d12[n] = sqrtf(	(p2.x - p1.x)*(p2.x - p1.x) + 
						(p2.y - p1.y)*(p2.y - p1.y) + 
						(p2.z - p1.z)*(p2.z - p1.z) );

}

__global__ void siddon_alpha_limits(	float *d_alpha_min, float *d_alpha_max, 
										unsigned int *d_mask, float *d_d12,
										float3 *d_alpha_i, float3 *d_alpha_f, float3 *d_delta_alpha,
										unsigned int *d_delta_index, 
										float *RM, float angle_degree,
										XraySystem sys, float3 *p1, float3 *p2)
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	calc_points12(p1[n], p2[n], sys.CalParam, sys.Det, sys.voxparam, RM, sys.XraySource, depth, height, angle_degree);
	
	d_d12[n] = sqrtf(	(p2[n].x - p1[n].x)*(p2[n].x - p1[n].x) + 
						(p2[n].y - p1[n].y)*(p2[n].y - p1[n].y) + 
						(p2[n].z - p1[n].z)*(p2[n].z - p1[n].z) );

	//d_d12[n] = 1.0f;
	//device functions that calculates alpha_1 and alpha_N;
	float2 alphax, alphay, alphaz;
	calc_alpha_1N(p1[n].x, p2[n].x, sys.P_initial.x, sys.P_final.x, alphax);
	calc_alpha_1N(p1[n].y, p2[n].y, sys.P_initial.y, sys.P_final.y, alphay);
	calc_alpha_1N(p1[n].z, p2[n].z, sys.P_initial.z, sys.P_final.z, alphaz);

	d_alpha_min[n] = fmaxf( fmaxf( fmaxf( 0, fminf(alphax.x, alphax.y) ), fminf(alphay.x, alphay.y) ), fminf(alphaz.x, alphaz.y) );
	d_alpha_max[n] = fminf( fminf( fminf( 1, fmaxf(alphax.x, alphax.y) ), fmaxf(alphay.x, alphay.y) ), fmaxf(alphaz.x, alphaz.y) );
	d_mask[n] = calc_mask(d_alpha_min[n], d_alpha_max[n]);
	d_alpha_min[n] = d_mask[n] * d_alpha_min[n];
	d_alpha_max[n] = d_mask[n] * d_alpha_max[n];


	uint3 index;
	index.x = siddon_calc_indices(d_alpha_i[n].x, d_alpha_f[n].x, d_mask[n], p1[n].x, p2[n].x, sys.P_initial.x, sys.P_final.x, sys.voxparam.Xdim + 1.0f, d_alpha_min[n], d_alpha_max[n], sys.P_delta.x);
	index.y = siddon_calc_indices(d_alpha_i[n].y, d_alpha_f[n].y, d_mask[n], p1[n].y, p2[n].y, sys.P_initial.y, sys.P_final.y, sys.voxparam.Ydim + 1.0f, d_alpha_min[n], d_alpha_max[n], sys.P_delta.y);
	index.z = siddon_calc_indices(d_alpha_i[n].z, d_alpha_f[n].z, d_mask[n], p1[n].z, p2[n].z, sys.P_initial.z, sys.P_final.z, sys.voxparam.Zdim + 1.0f, d_alpha_min[n], d_alpha_max[n], sys.P_delta.z);

	d_delta_index[n] = index.x + 1 + index.y + 1 + index.z + 1 + 1;
	
	//d_index_limits_i[n] = siddon_calc_indices2(d_alpha_i[n].x, d_alpha_f[n].x, d_mask[n], p1[n].x, p2[n].x, sys.P_initial.x, sys.P_final.x, sys.voxparam.Xdim + 1.0f, d_alpha_min[n], d_alpha_max[n], sys.P_delta.x);
	//d_index_limits_j[n] = siddon_calc_indices2(d_alpha_i[n].y, d_alpha_f[n].y, d_mask[n], p1[n].y, p2[n].y, sys.P_initial.y, sys.P_final.y, sys.voxparam.Ydim + 1.0f, d_alpha_min[n], d_alpha_max[n], sys.P_delta.y);
	//d_index_limits_k[n] = siddon_calc_indices2(d_alpha_i[n].z, d_alpha_f[n].z, d_mask[n], p1[n].z, p2[n].z, sys.P_initial.z, sys.P_final.z, sys.voxparam.Zdim + 1.0f, d_alpha_min[n], d_alpha_max[n], sys.P_delta.z);

	d_delta_alpha[n].x = delta_alpha(sys.voxparam.XVoxSize, p2[n].x, p1[n].x, index.x, d_alpha_i[n].x, d_alpha_f[n].x);
	d_delta_alpha[n].y = delta_alpha(sys.voxparam.YVoxSize, p2[n].y, p1[n].y, index.y, d_alpha_i[n].y, d_alpha_f[n].y);
	d_delta_alpha[n].z = delta_alpha(sys.voxparam.ZVoxSize, p2[n].z, p1[n].z, index.z, d_alpha_i[n].z, d_alpha_f[n].z);

}



__global__ void siddon_sort_alpha(	float3 *d_alpha, float3 *d_alpha_i, float3 *d_alpha_f, float3 *d_delta_alpha,
									float *d_current_alpha, float *d_next_alpha, float *d_alpha_max, float *d_alpha_min)
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;
	
	d_current_alpha[n] = d_next_alpha[n];

	if (d_current_alpha[n] == d_alpha[n].x && d_alpha_i[n].x != CUDART_NAN_F && d_alpha_f[n].x != CUDART_NAN_F)
	{
		d_alpha[n].x = d_alpha[n].x + d_delta_alpha[n].x;
		d_alpha[n].x = alpha_bound(d_alpha[n].x, d_alpha_min[n], d_alpha_i[n].x, d_alpha_f[n].x, d_alpha_max[n]);
	}
	
	else if (d_current_alpha[n] == d_alpha[n].y && d_alpha_i[n].y != CUDART_NAN_F && d_alpha_f[n].y != CUDART_NAN_F)
	{
		d_alpha[n].y = d_alpha[n].y + d_delta_alpha[n].y;
		d_alpha[n].y = alpha_bound(d_alpha[n].y, d_alpha_min[n], d_alpha_i[n].y, d_alpha_f[n].y, d_alpha_max[n]);
	}

	else if (d_current_alpha[n] == d_alpha[n].z && d_alpha_i[n].z != CUDART_NAN_F && d_alpha_f[n].z != CUDART_NAN_F)
	{
		d_alpha[n].z = d_alpha[n].z + d_delta_alpha[n].z;
		d_alpha[n].z = alpha_bound(d_alpha[n].z, d_alpha_min[n], d_alpha_i[n].z, d_alpha_f[n].z, d_alpha_max[n]);
	}
	else{}

	if ( d_alpha[n].x!= CUDART_NAN_F && d_alpha[n].y!= CUDART_NAN_F && d_alpha[n].z!= CUDART_NAN_F )
	{
		d_next_alpha[n] = fminf( fminf( d_alpha[n].x, d_alpha[n].y ), d_alpha[n].z );
	}
	else
	{
		d_next_alpha[n] = CUDART_NAN_F;
	}

}


__global__ void siddon_fp_calc_ray_path(	XraySystem sys, float3 *p1, float3 *p2, int3 *index,
											float *d_current_alpha, float *d_next_alpha, float *d_d12, unsigned int *d_mask,
											float *d_image, float *d_object)
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	float ray_length;
	if (d_mask[n] != CUDART_NAN_F && d_next_alpha[n]!= CUDART_NAN_F && d_current_alpha[n]!= CUDART_NAN_F && d_next_alpha[n] > d_current_alpha[n])
	{
		float alpha_mid =	__fdividef( (d_next_alpha[n] + d_current_alpha[n]), 2.0f );
		ray_length =	d_d12[n] * (d_next_alpha[n] - d_current_alpha[n]);
		

		index[n].x = __float2int_rz( (__fdividef( (p1[n].x + alpha_mid*(p2[n].x - p1[n].x) - sys.P_initial.x) , sys.P_delta.x ) ) );
		index[n].y = __float2int_rz( (__fdividef( (p1[n].y + alpha_mid*(p2[n].y - p1[n].y) - sys.P_initial.y) , sys.P_delta.y ) ) );
		index[n].z = __float2int_rz( (__fdividef( (p1[n].z + alpha_mid*(p2[n].z - p1[n].z) - sys.P_initial.z) , sys.P_delta.z ) ) );
	}
	else
	{
		index[n].x = -1;
		index[n].y = -1;
		index[n].z = -1;
		ray_length = 0.0f;
		d_image[n] = 0.0f;
	}

	if (index[n].x >= 0 && index[n].x < sys.voxparam.Xdim &&
		index[n].y >= 0 && index[n].y < sys.voxparam.Ydim && 
		index[n].z >= 0 && index[n].z < sys.voxparam.Zdim )
	{
		unsigned int object_index = index[n].z*(sys.voxparam.Ydim*sys.voxparam.Xdim) + index[n].y*sys.voxparam.Xdim + index[n].x;
		d_image[n] = ray_length * d_object[object_index];			
	}
	else
	{
		d_image[n] = 0.0f;
	}

}




__global__ void siddon_fp_end(	float *d_alpha_max,
								float *d_d12, unsigned int *d_mask,
								XraySystem sys, float3 *p1, float3 *p2, int3 *index,
								float *d_current_alpha, float *d_image, float *d_object)
{

	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	float ray_length;
	if ( d_mask[n] != CUDART_NAN_F && d_alpha_max[n] > d_current_alpha[n])
	{
		float alpha_mid = __fdividef( (d_alpha_max[n] + d_current_alpha[n]), 2.0f );
		ray_length = d_d12[n] * (d_alpha_max[n] - d_current_alpha[n]);

		index[n].x = __float2int_rz((__fdividef( (p1[n].x + alpha_mid*(p2[n].x - p1[n].x) - sys.P_initial.x) , sys.P_delta.x ) ) );
		index[n].y = __float2int_rz((__fdividef( (p1[n].y + alpha_mid*(p2[n].y - p1[n].y) - sys.P_initial.y) , sys.P_delta.y ) ) );
		index[n].z = __float2int_rz((__fdividef( (p1[n].z + alpha_mid*(p2[n].z - p1[n].z) - sys.P_initial.z) , sys.P_delta.z ) ) );
	}
	else
	{
		ray_length = 0.0f;
		index[n].x = -1;
		index[n].y = -1;
		index[n].z = -1;
		d_image[n] = 0.0f;
	}
		
	if (index[n].x >= 0 && index[n].x < sys.voxparam.Xdim &&
		index[n].y >= 0 && index[n].y < sys.voxparam.Ydim && 
		index[n].z >= 0 && index[n].z < sys.voxparam.Zdim )
	{	
		unsigned int object_index = index[n].z *(sys.voxparam.Ydim*sys.voxparam.Xdim) + index[n].y*sys.voxparam.Xdim + index[n].x;
		d_image[n] = ray_length *d_object[object_index]; //original code	
	}
	else
	{
		d_image[n] = 0.0f;
	}

}



__global__ void siddon_begin(	float *d_alpha_min, float3 *d_alpha, float3 *d_alpha_i,
								unsigned int *d_mask, float *d_next_alpha)
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	d_alpha[n] = d_alpha_i[n];
	d_alpha[n].x = d_mask[n] * d_alpha[n].x;
	d_alpha[n].y = d_mask[n] * d_alpha[n].y;
	d_alpha[n].z = d_mask[n] * d_alpha[n].z;

	if ( d_alpha_i[n].x!= CUDART_NAN_F && d_alpha_i[n].y!= CUDART_NAN_F && d_alpha_i[n].z!= CUDART_NAN_F )
	{
		d_next_alpha[n] = fminf( fminf( d_alpha_i[n].x, d_alpha_i[n].y ), d_alpha_i[n].z );
	}
	else
	{
		d_next_alpha[n] = CUDART_NAN_F;
	}
}



__global__ void siddon_bp_calc_ray_path(XraySystem sys, float3 *p1, float3 *p2, int3 *index, int *d_object_index,
										float *d_current_alpha, float *d_next_alpha, float *d_d12, unsigned int *d_mask,
										float *d_image, float *d_image_value)
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;
	
	float ray_length;
	if (d_mask[n] != CUDART_NAN_F && d_next_alpha[n]!= CUDART_NAN_F && d_current_alpha[n]!= CUDART_NAN_F && d_next_alpha[n] > d_current_alpha[n])
	{
		float alpha_mid =	__fdividef( d_next_alpha[n] + d_current_alpha[n] , 2.0f );
		ray_length = d_d12[n] * ( d_next_alpha[n] - d_current_alpha[n]);
			
		//int3 index;
		index[n].x = __float2int_rz((__fdividef( p1[n].x + alpha_mid*(p2[n].x - p1[n].x) - sys.P_initial.x , sys.P_delta.x ) ) );
		index[n].y = __float2int_rz((__fdividef( p1[n].y + alpha_mid*(p2[n].y - p1[n].y) - sys.P_initial.y , sys.P_delta.y ) ) );
		index[n].z = __float2int_rz((__fdividef( p1[n].z + alpha_mid*(p2[n].z - p1[n].z) - sys.P_initial.z , sys.P_delta.z ) ) );
	}
	else
	{   
		d_object_index[n] = sys.voxparam.Xdim*sys.voxparam.Ydim*sys.voxparam.Zdim;
		ray_length = 0.0f;
		index[n].x = -1;
		index[n].y = -1;
		index[n].z = -1;
		d_image_value[n] = 0.0f;
	}
	
	if (index[n].x >= 0 && index[n].x < sys.voxparam.Xdim &&
		index[n].y >= 0 && index[n].y < sys.voxparam.Ydim && 
		index[n].z >= 0 && index[n].z < sys.voxparam.Zdim )
	{
			d_object_index[n] = index[n].z*(sys.voxparam.Ydim*sys.voxparam.Xdim) + index[n].y*sys.voxparam.Xdim + index[n].x;
			d_image_value[n] = ray_length * d_image[n];
	}
	else
	{
		d_image_value[n] = 0.0f;
		d_object_index[n] = sys.voxparam.Xdim*sys.voxparam.Ydim*sys.voxparam.Zdim;
		index[n].x = -1;
		index[n].y = -1;
		index[n].z = -1;
	}

}

__global__ void kernel(float *d_object, int *d_object_index, float *d_image_value, int size)
{

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
  
	while (i < size)
	{
		atomicAdd( &d_object[d_object_index[i]], d_image_value[i] );
		i += stride;
	}
}

__global__ void siddon_bp_end(	float *d_alpha_max,
								float *d_d12, unsigned int *d_mask,
								XraySystem sys, float3 *p1, float3 *p2, int3 *index, int *d_object_index,
								float *d_current_alpha, float *d_image, float *d_image_value)
{

	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	float ray_length;
	if ( d_mask[n] != CUDART_NAN_F && d_alpha_max[n] > d_current_alpha[n])
	{
		float alpha_mid = __fdividef( (d_alpha_max[n] + d_current_alpha[n]), 2.0f );
		ray_length = d_d12[n] * (d_alpha_max[n] - d_current_alpha[n]);
		
		//int3 index;
		index[n].x = __float2int_rz((__fdividef( (p1[n].x + alpha_mid*(p2[n].x - p1[n].x) - sys.P_initial.x) , sys.P_delta.x ) ) );
		index[n].y = __float2int_rz((__fdividef( (p1[n].y + alpha_mid*(p2[n].y - p1[n].y) - sys.P_initial.y) , sys.P_delta.y ) ) );
		index[n].z = __float2int_rz((__fdividef( (p1[n].z + alpha_mid*(p2[n].z - p1[n].z) - sys.P_initial.z) , sys.P_delta.z ) ) );
	}
	else
	{
		d_object_index[n] = sys.voxparam.Xdim*sys.voxparam.Ydim*sys.voxparam.Zdim;
		d_image_value[n] = 0.0f;
		ray_length = 0.0f;
		index[n].x = -1;
		index[n].y = -1;
		index[n].z = -1;
	}

	if (index[n].x >= 0 && index[n].x < sys.voxparam.Xdim &&
		index[n].y >= 0 && index[n].y < sys.voxparam.Ydim && 
		index[n].z >= 0 && index[n].z < sys.voxparam.Zdim )
	{	
		d_object_index[n] = index[n].z *(sys.voxparam.Ydim*sys.voxparam.Xdim) + index[n].y*(sys.voxparam.Xdim) + index[n].x;
		d_image_value[n] = ray_length * d_image[n];
	}
	else
	{
		d_object_index[n] = sys.voxparam.Xdim*sys.voxparam.Ydim*sys.voxparam.Zdim;
		d_image_value[n] = 0.0f;
	}

}




#endif



//__global__ void siddon_fp_begin(	float *d_alpha_min,
//									float3 *d_alpha, float3 *d_alpha_i, float3 *d_alpha_f,
//									float3 *d_delta_alpha,
//									float *d_d12, unsigned int *d_mask,
//									float *RM, float angle_degree,
//									XraySystem sys, float3 *p1, float3 *p2,
//									float *d_current_alpha, float *d_image, float *d_object)
//{
//	int depth = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = depth + height * blockDim.x * gridDim.x;
//
//	d_alpha[n] = d_alpha_i[n];
//
//	d_delta_alpha[n].x = __fdividef( sys.voxparam.XVoxSize, fabsf(p2[n].x - p1[n].x) );
//	d_delta_alpha[n].y = __fdividef( sys.voxparam.YVoxSize, fabsf(p2[n].y - p1[n].y) );
//	d_delta_alpha[n].z = __fdividef( sys.voxparam.ZVoxSize, fabsf(p2[n].z - p1[n].z) );
//	
//	d_current_alpha[n] = fminf( fminf( d_alpha[n].x, d_alpha[n].y ), d_alpha[n].z );
//	
//	if (d_mask[n] != CUDART_NAN_F && d_alpha_min[n] != CUDART_NAN_F && d_current_alpha[n] != CUDART_NAN_F)
//	{
//		float alpha_mid = __fdividef( (d_alpha_min[n] + d_current_alpha[n]), 2.0f );
//		float ray_length = d_d12[n] * fabsf(d_current_alpha[n] - d_alpha_min[n]);
//		
//		int3 index;
//		index.x = __float2int_rd( (__fdividef( (p1[n].x + alpha_mid*(p2[n].x - p1[n].x) - sys.P_initial.x) , sys.P_delta.x ) ) );
//		index.y = __float2int_rd( (__fdividef( (p1[n].y + alpha_mid*(p2[n].y - p1[n].y) - sys.P_initial.y) , sys.P_delta.y ) ) );
//		index.z = __float2int_rd( (__fdividef( (p1[n].z + alpha_mid*(p2[n].z - p1[n].z) - sys.P_initial.z) , sys.P_delta.z ) ) );
//
//		if (index.x >= 0 && index.x < sys.voxparam.Xdim &&
//			index.y >= 0 && index.y < sys.voxparam.Ydim && 
//			index.z >= 0 && index.z < sys.voxparam.Zdim )
//		{	
//			unsigned int object_index = index.z *(sys.voxparam.Ydim*sys.voxparam.Xdim) + index.y*sys.voxparam.Xdim + index.x;
//			d_image[n] = ray_length * d_object[object_index]; //original code	
//			//d_image[n] = d_object[object_index];
//
//		}
//		else
//		{
//			d_image[n] = 0.0f;
//		}
//	}
//	else
//	{
//		d_image[n] = 0.0f;
//	}
//	
//}


//__global__ void siddon_fp_calc_ray_path(float *RM, float angle_degree,
//										XraySystem sys, float3 *p1, float3 *p2,
//										float *d_current_alpha, float *d_next_alpha, float *d_d12, unsigned int *d_mask,
//										float *d_image, float *d_object)
//{
//	int depth = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = depth + height * blockDim.x * gridDim.x;
//
//	if ( d_mask[n] != CUDART_NAN_F && d_next_alpha[n] != CUDART_NAN_F && d_current_alpha[n] != CUDART_NAN_F) 
//	{
//		float alpha_mid =	__fdividef( (d_next_alpha[n] + d_current_alpha[n]), 2.0f );
//		float ray_length =	d_d12[n] * (d_next_alpha[n] - d_current_alpha[n]);
//		
//		int3 index;
//		index.x = __float2int_rd( (__fdividef( (p1[n].x + alpha_mid*(p2[n].x - p1[n].x) - sys.P_initial.x) , sys.P_delta.x ) ) );
//		index.y = __float2int_rd( (__fdividef( (p1[n].y + alpha_mid*(p2[n].y - p1[n].y) - sys.P_initial.y) , sys.P_delta.y ) ) );
//		index.z = __float2int_rd( (__fdividef( (p1[n].z + alpha_mid*(p2[n].z - p1[n].z) - sys.P_initial.z) , sys.P_delta.z ) ) );
//
//		if (index.x >= 0 && index.x < sys.voxparam.Xdim &&
//			index.y >= 0 && index.y < sys.voxparam.Ydim && 
//			index.z >= 0 && index.z < sys.voxparam.Zdim )
//		{
//			unsigned int object_index = index.z*(sys.voxparam.Ydim*sys.voxparam.Xdim) + index.y*sys.voxparam.Xdim + index.x;
//			d_image[n] = ray_length * d_object[object_index];			
//			//d_image[n] = d_object[object_index];
//		}
//		else
//		{
//			d_image[n] = 0.0f;
//		}
//	}
//	else
//	{
//		d_image[n] = 0.0f;
//	}
//
//}

//
//__global__ void siddon_bp_begin_old(	float *d_alpha_min, 
//									float3 *d_alpha, float3 *d_alpha_i,
//									float *d_d12, unsigned int *d_mask,
//									float *RM, float angle_degree,
//									XraySystem sys, float3 *p1, float3 *p2, int3 *index, 
//									float *d_current_alpha, float *d_image, float *d_object)
//{
//	int depth = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = depth + height * blockDim.x * gridDim.x;
//
//	d_alpha[n] = d_alpha_i[n];
//
//	unsigned int object_index = 0;
//	float image_value = 0;
//	if (d_mask[n] != CUDART_NAN_F && d_alpha_min[n] != CUDART_NAN_F && d_current_alpha[n] != CUDART_NAN_F)
//	{
//		d_current_alpha[n] = fminf( fminf( d_alpha[n].x, d_alpha[n].y ), d_alpha[n].z );
//		float alpha_mid = __fdividef( (d_alpha_min[n] + d_current_alpha[n]), 2.0f );
//		float ray_length = d_d12[n]*(d_current_alpha[n] - d_alpha_min[n]);
//		
//		//int3 index;
//		index[n].x = __float2int_rz((__fdividef( (p1[n].x + alpha_mid*(p2[n].x - p1[n].x) - sys.P_initial.x) , sys.P_delta.x ) ) );
//		index[n].y = __float2int_rz((__fdividef( (p1[n].y + alpha_mid*(p2[n].y - p1[n].y) - sys.P_initial.y) , sys.P_delta.y ) ) );
//		index[n].z = __float2int_rz((__fdividef( (p1[n].z + alpha_mid*(p2[n].z - p1[n].z) - sys.P_initial.z) , sys.P_delta.z ) ) );
//
//
//		if (index[n].x >= 0 && index[n].x < sys.voxparam.Xdim &&
//			index[n].y >= 0 && index[n].y < sys.voxparam.Ydim && 
//			index[n].z >= 0 && index[n].z < sys.voxparam.Zdim )
//		{	
//			object_index = index[n].z*(sys.voxparam.Ydim*sys.voxparam.Xdim) + index[n].y*sys.voxparam.Xdim + index[n].x;
//			image_value = ray_length * d_image[n];
//		}
//	}
//	d_object[object_index] = image_value;
//}
//
//__global__ void siddon_bp_begin(	float *d_alpha_min, 
//									float3 *d_alpha, float3 *d_alpha_i,
//									unsigned int *d_mask,
//									float *d_next_alpha)
//{
//	int depth = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = depth + height * blockDim.x * gridDim.x;
//
//	d_alpha[n] = d_alpha_i[n];
//	d_alpha[n].x = d_mask[n] * d_alpha[n].x;
//	d_alpha[n].y = d_mask[n] * d_alpha[n].y;
//	d_alpha[n].z = d_mask[n] * d_alpha[n].z;
//
//	if ( d_alpha_i[n].x!= CUDART_NAN_F && d_alpha_i[n].y!= CUDART_NAN_F && d_alpha_i[n].z!= CUDART_NAN_F )
//	{
//		d_next_alpha[n] = fminf( fminf( d_alpha_i[n].x, d_alpha_i[n].y ), d_alpha_i[n].z );
//	}
//	else
//	{
//		d_next_alpha[n] = CUDART_NAN_F;
//	}
//}