//Written by Helen Fan

#ifndef HF_CUDA_KERNELS_CUH_
#define HF_CUDA_KERNELS_CUH_

#include "cuda.h"
#include "cuda_runtime.h"
#include "CTSystem_HF.hpp"
#include "hf_texture_helper.hpp"

#define CUDA_SAFE_CALL(call) do 																				\
	{																											\
		int err = call;																							\
		if(cudaSuccess != err)																					\
		{																										\
			fprintf(stderr, "Cuda driver error %04Xh in file \"%s\" in line %i.\n", err, __FILE__, __LINE__);	\
			fprintf(stderr, "CUDA Error message: %s. \n", cudaGetErrorString( (cudaError_t) err) );				\
			exit(EXIT_FAILURE);																					\
		}																										\
	} while(0)


#define M_PI 3.1415927// 3.14159265358979323846

__global__ void set_ones(float *d_data)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	d_data[n] = 1.0f;
}

__global__ void set_zeros(float *d_data)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	d_data[n] = 0.0f;
}

__global__ void divide_by_scale(float *d_imgdata, float *d_scale)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int n = x + y * blockDim.x * gridDim.x;

	d_imgdata[n] = __fdividef( d_imgdata[n], d_scale[n] );
}

__global__ void rotate_volume_3D(float *rotatedvolume, float *rotationmatrix)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	float xn = x / ( (float)blockDim.x * (float)gridDim.x ) - 0.5f;
	float yn = y / ( (float)blockDim.y * (float)gridDim.y ) - 0.5f;
	float zn = z / ( (float)blockDim.z * (float)gridDim.z ) - 0.5f;

	float xr = xn*rotationmatrix[0] + yn*rotationmatrix[1] + zn*rotationmatrix[2] + 0.5f;
	float yr = xn*rotationmatrix[3] + yn*rotationmatrix[4] + zn*rotationmatrix[5] + 0.5f;
	float zr = xn*rotationmatrix[6] + yn*rotationmatrix[7] + zn*rotationmatrix[8] + 0.5f;

	rotatedvolume[n] = tex3D(tex3_float, xr, yr, zr);
}

__device__ float distance_traveled(float YVoxSize, float v_id, float u_id, float R)
{
	return( __fdividef( sqrt(v_id*v_id + u_id*u_id + R*R), R ) * YVoxSize );
}

__device__ float pixel_iterate( float NumPixels, float PixelsDimMm, float Mag, int iterator)
{
	return ( -(NumPixels-1.0f)/2.0f*PixelsDimMm*Mag + (float)iterator*PixelsDimMm*Mag );

}


//more accurate when compared to matlab linspace, error is lowest towards 0, higher later, but around 1e-6 range
//more centered
__device__ float pixel_iterate2( float NumPixels, float PixelsDimMm, float Mag, int iterator)
{
	float result = __fdividef(-(NumPixels-1.0f),2.0f)*PixelsDimMm*Mag + __fdividef( (float)iterator, NumPixels)*NumPixels*PixelsDimMm*Mag;
	return (  __fdiv_rz( rintf(result * 100000.0f), 100000.0f)  );
	//return(result);
}

__device__ float total_distance(float v_id, float u_id, float R)
{

	return( sqrtf( powf(v_id, 2.0f) + powf(u_id, 2.0f) + powf(R, 2.0f) ) );

}

__global__ void add_total_distance_traveled(float *md_FPimage, CalibrationParam CalParam, Detector Det, float *RM)
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	float u = pixel_iterate( Det.NumAxPixels, Det.AxPixDimMm, CalParam.M, depth );
	float v = pixel_iterate( Det.NumTAxPixels, Det.TAxPixDimMm, CalParam.M, height );

	float u_id = __fdividef(CalParam.R, (CalParam.Ry + RM[3]*u + RM[5]*v)) *(RM[0]*u + RM[2]*v + CalParam.Dx);
	float v_id = __fdividef(CalParam.R, (CalParam.Ry + RM[3]*u + RM[5]*v)) *(RM[6]*u + RM[8]*v + CalParam.Dz);

	md_FPimage[n] = md_FPimage[n] + __fdividef( total_distance(u_id, v_id, CalParam.R), 1000.0f );

}

//forward projection 2
//alpha (angle in radians, is the rotation for CT)
__global__ void forward_projection(float *md_FPimage, CalibrationParam CalParam, Detector Det, ObjectParam ObjParam,
									VoxVolParam voxparam, float *RM, float y0, float alpha_deg)
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	//camera detector pixel positions
	//float u = -(Det.NumAxPixels/2.0f-1.0f)*Det.AxPixDimMm*CalParam.M-Det.AxPixDimMm/2.0f*CalParam.M + (float)depth*Det.AxPixDimMm*CalParam.M;
	//float v = -(Det.NumTAxPixels/2.0f-1.0f)*Det.TAxPixDimMm*CalParam.M-Det.TAxPixDimMm/2.0f*CalParam.M + (float)height*Det.TAxPixDimMm*CalParam.M;

	float u = pixel_iterate( Det.NumAxPixels, Det.AxPixDimMm, CalParam.M, depth );
	float v = pixel_iterate( Det.NumTAxPixels, Det.TAxPixDimMm, CalParam.M, height );

	float u_id = __fdividef(CalParam.R, (CalParam.Ry + RM[3]*u + RM[5]*v)) *(RM[0]*u + RM[2]*v + CalParam.Dx);
	float v_id = __fdividef(CalParam.R, (CalParam.Ry + RM[3]*u + RM[5]*v)) *(RM[6]*u + RM[8]*v + CalParam.Dz);

	float L = ((float)voxparam.Ydim-1.0f)*voxparam.YVoxSize;

	//alpha is in radians, alpha is the CT rotation angle

	float sin_alpha, cos_alpha;
	sincosf(alpha_deg * (float)M_PI/180.0f, &sin_alpha, &cos_alpha);

	float x0 = __fdividef( (y0 + CalParam.Rf)*u_id, CalParam.R );
	float z0 = __fdividef( (y0 + CalParam.Rf)*v_id, CalParam.R );

	float x_alpha_norm =  __fdividef(  x0*cos_alpha + y0*sin_alpha, L ) + 0.5f;
	float y_alpha_norm =  __fdividef( -x0*sin_alpha + y0*cos_alpha, L ) + 0.5f;
	float z_alpha_norm =  __fdividef(  z0, L ) + 0.5f;

	//float x_alpha =  ( x0*cos_alpha + y0*sin_alpha + (float)(voxparam.Xdim-1)/2.0f) ;
	//float y_alpha =  (-x0*sin_alpha + y0*cos_alpha + (float)(voxparam.Ydim-1)/2.0f);
	//float z_alpha =   (z0 + (float)(voxparam.Zdim-1)/2.0f) ;
	
	//float x_alpha = x_alpha_norm * (float)(voxparam.Xdim-1)*voxparam.XVoxSize;
	//float y_alpha = y_alpha_norm * (float)(voxparam.Ydim-1)*voxparam.YVoxSize;
	//float z_alpha = z_alpha_norm * (float)(voxparam.Zdim-1)*voxparam.ZVoxSize;

//	float3 coord = make_float3(x_alpha, y_alpha, z_alpha);
	//if (x_alpha_norm > -0.001 && x_alpha_norm < 1.001 && y_alpha_norm > -0.001 && y_alpha_norm < 1.001 && z_alpha_norm > -0.001 && z_alpha_norm < 1.001 )
	//{
		md_FPimage[n] = md_FPimage[n] + tex3D(tex3_float, x_alpha_norm, y_alpha_norm, z_alpha_norm) * distance_traveled(voxparam.YVoxSize, v_id, u_id, CalParam.R);
	//md_FPimage[n] = u;
	//}

	//float3 coord = make_float3( x_alpha_norm * (float)(voxparam.Xdim-1), y_alpha_norm*(float)(voxparam.Ydim-1), z_alpha_norm*(float)(voxparam.Zdim-1) );

	//if (x_alpha_norm > 0.0f && x_alpha_norm <= 1.0f && y_alpha_norm > 0.0f && y_alpha_norm <= 1.0f && z_alpha_norm > 0.0f && z_alpha_norm <= 1.0f)
	//{
	//	md_FPimage[n] = md_FPimage[n] + cubicTex3D(tex3_float, coord) * distance_traveled(voxparam.YVoxSize, v_id, u_id, CalParam.R);
	//	
	//}
	
}

//alpha is in degrees
__global__ void backward_projection(float *md_Obj, CalibrationParam CalParam, Detector Det, ObjectParam ObjParam, VoxVolParam voxparam, float *RM, float alpha, unsigned int Nprojections)
{
	int x  = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	//object points
	//float xobj = -(voxparam.Xdim/2.0f-1.0f)*voxparam.XVoxSize-voxparam.XVoxSize/2.0f + voxparam.XVoxSize*x;
	//float yobj = -(voxparam.Ydim/2.0f-1.0f)*voxparam.YVoxSize-voxparam.YVoxSize/2.0f + voxparam.YVoxSize*y;
	//float zobj = -(voxparam.Zdim/2.0f-1.0f)*voxparam.ZVoxSize-voxparam.ZVoxSize/2.0f + voxparam.ZVoxSize*z;
	
	float xobj = pixel_iterate(voxparam.Xdim, voxparam.XVoxSize, 1.0f, x);
	float yobj = pixel_iterate(voxparam.Ydim, voxparam.YVoxSize, 1.0f, y);
	float zobj = pixel_iterate(voxparam.Zdim, voxparam.ZVoxSize, 1.0f, z);


	float sin_alpha, cos_alpha;
	//sincosf(alpha*(float)M_PI/180.0f, &sin_alpha, &cos_alpha);
	sin_alpha = sin(alpha*(float)M_PI/180.0f);
	cos_alpha = cos(alpha*(float)M_PI/180.0f);

	float x_alpha =  xobj*cos_alpha + yobj*sin_alpha;
	float y_alpha = -xobj*sin_alpha + yobj*cos_alpha;

	float u_id = __fdividef( CalParam.R*x_alpha, y_alpha + CalParam.Rf);
	float v_id = __fdividef( CalParam.R*zobj,    y_alpha + CalParam.Rf );
	
	float detQ = (RM[0]-RM[3]*u_id/CalParam.R)*(RM[8]-RM[5]*v_id/CalParam.R) - (RM[2]-RM[5]*u_id/CalParam.R)*(RM[6]-RM[3]*v_id/CalParam.R);

	float u = __fdividef(1, detQ)*( (RM[8]-RM[5]*v_id/CalParam.R)*(CalParam.Ry/CalParam.R*u_id - CalParam.Dx) - (RM[2]-RM[5]*u_id/CalParam.R)*(CalParam.Ry/CalParam.R*v_id - CalParam.Dz) );
	float v = __fdividef(1, detQ)*(-(RM[6]-RM[3]*v_id/CalParam.R)*(CalParam.Ry/CalParam.R*u_id - CalParam.Dx) + (RM[0]-RM[3]*u_id/CalParam.R)*(CalParam.Ry/CalParam.R*v_id - CalParam.Dz) );

	//float u_normal = fdividef( u, Det.AxPixDimMm*((float)Det.NumAxPixels-1.0f)*CalParam.M ) + 0.5f; //is it suppose to be u or u_id?
	//float v_normal = fdividef( v, Det.TAxPixDimMm*((float)Det.NumTAxPixels-1.0f)*CalParam.M ) + 0.5f;

	float u_pix = u + (float)(Det.NumAxPixels-1)/2.0f*Det.AxPixDimMm*CalParam.M; //pixel coordinate
	float v_pix = v + (float)(Det.NumTAxPixels-1)/2.0f*Det.TAxPixDimMm*CalParam.M; //pixel coordinate
	
	//md_Obj[n] = md_Obj[n] + __fdividef( tex2D(tex2, u_normal, v_normal) , (float)Nprojections*(float)voxparam.Ydim*(float)voxparam.YVoxSize );

	//if (u_normal > 0.0f && u_normal <= 1.0f && v_normal > 0.0f && v_normal <= 1.0f)
	//{
	//	//md_Obj[n] = md_Obj[n] + tex2D(tex2, u_normal, v_normal);
	//	//md_Obj[n] = md_Obj[n] + tex2D(tex2_float, u_normal, v_normal);
	//	md_Obj[n] = md_Obj[n] + __fdividef( cubicTex2D(tex2_float, u_normal*(float)(Det.NumAxPixels-1), v_normal*(float)(Det.NumTAxPixels-1) ), (float)Nprojections );
	//}

	if (u_pix >= 0 && u_pix < (float)(Det.NumAxPixels-1)*Det.AxPixDimMm*CalParam.M && v_pix >= 0.f && v_pix < (float)(Det.NumTAxPixels-1)*Det.TAxPixDimMm*CalParam.M)
	{
		md_Obj[n] = md_Obj[n] + __fdividef( cubicTex2D(tex2_float, u_pix, v_pix ), (float)Nprojections );
	}
	
}

//alpha is in degrees
__global__ void backward_projection2(float *md_Obj, float *d_detector_value, CalibrationParam CalParam, Detector Det, ObjectParam ObjParam, VoxVolParam voxparam, float *RM, float alpha, unsigned int Nprojections)
{
	int x  = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	float xobj = pixel_iterate(voxparam.Xdim, voxparam.XVoxSize, 1.0f, x);
	float yobj = pixel_iterate(voxparam.Ydim, voxparam.YVoxSize, 1.0f, y);
	float zobj = pixel_iterate(voxparam.Zdim, voxparam.ZVoxSize, 1.0f, z);


	float sin_alpha, cos_alpha;
	//sincosf(alpha*(float)M_PI/180.0f, &sin_alpha, &cos_alpha);
	sin_alpha = sinf(alpha*(float)M_PI/180.0f);
	cos_alpha = cosf(alpha*(float)M_PI/180.0f);

	float x_alpha =  xobj*cos_alpha + yobj*sin_alpha;
	float y_alpha = -xobj*sin_alpha + yobj*cos_alpha;

	float u_id = __fdividef( CalParam.R*x_alpha, y_alpha + CalParam.Rf);
	float v_id = __fdividef( CalParam.R*zobj,    y_alpha + CalParam.Rf );
	
	float detQ = (RM[0]-RM[3]*u_id/CalParam.R)*(RM[8]-RM[5]*v_id/CalParam.R) - (RM[2]-RM[5]*u_id/CalParam.R)*(RM[6]-RM[3]*v_id/CalParam.R);

	float u = __fdividef(1, detQ)*( (RM[8]-RM[5]*v_id/CalParam.R)*(CalParam.Ry/CalParam.R*u_id - CalParam.Dx) - (RM[2]-RM[5]*u_id/CalParam.R)*(CalParam.Ry/CalParam.R*v_id - CalParam.Dz) );
	float v = __fdividef(1, detQ)*(-(RM[6]-RM[3]*v_id/CalParam.R)*(CalParam.Ry/CalParam.R*u_id - CalParam.Dx) + (RM[0]-RM[3]*u_id/CalParam.R)*(CalParam.Ry/CalParam.R*v_id - CalParam.Dz) );

	float u_pix = u + (float)(Det.NumAxPixels-1)/2.0f*Det.AxPixDimMm*CalParam.M;   //pixel coordinate
	float v_pix = v + (float)(Det.NumTAxPixels-1)/2.0f*Det.TAxPixDimMm*CalParam.M; //pixel coordinate
	
	unsigned int u_int = __float2uint_rd(u_pix);
	unsigned int v_int = __float2uint_rd(v_pix);

	float u_weight = (u_pix-(float)u_int);
	float v_weight = (v_pix-(float)v_int);

	if(u_pix >= 0.0 && u_pix <= (float)(Det.NumAxPixels-1)*Det.AxPixDimMm*CalParam.M && v_pix >= 0.0f && v_pix <= (float)(Det.NumTAxPixels-1)*Det.TAxPixDimMm*CalParam.M)
	{
		md_Obj[n] = md_Obj[n] + __fdividef( (d_detector_value[u_int + v_int*Det.NumAxPixels]*(1-u_weight) + d_detector_value[u_int+1 + v_int*Det.NumAxPixels]*u_weight)*(1-v_weight) + 
								(d_detector_value[u_int + (v_int+1)*Det.NumAxPixels]*(1-u_weight) + d_detector_value[u_int+1 + (v_int+1)*Det.NumAxPixels]*u_weight)*v_weight, (float)Nprojections );
	}

}


__global__ void round_3(float *value)
{
	int x  = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	value[n] = roundf( value[n] * 10000 ) / 10000;

}




__global__ void add(float *first, float *second)
{

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int z = threadIdx.z + blockIdx.z * blockDim.z;

	int n = x + y * blockDim.x * gridDim.x +
				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	first[n] = first[n] + second[n];
}

__global__ void negative_exponential(float *md_FPimage, float gain, float scale_factor) //used 
{
	int depth = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int n = depth + height * blockDim.x * gridDim.x;

	md_FPimage[n] = gain*expf(-1.0f*md_FPimage[n]*scale_factor);
}



__global__ void half(float *data)
{
	int width  = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = width + height * blockDim.x * gridDim.x +
				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	data[n] = __fdividef(data[n], 2.0f);
}


__global__ void apply_negative_log(float *md_Obj, float scale_factor)
{
	int width  = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = width + height * blockDim.x * gridDim.x +
				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	md_Obj[n] = -1.0f*scale_factor*logf(md_Obj[n]);

}

__global__ void divide(float *d_data, float number)
{
	int width  = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = width + height * blockDim.x * gridDim.x +
				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	if (number == 0)
	{
		d_data[n] = 0.0f;
	}
	else
	{

		d_data[n] = __fdividef( d_data[n], number );
	}

}

template <typename T>
__global__ void divide1 (T *d_numerator, T *d_denominator)
{
	int width  = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = width + height * blockDim.x * gridDim.x +
				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	if (d_denominator[n] != 0 )
	{
		d_numerator[n] = __fdividef( d_numerator[n], d_denominator[n] );
	}
	else
	{
		d_numerator[n] = 0.0f;
	}
}

template <typename T>
__global__ void divide2 (T *d_numerator, T *d_denominator)
{
	int width  = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = width + height * blockDim.x * gridDim.x +
				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	if (d_denominator[n] != 0)
	{
		d_denominator[n] = __fdividef( d_numerator[n], d_denominator[n] );
		
	}
	else
	{
		d_denominator[n] = 0.0f;
	}
}

template <typename T>
__global__ void multiply1(T *d_pointer1, T *d_pointer2)
{
	int width  = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = width + height * blockDim.x * gridDim.x +
				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	
	d_pointer1[n] = d_pointer1[n] * d_pointer2[n];

}

template <typename T>
__global__ void multiply2(T *d_pointer1, T *d_pointer2)
{
	int width  = threadIdx.x + blockIdx.x * blockDim.x;
	int height = threadIdx.y + blockIdx.y * blockDim.y;
	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
	int n = width + height * blockDim.x * gridDim.x +
				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;

	d_pointer2[n] = d_pointer1[n] * d_pointer2[n];
}


#endif


////Jared's bp code
////
//__global__ void BackProject(float* d_VoxelVolume, float* d_Projection,
//							unsigned int xdim, unsigned int ydim, unsigned int zdim,
//							float xvoxsize, float yvoxsize, float zvoxsize,
//							float R, float Rf, float ProjAngle, float z,
//							float Dx, float Dy, float Dz,
//							unsigned int AxPixels, unsigned int TAxPixels,
//							float AxSize, float TAxSize,
//							float* MMatrix)
//{
//	/* calculate cos(alpha) and sin(alpha) since they are used alot */
//	/* alpha is the angle of the projection */
//	float cosAlpha = cosf(ProjAngle);
//	float sinAlpha = sinf(ProjAngle);
//	
//	/* Voxel volume indices */
//	unsigned int xIdx = blockIdx.x;
//	unsigned int yIdx = blockIdx.y;
//	unsigned int zIdx = threadIdx.x;
//
//	/* Convert raster index to cartesian coordinates */
//	float xReal = -0.5*(xdim-1)*xvoxsize+xIdx*(xvoxsize);
//	float yReal =  0.5*(ydim-1)*yvoxsize-yIdx*(yvoxsize);
//	float zReal =  0.5*(zdim-1)*zvoxsize-zIdx*(zvoxsize)-z;
//	
//	xReal =-xReal;
//	//yReal =-yReal;
//	//zReal = -zReal;
//	
//	/* Rotate the voxel coordinates  */
//	float xrReal = xReal*cosAlpha - yReal*sinAlpha;
//	float yrReal = xReal*sinAlpha + yReal*cosAlpha;
//	float zrReal = zReal;
//
//	/* Calculate ideal detector locations with shifts  */
//	float tIdeal = (Rf-R+Dx);						//detector coord in x 
//	float uIdeal = yrReal*(R-Dx)/(xrReal+Rf)-Dy;	//detector coord in y
//	float vIdeal = zrReal*(R-Dx)/(xrReal+Rf)-Dz ;	//detector coord in z
//	
//	/* Calculate detector locations after accounting for angular misalignment */
//	float uReal = MMatrix[3]*tIdeal + MMatrix[4]*uIdeal + MMatrix[5]*vIdeal;
//	float vReal = MMatrix[6]*tIdeal + MMatrix[7]*uIdeal + MMatrix[8]*vIdeal;
//	
//	/* Calculate detector indices */
//	uReal = -(uReal-0.5*(TAxPixels-1)*TAxSize)/TAxSize;
//	vReal = -(vReal-0.5*(AxPixels -1)*AxSize )/AxSize;
//	//vReal = vReal
//
//	/* Round real location to an index */
//	unsigned int uIdx = __float2uint_rd(uReal);
//	unsigned int vIdx = __float2uint_rd(vReal);
//
//	/* weights for linear interpolation */
//	float uweight = uReal - uIdx;
//	float vweight = vReal - vIdx;
//
//	///* fetch interpolated detector value to voxel volume */
//	if(uIdx<(TAxPixels-1) && vIdx<(AxPixels-1) && uIdx>0 && vIdx>0 )
//	{
//		//Fetch detector values with bilinear interpolation
//		d_VoxelVolume[xIdx + yIdx*xdim + zIdx*xdim*ydim] +=	(1-uweight)*((1-vweight)*d_Projection[ uIdx   *AxPixels + vIdx] + vweight*d_Projection[ uIdx   *AxPixels + vIdx+1]) +
//														(  uweight)*((1-vweight)*d_Projection[(uIdx+1)*AxPixels + vIdx] + vweight*d_Projection[(uIdx+1)*AxPixels + vIdx+1]);
//	}
//
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


////alpha is in radians
//__global__ void backward_projection(float *md_Obj, CalibrationParam CalParam, Detector Det, ObjectParam ObjParam, VoxVolParam voxparam, float *RM, float alpha, unsigned int Nprojections)
//{
//	int width  = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	//object points
//	float xobj = -(voxparam.Xdim/2.0f-1.0f)*voxparam.XVoxSize-voxparam.XVoxSize/2.0f + voxparam.XVoxSize*depth;
//	float yobj = -(voxparam.Ydim/2.0f-1.0f)*voxparam.YVoxSize-voxparam.YVoxSize/2.0f + voxparam.YVoxSize*width;
//	float zobj = -(voxparam.Zdim/2.0f-1.0f)*voxparam.ZVoxSize-voxparam.ZVoxSize/2.0f + voxparam.ZVoxSize*height;
//	//float zobj = 0.0f;
//
//	float sin_alpha, cos_alpha;
//	sincosf(alpha, &sin_alpha, &cos_alpha);
//
//	float x_alpha =  xobj*cos_alpha + yobj*sin_alpha;
//	float y_alpha = -xobj*sin_alpha + yobj*cos_alpha;
//
//	float u_id = fdividef( CalParam.R*x_alpha, y_alpha + CalParam.Rf);
//	float v_id = fdividef( CalParam.R*zobj,    y_alpha + CalParam.Rf );
//	
//	float detQ = (RM[0]-RM[3]*u_id/CalParam.R)*(RM[8]-RM[5]*v_id/CalParam.R) - (RM[2]-RM[5]*u_id/CalParam.R)*(RM[6]-RM[3]*v_id/CalParam.R);
//
//	float u = fdividef(1, detQ)*( (RM[8]-RM[5]*v_id/CalParam.R)*(CalParam.Ry/CalParam.R*u_id - CalParam.Dx) - (RM[2]-RM[5]*u_id/CalParam.R)*(CalParam.Ry/CalParam.R*v_id - CalParam.Dz) );
//	float v = fdividef(1, detQ)*(-(RM[6]-RM[3]*v_id/CalParam.R)*(CalParam.Ry/CalParam.R*u_id - CalParam.Dx) + (RM[0]-RM[3]*u_id/CalParam.R)*(CalParam.Ry/CalParam.R*v_id - CalParam.Dz) );
//
//	float u_normal = fdividef( u_id, Det.AxPixDimMm*((float)Det.NumAxPixels-1.0f)*CalParam.M ) + 0.5f;
//	float v_normal = fdividef( v_id, Det.TAxPixDimMm*((float)Det.NumTAxPixels-1.0f)*CalParam.M ) + 0.5f;
//
//	md_Obj[n] = md_Obj[n] + __fdividef( tex2D(tex2, u_normal, v_normal) , (float)Nprojections );
//	//md_Obj[n] = __fdividef( tex2D(tex2, u_normal, v_normal) , (float)Nprojections );
//	//md_Obj[n] = md_Obj[n] + __fdividef( (-5.0f*voxparam.Ydim)*logf( tex2D(tex2, u_normal, v_normal) ), (float)Nprojections );
//}









////forward projection 2
////alpha (angle in radians, is the rotation for CT)
////I don't think this kernel is used at all, I think I got lazy 1/24/14
//__global__ void forward_projection_sensitivity(float *md_FPimage, CalibrationParam CalParam, Detector Det, ObjectParam ObjParam,
//									VoxVolParam voxparam, float *RM, float y0, float alpha)
//{
//	int depth = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = depth + height * blockDim.x * gridDim.x;
//
//	//camera detector pixel positions
//	float u = -(Det.NumAxPixels/2.0f-1.0f)*Det.AxPixDimMm*CalParam.M-Det.AxPixDimMm/2.0f*CalParam.M + (float)depth*Det.AxPixDimMm*CalParam.M;
//	float v = -(Det.NumTAxPixels/2.0f-1.0f)*Det.TAxPixDimMm*CalParam.M-Det.TAxPixDimMm/2.0f*CalParam.M + (float)height*Det.TAxPixDimMm*CalParam.M;
//
//	float u_id = __fdividef(CalParam.R, (CalParam.Ry + RM[3]*u + RM[5]*v)) *(RM[0]*u + RM[2]*v + CalParam.Dx);
//	float v_id = __fdividef(CalParam.R, (CalParam.Ry + RM[3]*u + RM[5]*v)) *(RM[6]*u + RM[8]*v + CalParam.Dz);
//	float L = ((float)voxparam.Ydim-1.0f)*voxparam.YVoxSize;
//
//
//	//alpha is in radians
//	float sin_alpha, cos_alpha;
//	sincosf(alpha, &sin_alpha, &cos_alpha);
//
//	float x0 = __fdividef( (y0 + CalParam.Rf)*u_id, CalParam.R );
//	float z0 = __fdividef( (y0 + CalParam.Rf)*v_id, CalParam.R );
//
//	float x_alpha_norm =  __fdividef(  x0*cos_alpha + y0*sin_alpha, L ) + 0.5f;
//	float y_alpha_norm =  __fdividef( -x0*sin_alpha + y0*cos_alpha, L ) + 0.5f;
//	float z_alpha_norm =  __fdividef(  z0, L ) + 0.5f;
//	
//	//need to multiply by 255 because the texture memory is readmode is cudaReadModeNormalizedFloat, which
//	//means that it's a floating point number ranged from [0, 1] on the scale of the element type, which
//	//in the binvox data type case is unsigned __int8, that goes from 0-255
//	//md_FPimage[n] =  md_FPimage[n] + tex3D(tex3D_uint8, y_alpha_norm, z_alpha_norm, x_alpha_norm)*(voxparam.YVoxSize);
//
//	md_FPimage[n] = md_FPimage[n] + tex3D(tex3D_float, y_alpha_norm, z_alpha_norm, x_alpha_norm)*(voxparam.YVoxSize);
//}

////forward projection
//__global__ void forwardproject(float *md_FPimage, CTGeometry CTGeom, 
//									Detector Det, ObjectParam ObjParam, 
//									VoxVolParam voxparam, float xplane, 
//									float theta_rad, float *RMobj, float *RMpho, float *RMdet)
//{
//	int depth = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = depth + height * blockDim.x * gridDim.x;
//
//
//	//object plane origins
//	float x_origin = ObjParam.ObjX + xplane*RMobj[0];
//	float y_origin = ObjParam.ObjY + xplane*RMobj[3];
//	float z_origin = ObjParam.ObjZ + xplane*RMobj[6];
//
//	//camera detector pixel positions
//	float xcam = 0.0f;
//	float ycam = -(Det.NumTAxPixels/2.0f-1.0f)*Det.TAxPixDimMm-Det.TAxPixDimMm/2.0f + height*Det.TAxPixDimMm;
//	float zcam = -(Det.NumAxPixels/2.0f-1.0f)*Det.AxPixDimMm-Det.AxPixDimMm/2.0f + depth*Det.AxPixDimMm;
//
//	float xcam_rotshift = RMdet[0]*xcam + RMdet[1]*ycam + RMdet[2]*zcam;
//	float ycam_rotshift = RMdet[3]*xcam + RMdet[4]*ycam + RMdet[5]*zcam + CTGeom.Dy;
//	float zcam_rotshift = RMdet[6]*xcam + RMdet[7]*ycam + RMdet[8]*zcam + CTGeom.Dz;
//
//	//phosphor screen positions
//	float xpho = xcam_rotshift*CTGeom.M*RMpho[0] + ycam_rotshift*CTGeom.M*RMpho[1] + zcam_rotshift*CTGeom.M*RMpho[2] + CTGeom.R;
//	float ypho = xcam_rotshift*CTGeom.M*RMpho[3] + ycam_rotshift*CTGeom.M*RMpho[4] + zcam_rotshift*CTGeom.M*RMpho[5];
//	float zpho = xcam_rotshift*CTGeom.M*RMpho[6] + ycam_rotshift*CTGeom.M*RMpho[7] + zcam_rotshift*CTGeom.M*RMpho[8];
//
//
//	float t2 = __fdividef( RMobj[0]*(x_origin+CTGeom.Rf) + RMobj[3]*y_origin + RMobj[6]*z_origin, 
//							RMobj[0]*xpho + RMobj[3]*ypho + RMobj[6]*zpho);
//
//	float xv = -CTGeom.Rf + xpho*t2;
//	float yv = t2*ypho;
//	float zv = t2*zpho;
//
//	float xprojected_normal = __fdividef(xplane, (voxparam.Xdim-1.0f)*voxparam.XVoxSize);
//	float yprojected_normal = __fdividef(RMobj[1]*(xv-x_origin) + RMobj[4]*(yv-y_origin) + RMobj[7]*(zv-z_origin), 
//										(voxparam.Ydim-1.0f)*voxparam.YVoxSize);
//	float zprojected_normal = __fdividef(RMobj[2]*(xv-x_origin) + RMobj[5]*(yv-y_origin) + RMobj[8]*(zv-z_origin), 
//										(voxparam.Zdim-1.0f)*voxparam.ZVoxSize);
//	
//	//rotating the object (regular CT procedure, about the verical y axis);
//	float cos, sin;
//	sincosf(theta_rad, &cos, &sin);
//	float xr = xprojected_normal*cos + zprojected_normal*sin + 0.5f;
//	float yr = yprojected_normal + 0.5f;
//	float zr = -xprojected_normal*sin + zprojected_normal*cos + 0.5f;
//
//	md_FPimage[n] = md_FPimage[n] + tex3D(tex_int8, xr, yr, zr)*voxparam.XVoxSize*128;
//	
//}
//
//

//
//__global__ void sum_planes(float *md_FPimage, float *d_temp)
//{
//	int depth = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = depth + height * blockDim.x * gridDim.x;
//
//	md_FPimage[n] = md_FPimage[n] + d_temp[n];
//}
//
//__global__ void difference_squared(float *data, float *experimentImage)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = x + y * blockDim.x * gridDim.x;
//
//	data[n] = (data[n] - experimentImage[n])*(data[n] - experimentImage[n]);
//}
//
//
////backward projection
//__global__ void backprojection(float *result, CTGeometry CTGeom, Detector Det, ObjectParam ObjParam, 
//									VoxVolParam voxparam, float theta_rad, float *RMobj, float *RMpho, float *RMdet, float Nprojections)
//{
//	int width  = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	//find all the points on the object plane if the plane was perfect
//	float xobj = -(voxparam.Xdim/2.0f-1.0f)*voxparam.XVoxSize-voxparam.XVoxSize/2.0f + voxparam.XVoxSize*width;
//	float yobj = -(voxparam.Ydim/2.0f-1.0f)*voxparam.YVoxSize-voxparam.YVoxSize/2.0f + voxparam.YVoxSize*height;
//	float zobj = -(voxparam.Zdim/2.0f-1.0f)*voxparam.ZVoxSize-voxparam.ZVoxSize/2.0f + voxparam.ZVoxSize*depth;
//	
//	//CT rotation
//	float sin, cos;
//	sincosf(theta_rad, &sin, &cos);
//	float xr = xobj*cos + yobj*sin;
//	float yr = yobj;
//	float zr = -xobj*sin + zobj*cos;
//	
//	//find all the points on the object plane if the plane was rotated and shifted, then subtracted by the xray source point
//	float x_vector = RMobj[0]*xr + RMobj[1]*yr + RMobj[2]*zr + ObjParam.ObjX + CTGeom.Rf;
//	float y_vector = RMobj[3]*xr + RMobj[4]*yr + RMobj[5]*zr + ObjParam.ObjY; 
//	float z_vector = RMobj[6]*xr + RMobj[7]*yr + RMobj[8]*zr + ObjParam.ObjZ;
//
//	float t = __fdividef( CTGeom.R*RMpho[0], RMpho[0]*x_vector + RMpho[3]*y_vector + RMpho[6]*z_vector );
//
//	//find all points on the detector plane that intersects with the line that goes from xray source through the points on the object plane;
//	float pt_x = (x_vector + CTGeom.Rf)*t - CTGeom.R;
//	float pt_y = y_vector*t-CTGeom.Dy;
//	float pt_z = z_vector*t-CTGeom.Dz;
//
//	//project all the points to the rotated y and z axis on the detector so it's back to its original coordinate system
//	float ydet = pt_x*RMdet[1] + pt_y*RMdet[4] + pt_z*RMdet[7];
//	float zdet = pt_x*RMdet[2] + pt_y*RMdet[5] + pt_z*RMdet[8];
//
//	float ydet_normal = __fdividef(ydet, (Det.NumTAxPixels-1.0f)*Det.TAxPixDimMm*CTGeom.M) + 0.5f;
//	float zdet_normal = __fdividef(zdet, (Det.NumAxPixels-1.0f)*Det.AxPixDimMm*CTGeom.M) + 0.5f;
//
//	float temp = tex2D(tex2, zdet_normal, ydet_normal);
//	result[n] = result[n] + temp/Nprojections;
//}
//
//
//
//
////sums volume, for backprojection, to create the final volume result
//__global__ void sum_volume(float *md_bpvolume, float *temp_volume, ScanParameters a_ScanParam)
//{
//
//	int width = threadIdx.x + blockIdx.x * blockDim.x; 
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	md_bpvolume[n] = md_bpvolume[n] + temp_volume[n]/(float)a_ScanParam.NumProj;
//
//}
//
////sums up the volume into a 2D image, to generate projection image result
//__global__ void sum_texture_volume(float *d_tempTextureVolume, float *md_FPimage, float m_dx)
//{
//	
//	extern __shared__ float cache[];
//
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * blockDim.x * gridDim.x;
//	
//	int cacheIndex = threadIdx.x;
//	cache[cacheIndex] = d_tempTextureVolume[offset];
//	
//	__syncthreads();
//	int i = blockDim.x/2;
//	while (i != 0)
//	{
//		if (cacheIndex < i)
//			cache[cacheIndex] = cache[cacheIndex + i] + cache[cacheIndex];
//		__syncthreads();
//		i = i/2;
//	}
//
//	if (cacheIndex == 0)
//	{
//		md_FPimage[blockIdx.x + blockIdx.y*gridDim.x] = cache[0]*m_dx;
//	}
//
//}
//
//
//__global__ void extract_2D_texture(float *temp_volume, CTGeometry a_CTGeom, float theta_rad, 
//									Detector a_Det, VoxVolParam m_objectvox, float *RotationMatrix)
//{
//	int width = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//	
//	//original x,y,z linspace index for the object cube
//	float xn = -m_objectvox.XVoxSize/2.0f - (blockDim.x*gridDim.x/2.0f-1.0f)*m_objectvox.XVoxSize + m_objectvox.XVoxSize*width;
//	float yn = -m_objectvox.YVoxSize/2.0f - (blockDim.y*gridDim.y/2.0f-1.0f)*m_objectvox.YVoxSize + m_objectvox.YVoxSize*height;
//	float zn = -m_objectvox.ZVoxSize/2.0f - (blockDim.z*gridDim.z/2.0f-1.0f)*m_objectvox.ZVoxSize + m_objectvox.ZVoxSize*depth;
//
//	//object cube after correcting for rotation deviation
//	float xr = xn*RotationMatrix[0] + yn*RotationMatrix[1] + zn*RotationMatrix[2];
//	float yr = xn*RotationMatrix[3] + yn*RotationMatrix[4] + zn*RotationMatrix[5];
//	float zr = xn*RotationMatrix[6] + yn*RotationMatrix[7] + zn*RotationMatrix[8];
//
//	float sin_theta, cos_theta;
//	sincosf(theta_rad, &sin_theta, &cos_theta);
//
//	//rotating the object volume
//	float x = xr*cos_theta + zr*sin_theta;
//	float y = yr;
//	float z = -xr*sin_theta + zr*cos_theta;
//
//	//calculate the corresponding location on the detector, taking account for object rotation deviation
//	float yd = __fdividef( y, a_CTGeom.Rf + x) * a_CTGeom.R;
//	float zd = __fdividef( z, a_CTGeom.Rf + x) * a_CTGeom.R;
//
//	//detector rotation deviation 
//	float sin_x, cos_x, sin_y, cos_y, sin_z, cos_z;
//	sincosf(a_CTGeom.xangle*M_PI/180.0f, &sin_x, &cos_x);
//	sincosf(a_CTGeom.yangle*M_PI/180.0f, &sin_y, &cos_y);
//	sincosf(a_CTGeom.zangle*M_PI/180.0f, &sin_z, &cos_z);
//
//	//account for detector rotation deviation
//	float zd_rx = zd*cos_x - yd*sin_x;
//	float yd_rx = zd*sin_x + yd*cos_x;
//
//	float ydr = __fdividef(a_CTGeom.R, a_CTGeom.R*cos_z + yd_rx*sin_z)*yd_rx;
//	float zdr = __fdividef(a_CTGeom.R, a_CTGeom.R*cos_y + zd_rx*sin_y)*zd_rx;
//
//	//normalizing the detector and make it go from 0 to 1;
//	float norm_yd = __fdividef(ydr, (a_Det.NumTAxPixels-1.0f)*a_Det.TAxPixDimMm*a_CTGeom.M) + 0.5f;
//	float norm_zd = __fdividef(zdr, (a_Det.NumAxPixels-1.0f)*a_Det.AxPixDimMm*a_CTGeom.M) + 0.5f;
//
//	temp_volume[n] = tex2D(tex2, norm_yd, norm_zd);
//}
//
//
//__global__ void rotate_volume_3D(float *rotatedvolume, float *rotationmatrix)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int z = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = x + y * blockDim.x * gridDim.x +
//				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	float xn = x / ( (float)blockDim.x * (float)blockDim.x ) - 0.5f;
//	float yn = y / ( (float)blockDim.y * (float)blockDim.y ) - 0.5f;
//	float zn = z / ( (float)blockDim.z * (float)blockDim.z ) - 0.5f;
//
//	float xr = xn*rotationmatrix[0] + yn*rotationmatrix[1] + zn*rotationmatrix[2] + 0.5f;
//	float yr = xn*rotationmatrix[3] + yn*rotationmatrix[4] + zn*rotationmatrix[5] + 0.5f;
//	float zr = xn*rotationmatrix[6] + yn*rotationmatrix[7] + zn*rotationmatrix[8] + 0.5f;
//
//	rotatedvolume[n] = tex3D(tex, xr, yr, zr);
//}



////sums up the volume into a 2D image, to generate projection image result
//__global__ void sum_texture_volume(float *d_tempTextureVolume, float *md_FPimage, float m_dx)
//{
//	
//	extern __shared__ float cache[];
//
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * blockDim.x * gridDim.x;
//	
//	int cacheIndex = threadIdx.x;
//	cache[cacheIndex] = d_tempTextureVolume[offset];
//	
//	__syncthreads();
//	int i = blockDim.x/2;
//	while (i != 0)
//	{
//		if (cacheIndex < i)
//			cache[cacheIndex] = cache[cacheIndex + i] + cache[cacheIndex];
//		__syncthreads();
//		i = i/2;
//	}
//
//	if (cacheIndex == 0)
//	{
//		md_FPimage[blockIdx.x + blockIdx.y*gridDim.x] = cache[0]*m_dx;
//	}
//
//}
//

//__global__ void extract_2D_texture(float *temp_volume, CTGeometry a_CTGeom, float theta_rad, 
//									Detector a_Det, VoxVolParam m_objectvox, float *RotationMatrix)
//{
//	int width = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//	
//	//original x,y,z linspace index for the object cube
//	float xn = -m_objectvox.XVoxSize/2.0f - (blockDim.x*gridDim.x/2.0f-1.0f)*m_objectvox.XVoxSize + m_objectvox.XVoxSize*width;
//	float yn = -m_objectvox.YVoxSize/2.0f - (blockDim.y*gridDim.y/2.0f-1.0f)*m_objectvox.YVoxSize + m_objectvox.YVoxSize*height;
//	float zn = -m_objectvox.ZVoxSize/2.0f - (blockDim.z*gridDim.z/2.0f-1.0f)*m_objectvox.ZVoxSize + m_objectvox.ZVoxSize*depth;
//
//	//object cube after correcting for rotation deviation
//	float xr = xn*RotationMatrix[0] + yn*RotationMatrix[1] + zn*RotationMatrix[2];
//	float yr = xn*RotationMatrix[3] + yn*RotationMatrix[4] + zn*RotationMatrix[5];
//	float zr = xn*RotationMatrix[6] + yn*RotationMatrix[7] + zn*RotationMatrix[8];
//
//	float sin_theta, cos_theta;
//	sincosf(theta_rad, &sin_theta, &cos_theta);
//
//	//rotating the object volume
//	float x = xr*cos_theta + zr*sin_theta;
//	float y = yr;
//	float z = -xr*sin_theta + zr*cos_theta;
//
//	//calculate the corresponding location on the detector, taking account for object rotation deviation
//	float yd = __fdividef( y, a_CTGeom.Rf + x) * a_CTGeom.R;
//	float zd = __fdividef( z, a_CTGeom.Rf + x) * a_CTGeom.R;
//
//	//detector rotation deviation 
//	float sin_x, cos_x, sin_y, cos_y, sin_z, cos_z;
//	sincosf(a_CTGeom.xangle*M_PI/180.0f, &sin_x, &cos_x);
//	sincosf(a_CTGeom.yangle*M_PI/180.0f, &sin_y, &cos_y);
//	sincosf(a_CTGeom.zangle*M_PI/180.0f, &sin_z, &cos_z);
//
//	//account for detector rotation deviation
//	float zd_rx = zd*cos_x - yd*sin_x;
//	float yd_rx = zd*sin_x + yd*cos_x;
//
//	float ydr = __fdividef(a_CTGeom.R, a_CTGeom.R*cos_z + yd_rx*sin_z)*yd_rx;
//	float zdr = __fdividef(a_CTGeom.R, a_CTGeom.R*cos_y + zd_rx*sin_y)*zd_rx;
//
//	//normalizing the detector and make it go from 0 to 1;
//	float norm_yd = __fdividef(ydr, (a_Det.NumTAxPixels-1.0f)*a_Det.TAxPixDimMm*a_Det.mag) + 0.5f;
//	float norm_zd = __fdividef(zdr, (a_Det.NumAxPixels-1.0f)*a_Det.AxPixDimMm*a_Det.mag) + 0.5f;
//
//	temp_volume[n] = tex2D(tex2, norm_yd, norm_zd);
//}
//
//
//__global__ void rotate_volume_3D(float *rotatedvolume, float *rotationmatrix)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int z = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = x + y * blockDim.x * gridDim.x +
//				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	float xn = x / ( (float)blockDim.x * (float)blockDim.x ) - 0.5f;
//	float yn = y / ( (float)blockDim.y * (float)blockDim.y ) - 0.5f;
//	float zn = z / ( (float)blockDim.z * (float)blockDim.z ) - 0.5f;
//
//	float xr = xn*rotationmatrix[0] + yn*rotationmatrix[1] + zn*rotationmatrix[2] + 0.5f;
//	float yr = xn*rotationmatrix[3] + yn*rotationmatrix[4] + zn*rotationmatrix[5] + 0.5f;
//	float zr = xn*rotationmatrix[6] + yn*rotationmatrix[7] + zn*rotationmatrix[8] + 0.5f;
//
//	rotatedvolume[n] = tex3D(tex, xr, yr, zr);
//}

//__global__ void extract_3D_texture(float *d_tempTextureVolume, float theta_rad, VoxVolParam m_objectvox, 
//										CTGeometry a_CTGeom, Detector a_Det, float *RotationMatrix)
//{
//	//iterate all threads index
//	int width = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	//Length of voxel in the x dimention (along the longest axis)
//	float Lx = ( (float)m_objectvox.Xdim-1.0f )*(float)m_objectvox.XVoxSize;
//	float xn = -m_objectvox.XVoxSize/2.0f-Lx + m_objectvox.XVoxSize*width; //the z axis iterated out, it's like linspace
//
//	float zn = -a_Det.AxPixDimMm*a_Det.mag/2.0f-(a_Det.NumAxPixels/2.0f-1.0f)*a_Det.AxPixDimMm*a_Det.mag + a_Det.AxPixDimMm*a_Det.mag*depth; //x axis iterated out, linspace
//	float yn = -a_Det.TAxPixDimMm*a_Det.mag/2.0f-(a_Det.NumTAxPixels/2.0f-1.0f)*a_Det.TAxPixDimMm*a_Det.mag + a_Det.TAxPixDimMm*a_Det.mag*height; // y axis iterated out, linspace
//	
//
//	float sin_x, cos_x, sin_y, cos_y, sin_z, cos_z;
//	sincosf(a_CTGeom.xangle*M_PI/180.0f, &sin_x, &cos_x);
//	sincosf(a_CTGeom.yangle*M_PI/180.0f, &sin_y, &cos_y);
//	sincosf(a_CTGeom.zangle*M_PI/180.0f, &sin_z, &cos_z);
//
//	float z_rx = zn*cos_x - yn*sin_x;
//	float y_rx = zn*sin_x + yn*cos_x;
//
//	float znr = __fdividef(a_CTGeom.R, a_CTGeom.R*cos_z + y_rx*sin_z)*y_rx;
//	float ynr = __fdividef(a_CTGeom.R, a_CTGeom.R*cos_y + z_rx*sin_y)*z_rx;
//
//	float norm_voxwidth =  __fdividef( xn, Lx ); //normalize x
//	float voxheight = __fdividef(ynr, a_CTGeom.R) * (a_CTGeom.Rf - xn);	
//	float norm_voxheight = __fdividef(voxheight, ((float)m_objectvox.Ydim - 1.0f)*(float)m_objectvox.YVoxSize);
//
//	float voxdepth = __fdividef(znr, a_CTGeom.R) * (a_CTGeom.Rf - xn);
//	float norm_voxdepth = __fdividef(voxdepth, ((float)m_objectvox.Zdim - 1.0f)*(float)m_objectvox.ZVoxSize);
//
//	float sin_theta, cos_theta;
//	sincosf(theta_rad, &sin_theta, &cos_theta);
//
//	float norm_voxwidthR = norm_voxwidth*cos_theta + norm_voxdepth*sin_theta;//x
//	float norm_voxheightR = norm_voxheight;//y
//	float norm_voxdepthR = -norm_voxwidth*sin_theta + norm_voxdepth*cos_theta;//z
//
//	float norm_voxwidthRR =		norm_voxwidthR*RotationMatrix[0] + norm_voxheightR*RotationMatrix[1] + norm_voxdepthR*RotationMatrix[2] + 0.5f;
//	float norm_voxheightRR =	norm_voxwidthR*RotationMatrix[3] + norm_voxheightR*RotationMatrix[4] + norm_voxdepthR*RotationMatrix[5] + 0.5f;
//	float norm_voxdepthRR =		norm_voxwidthR*RotationMatrix[6] + norm_voxheightR*RotationMatrix[7] + norm_voxdepthR*RotationMatrix[8] + 0.5f;
//
//	d_tempTextureVolume[n] = tex3D(tex, norm_voxwidthRR, norm_voxheightRR, norm_voxdepthRR);
//}
//

//__global__ void extract_2D_texture(float *temp_volume, CTGeometry a_CTGeom, float theta_rad, Detector a_Det, VoxVolParam m_objectvox)
//{
//	int width = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	float cos_theta;
//	float sin_theta;
//	sincosf(theta_rad, &cos_theta, &sin_theta);
//
//	//blockDim.x * gridDim.x = Zdim
//	//blockDim.y * gridDim.y = Ydim
//	//blockDim.z * gridDim.z = Xdim
//	float LdetY = (a_Det.NumTAxPixels-1.0f)*a_Det.TAxPixDimMm/2.0f;
//	float LdetX = (a_Det.NumAxPixels-1.0f)*a_Det.AxPixDimMm/2.0f;
//
//	float LvoxZ = fmaxf(LdetY, LdetX)/a_CTGeom.R*a_CTGeom.Rf;
//	float zn = -LvoxZ + 2.0f*LvoxZ/(blockDim.x*gridDim.x-1)*width;
//
//	float LvoxY = LdetY/a_CTGeom.R*(a_CTGeom.Rf-LvoxZ);
//	float LvoxX = LdetX/a_CTGeom.R*(a_CTGeom.Rf-LvoxZ);
//
//	float xn = -LvoxX + 2.0f*LvoxX/(blockDim.x*gridDim.x-1.0f)*depth;
//	float yn = -LvoxY + 2.0f*LvoxY/(blockDim.y*gridDim.y-1.0f)*height;
//
//	float x = xn*cos_theta + zn*sin_theta;
//	float y = yn;
//	float z = -xn*sin_theta + zn*cos_theta;
//
//	float yd = __fdividef( y, a_CTGeom.Rf + z) * a_CTGeom.R;
//	float xd = __fdividef( x, a_CTGeom.Rf + z) * a_CTGeom.R;
//
//	float norm_yd = __fdividef(yd, (a_Det.NumTAxPixels-1.0f)*a_Det.TAxPixDimMm) + 0.5f;
//	float norm_xd = __fdividef(xd, (a_Det.NumAxPixels-1.0f)*a_Det.AxPixDimMm) + 0.5f;
//
//	temp_volume[n] = tex2D(tex2, norm_xd, norm_yd);
//}



//__global__ void extract_2D_texture(float *temp_volume, float *md_x, float *md_y, float *md_z,
//									CTGeometry a_CTGeom, float theta_rad, Detector a_Det, VoxVolParam a_VoxParam)
//{
//	int width = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//
//	float cos_theta;
//	float sin_theta;
//	sincosf(theta_rad, &cos_theta, &sin_theta);
//
//	float x = md_x[depth]*cos_theta + md_z[width]*sin_theta;
//	float y = md_y[height];
//	float z = -md_x[depth]*sin_theta + md_z[width]*cos_theta;
//
//	float yd = __fdividef( y, a_CTGeom.Rf + z) * a_CTGeom.R;
//	float xd = __fdividef( x, a_CTGeom.Rf + z) * a_CTGeom.R;
//
//	float norm_yd = __fdividef(yd, (a_Det.NumTAxPixels-1.0f)*a_Det.TAxPixDimMm) + 0.5f;
//	float norm_xd = __fdividef(xd, (a_Det.NumAxPixels-1.0f)*a_Det.AxPixDimMm) + 0.5f;
//
//	temp_volume[n] = tex2D(tex2, norm_xd, norm_yd);
//}




//__global__ void extract_texture_info(float *d_tempTextureVolume, float *md_zplane, float *md_detX, float *md_detY, float theta_rad,
//								VoxVolParam a_VoxParam, CTGeometry a_CTGeom, Detector a_Det, float *RotationMatrix)
//{
//	int width = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth = threadIdx.z + blockIdx.z * blockDim.z;
//
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	float norm_voxZwidth =  __fdividef(md_zplane[width], ((float)a_VoxParam.Zdim - 1.0f)*(float)a_VoxParam.ZVoxSize);
//	
//	float voxYheight = __fdividef(md_detY[height], a_CTGeom.R) * (a_CTGeom.Rf - md_zplane[width]);	
//	float norm_voxYheight = __fdividef(voxYheight, ((float)a_VoxParam.Ydim - 1.0f)*(float)a_VoxParam.YVoxSize);
//	
//	float voxXdepth = __fdividef(md_detX[depth], a_CTGeom.R) * (a_CTGeom.Rf - md_zplane[width]);
//	float norm_voxXdepth = __fdividef(voxXdepth, ((float)a_VoxParam.Xdim - 1.0f)*(float)a_VoxParam.XVoxSize);
//	
//	float cos_theta;
//	float sin_theta;
//
//	sincosf(theta_rad, &cos_theta, &sin_theta);
//
//	float norm_voxZwidthR = norm_voxZwidth*cos_theta + norm_voxXdepth*sin_theta;//z
//	float norm_voxYheightR = norm_voxYheight;//y
//	float norm_voxXdepthR = -norm_voxZwidth*sin_theta + norm_voxXdepth*cos_theta;//x
//
//	float norm_voxZwidthRR = norm_voxZwidthR*RotationMatrix[0] + norm_voxYheightR*RotationMatrix[1] + norm_voxXdepthR*RotationMatrix[2] + 0.5f;
//	float norm_voxYheightRR = norm_voxZwidthR*RotationMatrix[3] + norm_voxYheightR*RotationMatrix[4] + norm_voxXdepthR*RotationMatrix[5] + 0.5f;
//	float norm_voxXdepthRR = norm_voxZwidthR*RotationMatrix[6] + norm_voxYheightR*RotationMatrix[7] + norm_voxXdepthR*RotationMatrix[8] + 0.5f;
//
//	d_tempTextureVolume[n] = tex3D(tex, norm_voxZwidthRR, norm_voxYheightRR, norm_voxXdepthRR);
//	
//}












//__global__ void sum_texture_volume2(float *d_tempTextureVolume, float *md_FPimage, float m_dx)
//{
//	
//	extern __shared__ float cache[];
//
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * blockDim.x * gridDim.x;
//	
//	int cacheIndex = threadIdx.x;
//	cache[cacheIndex] = d_tempTextureVolume[offset];
//	
//	__syncthreads();
//	int i = blockDim.x/2;
//	while (i != 0)
//	{
//		if (cacheIndex < i)
//			cache[cacheIndex] = cache[cacheIndex + i] + cache[cacheIndex];
//		__syncthreads();
//		i = i/2;
//	}
//
//	if (cacheIndex == 0)
//	{
//		md_FPimage[blockIdx.x + blockIdx.y*gridDim.x] = cache[0]*m_dx;
//	}
//
//}


//__global__ void ForwardProject(float *dev_out3D, float *dev_xplane, float *dev_detY, float *dev_detZ, 
//						  float xdet, float xsource, VoxVolParam xyphantom, float sin_theta, float cos_theta)
//{
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int z = threadIdx.z + blockIdx.z * blockDim.z;
//	int n = x + y * blockDim.x * gridDim.x +
//				 z * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	float normalizedX = __fdividef(dev_xplane[x], (float)xyphantom.Xdim*xyphantom.XVoxSize);
//	float normalizedY = __fdividef(__fdividef(dev_detY[y], (xdet-xsource)) * (dev_xplane[x]-xsource), (float)xyphantom.Ydim*xyphantom.YVoxSize);
//	float normalizedZ = __fdividef(__fdividef(dev_detZ[z], (xdet-xsource)) * (dev_xplane[x]-xsource), (float)xyphantom.Zdim*xyphantom.ZVoxSize);
//
//	float xplaneR = normalizedX*cos_theta - normalizedY*sin_theta + 0.5f;
//	float yplaneR = normalizedX*sin_theta + normalizedY*cos_theta + 0.5f;
//	float zplaneR = normalizedZ + 0.5f;
//	
//	dev_out3D[n] = tex3D(tex, xplaneR, yplaneR, zplaneR);
//
//}
//
//
//__global__ void sum3D(float *dev_out3D, float *dev_out2D, float dx)
//{
//	
//	extern __shared__ float cache[];
//
//	int x = threadIdx.x + blockIdx.x * blockDim.x;
//	int y = threadIdx.y + blockIdx.y * blockDim.y;
//	int offset = x + y * blockDim.x * gridDim.x;
//	
//	int cacheIndex = threadIdx.x;
//	cache[cacheIndex] = dev_out3D[offset];
//	
//	__syncthreads();
//	int i = blockDim.x/2;
//	while (i != 0)
//	{
//		if (cacheIndex < i)
//			cache[cacheIndex] = cache[cacheIndex + i] + cache[cacheIndex];
//		__syncthreads();
//		i = i/2;
//	}
//
//	if (cacheIndex == 0)
//	{
//		dev_out2D[blockIdx.x + blockIdx.y*gridDim.x] = cache[0]*dx;
//	}
//	
//}
//

////forward projection
//__global__ void forwardproject(float *md_FPimage, CTGeometry CTGeom, 
//									Detector Det, ObjectParam ObjParam, 
//									VoxVolParam voxparam, float xplane, 
//									float theta_rad, float *RMobj, float *RMdet)
//{
//	int depth = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int n = depth + height * blockDim.x * gridDim.x;
//
//
//	//object plane origins
//	float x_origin = ObjParam.ObjX + xplane*RMobj[0];
//	float y_origin = ObjParam.ObjY + xplane*RMobj[3];
//	float z_origin = ObjParam.ObjZ + xplane*RMobj[6];
//
//	//unshifted detector pixel locations
//	float xd = 0.0f;
//	float yd = -(Det.NumTAxPixels/2.0f-1.0f)*Det.TAxPixDimMm*Det.mag-Det.TAxPixDimMm*Det.mag/2.0f + height*Det.TAxPixDimMm*Det.mag;
//	float zd = -(Det.NumAxPixels/2.0f-1.0f)*Det.AxPixDimMm*Det.mag-Det.AxPixDimMm*Det.mag/2.0f + depth*Det.AxPixDimMm*Det.mag;
//
//	//rotated and shifted detector pixel locations to source location (sd = source to detector)
//	float xd_rot = RMdet[0]*xd + RMdet[1]*yd + RMdet[2]*zd;
//	float yd_rot = RMdet[3]*xd + RMdet[4]*yd + RMdet[5]*zd;
//	float zd_rot = RMdet[6]*xd + RMdet[7]*yd + RMdet[8]*zd;
//
//	float xd_rotshift = xd_rot + CTGeom.R-CTGeom.Rf;
//	float yd_rotshift = yd_rot + CTGeom.Dy;
//	float zd_rotshift = zd_rot + CTGeom.Dz; 
//
//	float x_sd = xd_rotshift + CTGeom.Rf;
//	float y_sd = yd_rotshift;
//	float z_sd = zd_rotshift;
//
//	float t2 = __fdividef( RMobj[0]*(x_origin+CTGeom.Rf) + RMobj[3]*y_origin + RMobj[6]*z_origin, 
//							RMobj[0]*x_sd + RMobj[3]*y_sd + RMobj[6]*z_sd);
//
//	float xv = -CTGeom.Rf + x_sd*t2;
//	float yv = t2*y_sd;
//	float zv = t2*z_sd;
//
//	float xprojected_normal = __fdividef(xplane, (voxparam.Xdim-1.0f)*voxparam.XVoxSize);
//	float yprojected_normal = __fdividef(RMobj[1]*(xv-x_origin) + RMobj[4]*(yv-y_origin) + RMobj[7]*(zv-z_origin), 
//										(voxparam.Ydim-1.0f)*voxparam.YVoxSize);
//	float zprojected_normal = __fdividef(RMobj[2]*(xv-x_origin) + RMobj[5]*(yv-y_origin) + RMobj[8]*(zv-z_origin), 
//										(voxparam.Zdim-1.0f)*voxparam.ZVoxSize);
//	
//	//rotating the object (regular CT procedure, about the verical y axis);
//	float cos, sin;
//	sincosf(theta_rad, &cos, &sin);
//	float xr = xprojected_normal*cos + zprojected_normal*sin + 0.5f;
//	float yr = yprojected_normal + 0.5f;
//	float zr = -xprojected_normal*sin + zprojected_normal*cos + 0.5f;
//
//	md_FPimage[n] = md_FPimage[n] + tex3D(tex, xr, yr, zr)*voxparam.XVoxSize;
//	
//}

////backward projection
//__global__ void backprojection(float *result, CTGeometry CTGeom, Detector Det, ObjectParam ObjParam, 
//									VoxVolParam voxparam, float theta_rad, float *RMobj, float *RMdet, float Nprojections)
//{
//	int width  = threadIdx.x + blockIdx.x * blockDim.x;
//	int height = threadIdx.y + blockIdx.y * blockDim.y;
//	int depth  = threadIdx.z + blockIdx.z * blockDim.z;
//	int n = width + height * blockDim.x * gridDim.x +
//				 depth * blockDim.x * gridDim.x * blockDim.y * gridDim.y;
//
//	//find all the points on the object plane if the plane was perfect
//	float xobj = -(voxparam.Xdim/2.0f-1.0f)*voxparam.XVoxSize-voxparam.XVoxSize/2.0f + voxparam.XVoxSize*width;
//	float yobj = -(voxparam.Ydim/2.0f-1.0f)*voxparam.YVoxSize-voxparam.YVoxSize/2.0f + voxparam.YVoxSize*height;
//	float zobj = -(voxparam.Zdim/2.0f-1.0f)*voxparam.ZVoxSize-voxparam.ZVoxSize/2.0f + voxparam.ZVoxSize*depth;
//	
//	//CT rotation
//	float sin, cos;
//	sincosf(theta_rad, &sin, &cos);
//	float xr = xobj*cos + yobj*sin;
//	float yr = yobj;
//	float zr = -xobj*sin + zobj*cos;
//	
//	//find all the points on the object plane if the plane was rotated and shifted, then subtracted by the xray source point
//	float x_vector = RMobj[0]*xr + RMobj[1]*yr + RMobj[2]*zr + ObjParam.ObjX + CTGeom.Rf;
//	float y_vector = RMobj[3]*xr + RMobj[4]*yr + RMobj[5]*zr + ObjParam.ObjY; 
//	float z_vector = RMobj[6]*xr + RMobj[7]*yr + RMobj[8]*zr + ObjParam.ObjZ;
//
//	float t = __fdividef( CTGeom.R*RMdet[0]+RMdet[3]*CTGeom.Dy+RMdet[6]*CTGeom.Dz, 
//							RMdet[0]*x_vector + RMdet[3]*y_vector + RMdet[6]*z_vector );
//
//	//find all points on the detector plane that intersects with the line that goes from xray source through the points on the object plane;
//	float pt_x = x_vector*t + CTGeom.R;
//	float pt_y = y_vector*t - CTGeom.Dy;
//	float pt_z = z_vector*t - CTGeom.Dz;
//
//	//project all the points to the rotated y and z axis on the detector so it's back to its original coordinate system
//	float ydet = pt_x*RMdet[1] + pt_y*RMdet[4] + pt_z*RMdet[7];
//	float zdet = pt_x*RMdet[2] + pt_y*RMdet[5] + pt_z*RMdet[8];
//
//	float ydet_normal = __fdividef(ydet, (Det.NumTAxPixels-1.0f)*Det.TAxPixDimMm*CTGeom.M) + 0.5f;
//	float zdet_normal = __fdividef(zdet, (Det.NumAxPixels-1.0f)*Det.AxPixDimMm*CTGeom.M) + 0.5f;
//
//	float temp = tex2D(tex2, zdet_normal, ydet_normal);
//	result[n] = result[n] + temp/Nprojections;
//}
//
