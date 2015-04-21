//Written by Helen Fan


#pragma once
#ifndef SIDDON_DEVICE_KERNELSMLEM_CUH
#define SIDDON_DEVICE_KERNELSMLEM_CUH

#define ALPHA_OUT_BOUND		1
#define INDEX_OUT_BOUND		0
#define CUDART_NAN_F		__int_as_float(0x7FFFFFFF)

struct Planes 
{
	float x;
	float y;
	float z;

	void fill(float x_value, float y_value, float z_value)
	{
		x = x_value;
		y = y_value;
		z = z_value;
	}
};


struct XraySystem
{
	CalibrationParam CalParam;
	Detector Det;
	VoxVolParam voxparam;
	Planes P_initial;
	Planes P_final;
	Planes P_delta;
	Planes XraySource;

	void create(CalibrationParam p, Detector d, VoxVolParam v, Planes P_i, Planes P_f, Planes P_d, Planes X)
	{
		CalParam = p;
		Det = d;
		voxparam = v;
		P_initial = P_i;
		P_final = P_f;
		P_delta = P_d;
		XraySource = X;

	}
};

__device__ void bound_index(float &index_min, float &index_max, unsigned int &d_mask)
{
	if ( (index_max - index_min) < 0 && index_min <= 1 && index_max <= 1)
	{
		index_min = 0;
		index_max = 0;
		d_mask = CUDART_NAN_F;
	}
}

__device__ void bound_index2(float &index_min, float &index_max, unsigned int &d_mask)
{
	if ( (index_max - index_min) < 0 || index_min < 1 || index_max < 1 )
	{
		index_min = CUDART_NAN_F;
		index_max = CUDART_NAN_F;
		d_mask = CUDART_NAN_F;

	}
}

__device__ void calc_alpha_1N( float point1, float point2, float plane1, float plane2, float2 &alpha)
{
	if( (point2-point1) != 0 )
	{
		alpha.x = __fdividef( plane1 - point1, point2 - point1 ); //1
		alpha.y = __fdividef( plane2 - point1, point2 - point1 ); //N

	}
	else
	{
		alpha.x = CUDART_NAN_F; //1, make sure it would not be selected as alpha_max in fmaxf()
		alpha.y = CUDART_NAN_F; //N make sure it would not be selected as alpha_min, fminf()
	}

}


__device__ unsigned int calc_mask( float alpha_min, float alpha_max)
{
	if (alpha_max <= alpha_min)
	{
		return CUDART_NAN_F;
	}
	else
	{
		return 1;
	}
}

__device__ float delta_alpha(float voxel_size, float p2, float p1, unsigned int delta_index, float alpha_i, float alpha_f)
{
	float d_alpha = fabsf( __fdividef( voxel_size, p2 - p1 ) );

	if ( alpha_i != CUDART_NAN_F && alpha_f != CUDART_NAN_F )
	{
		return(d_alpha);
	}
	else
	{
		return(CUDART_NAN_F);
	}
	
}

__device__ float get_alpha( float point1, float point2, float plane1, float ijk_min, float ijk_max, float delta_plane)
{
	float plane_ijk;
	if ( ((point2 - point1) > 0.0f) && (ijk_min > 0) )
	{
		plane_ijk = ( plane1 + delta_plane * (ijk_min - 1.0f) );  //original code
		return (__fdividef( (plane_ijk - point1), (point2 - point1) ) );  //original code
	}
	else if ( ((point2 - point1) < 0.0f) && (ijk_min > 0) )
	{
		plane_ijk = ( plane1 + delta_plane * (ijk_max - 1.0f) ); //original code
		return (__fdividef( (plane_ijk - point1), (point2 - point1) ) );  //original code
	}
	else
	{
		return (CUDART_NAN_F);
	}
}

__device__ float alpha_bound(float alpha, float alpha_min, float alpha_i, float alpha_f, float alpha_max)
{
	if (alpha >= alpha_i && alpha <= alpha_f)
	{
		return (alpha);
	}
	else if (alpha > alpha_f)
	{
		return(alpha_max);
	}
	else
	{
		return(CUDART_NAN_F);
	}
}


__device__ float find_index(float index, float alpha_min, float alpha_max, float point1, float point2, float plane1, float delta_plane)
{
	if (index > 0.0f)
	{
		float low, high;
		low =  floorf(index);
		high = ceilf(index);

		float plane_low, plane_high;
		float alpha_low, alpha_high;

		plane_low = ( plane1 + delta_plane*( low-1.0f) );
		plane_high = ( plane1 + delta_plane*( high-1.0f) );

		alpha_low = __fdividef( plane_low-point1, point2-point1 );
		alpha_high = __fdividef( plane_high-point1, point2-point1 );
		
		if ( alpha_low >= alpha_min && alpha_low <= alpha_max )
		{
			return (low);
		}
		else if ( alpha_high >= alpha_min && alpha_high <= alpha_max )
		{
			return (high);
		}
		else
		{
			return CUDART_NAN_F;
		}
	}
	else
	{
		return CUDART_NAN_F;
	}
}



__device__ void index_check(unsigned int &index_min, unsigned int &index_max, int &xyz_mask)
{
	if (index_max < index_min || index_min == 0)
	{
		index_max = CUDART_NAN_F;
		index_min = CUDART_NAN_F;
		xyz_mask = CUDART_NAN_F;
	}
	else
	{
		index_max = index_max;
		index_min = index_min;
	}
}

__device__ void index_check2(unsigned int &index_min, unsigned int &index_max)
{
	if (index_max < index_min || index_min == 0)
	{
		index_max = CUDART_NAN_F;
		index_min = CUDART_NAN_F;
	}
	else
	{
		index_max = index_max;
		index_min = index_min;
	}
}

__device__ float alpha_float(unsigned int min, unsigned max, float p1, float p2, float plane1, float delta_plane)
{
	float plane;

	if ( (p2-p1) > 0 && (max-min) > 0 )
	{
		plane = plane1 + delta_plane * ((float)min - 1.0f );
		return(__fdividef( plane - p1, p2 - p1 ) );		
	}
	else if ( (p2 - p1) < 0 && (max-min) > 0 )
	{
		plane = plane1 + delta_plane * ((float)max - 1.0f );
		return(__fdividef( plane - p1, p2 - p1 ));
	}
	else
	{
		return(CUDART_NAN_F);
	}

}


__device__ unsigned int siddon_calc_indices(float &alpha_i, float &alpha_f, unsigned int &d_mask, float point1, float point2, float plane1, float plane2, float Nplanes, float alpha_min, float alpha_max, float delta_plane)
{
	float2 index;
	if ( (point2 - point1) > 0 )
	{
		index.x = Nplanes - __fdividef( (plane2 - alpha_min*(point2-point1) - point1), delta_plane ); //index min
		index.y = 1.0f + __fdividef( (point1 + alpha_max*(point2-point1) - plane1), delta_plane ); //index max
	}
	else if ( (point2 - point1) < 0 )
	{
		index.x = Nplanes - __fdividef( (plane2 - alpha_max*(point2-point1) - point1), delta_plane ); //index min
		index.y = 1.0f + __fdividef( (point1 + alpha_min*(point2-point1) - plane1), delta_plane ); //index max
	}
	else
	{
		index.x = 0;  
		index.y = 0;
		d_mask = CUDART_NAN_F;
	}
		
	
	index.x = d_mask * find_index(index.x, alpha_min, alpha_max, point1, point2, plane1, delta_plane);
	index.y = d_mask * find_index(index.y, alpha_min, alpha_max, point1, point2, plane1, delta_plane);
	
	bound_index(index.x, index.y, d_mask);

	alpha_i = d_mask * alpha_float(index.x, index.y, point1, point2, plane1, delta_plane);
	alpha_f = d_mask * alpha_float(index.y, index.x, point1, point2, plane1, delta_plane);

	return ( index.y - index.x );
}


__device__ float2 siddon_calc_indices2(float &alpha_i, float &alpha_f, unsigned int &d_mask, float point1, float point2, float plane1, float plane2, float Nplanes, float alpha_min, float alpha_max, float delta_plane)
{
	float2 index;
	if ( (point2 - point1) > 0 )
	{
		index.x = Nplanes - __fdividef( (plane2 - alpha_min*(point2-point1) - point1), delta_plane ); //index min
		index.y = 1.0f + __fdividef( (point1 + alpha_max*(point2-point1) - plane1), delta_plane ); //index max
	}
	else if ( (point2 - point1) < 0 )
	{
		index.x = Nplanes - __fdividef( (plane2 - alpha_max*(point2-point1) - point1), delta_plane ); //index min
		index.y = 1.0f + __fdividef( (point1 + alpha_min*(point2-point1) - plane1), delta_plane ); //index max
	}
	else
	{
		index.x = 0;  
		index.y = 0;
		d_mask = CUDART_NAN_F;
	}
		
	
	index.x = d_mask * find_index(index.x, alpha_min, alpha_max, point1, point2, plane1, delta_plane);
	index.y = d_mask * find_index(index.y, alpha_min, alpha_max, point1, point2, plane1, delta_plane);
	
	bound_index(index.x, index.y, d_mask);

	alpha_i = d_mask * alpha_float( (unsigned int)index.x, (unsigned int)index.y, point1, point2, plane1, delta_plane);
	alpha_f = d_mask * alpha_float( (unsigned int)index.y, (unsigned int)index.x, point1, point2, plane1, delta_plane);

	return ( index );
}

__device__ void calc_points12(float3 &p1, float3 &p2,
							  CalibrationParam CalParam, Detector Det, VoxVolParam voxparam, float *RM, Planes XraySource,
							  int depth_iterate, int height_iterate, float angle_degree)
{
	float3 detector;
	//The shift here added by AxShift and TAxShift are found by ucenter and venter in the contracting-grid
	detector.x = pixel_iterate2(Det.NumAxPixels, Det.AxPixDimMm, CalParam.M, depth_iterate, Det.AxShift);
	detector.y = CalParam.R - CalParam.Rf;
	detector.z = pixel_iterate2(Det.NumTAxPixels, Det.TAxPixDimMm, CalParam.M, height_iterate, Det.TAxShift);

	//The detector is then shifted and rotated using calibrated parameters
	detector.x = __fdividef(CalParam.R, (CalParam.Ry + RM[3]*detector.x + RM[5]*detector.z)) *(RM[0]*detector.x + RM[2]*detector.z + CalParam.Dx);
	detector.z = __fdividef(CalParam.R, (CalParam.Ry + RM[3]*detector.x + RM[5]*detector.z)) *(RM[6]*detector.x + RM[8]*detector.z + CalParam.Dz);

	float sin_alpha, cos_alpha;
	//sincosf( __fdividef( angle_degree*M_PI, 180.0f) , &sin_alpha, &cos_alpha);
	cos_alpha = cospif( __fdividef( angle_degree, 180.0f) );
	sin_alpha = sinpif( __fdividef( angle_degree, 180.0f) );

	p1.x =	XraySource.x*cos_alpha + XraySource.y*sin_alpha;
	p1.y = -XraySource.x*sin_alpha + XraySource.y*cos_alpha;
	p1.z =	XraySource.z;

	p2.x =	detector.x*cos_alpha + detector.y*sin_alpha;
	p2.y = -detector.x*sin_alpha + detector.y*cos_alpha;
	p2.z =	detector.z; 

}

__device__ void increment(unsigned int &iterator, float p1, float p2)
{
	if (iterator == 0)
	{
		iterator = 0;
	}
	else
	{
		if ( (p2 - p1) > 0.0f )
		{
			iterator = iterator + 1;
		}
		else if ( (p2 - p1) < 0.0f )
		{
			iterator = iterator - 1;
		}
		else
		{
			iterator = iterator;
		}
	}
}


#endif