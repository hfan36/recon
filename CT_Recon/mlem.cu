#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <Windows.h>
#include <cuda_profiler_api.h>
#include "hf_siddon_recon.hpp"


int main()
{
	cudaProfilerStart();


	double totalTime;
	StopWatchInterface *timer;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	
	//----------------reconstruction part--------------------------------
	siddon_recon recon;
	recon.a1_recon_initiate("CTParameters_042414.h");
	//recon.a_MLEM();
	//WriteToParamFile w;
	//w.CalParam.LoadFromConfigFile("CTParameters_042414.h");
	//w.ObjParam.LoadFromConfigFile("CTParameters_042414.h");
	//w.WriteToFile("test.txt");

	//CalibrationParam param;
	//Detector det;
	//VoxVolParam vox;
	//ScanParameters scanparam;
	//FilePaths filepath;
	//param.LoadFromConfigFile("CTParameters_042414.h");
	//det.LoadFromConfigFile("CTParameters_042414.h");
	//vox.LoadFromConfigFile("CTParameters_042414.h");
	//scanparam.LoadFromConfigFile("CTParameters_042414.h");
	//filepath.LoadFromconfigFile("CTParameters_042414.h");

	//WriteParameterFile w;
	//w.PrintToFile("test.cfg", param, det, vox, scanparam, filepath);
	
	//-------------------------------------------------------------------


	//-------------- just forward projection----------------------------
	//siddon_recon recon;
	//recon.a1_recon_initiate("CTParameters_042414.h");
	//dim3 threads_image, blocks_image, threads_object, blocks_object, threads_atomic, blocks_atomic;
	//recon.a_pull_kernel_threads(blocks_image, threads_image, blocks_object, threads_object, blocks_atomic, threads_atomic);
	//int N_image_pixels = recon.siddon_var.N_image_pixels;
	//int N_object_voxels = recon.siddon_var.N_object_voxels;
	//float *image = new float [N_image_pixels];
	//float *object = new float [N_object_voxels];
	//float *d_image, *d_object;
	//LoadRawData<float> readfloat;
	//SaveRawData<float> savefloat;
	//
	//CUDA_SAFE_CALL( cudaMalloc((void**)&d_image, N_image_pixels*sizeof(float)) );
	//CUDA_SAFE_CALL( cudaMalloc((void**)&d_object, N_object_voxels*sizeof(float)) );
	//CUDA_SAFE_CALL( cudaMemset(d_object, 0, N_object_voxels*sizeof(float) ) );
	//
	//readfloat(object, N_object_voxels*sizeof(float), "ball64_float.bin");
	//CUDA_SAFE_CALL( cudaMemcpy(d_object, object, N_object_voxels*sizeof(float), cudaMemcpyHostToDevice) );

	//for (int n = 0; n < 180; n++)
	//{
	//	float angle_degree = (float)n*2.0f;
	//	recon.siddon_var.a_calc_initial_limits(angle_degree, blocks_image, threads_image);
	//	recon.a2_fp_per_angle(d_object, recon.siddon_var, blocks_image, threads_image);
	//	CUDA_SAFE_CALL( cudaMemcpy(image, recon.a_Fdevicepointer(), N_image_pixels*sizeof(float), cudaMemcpyDeviceToHost) );
	//	savefloat(image, N_image_pixels*sizeof(float), 
	//		create_filename("H:/Visual Studio 2010/CTSolution/Siddon/ball/", "fp_ball64", n, ".bin") );
	//	std::cout << "angle = " << angle_degree << std::endl;
	//}

	//delete [] image, object;
	//CUDA_SAFE_CALL( cudaFree(d_image) );
	//CUDA_SAFE_CALL( cudaFree(d_object) );

	//--------------------------------------------------------------------


	//------- just backprojection--------------------------------------
	//dim3 threads_image, blocks_image, threads_object, blocks_object, threads_atomic, blocks_atomic;

	//siddon_recon recon;
	//recon.a1_recon_initiate("CTParameters_042414.h");
	//recon.a_pull_kernel_threads(blocks_image, threads_image, blocks_object, threads_object, blocks_atomic, threads_atomic);
	//int N_image_pixels = recon.siddon_var.N_image_pixels;
	//int N_object_voxels = recon.siddon_var.N_object_voxels;


	//float *image = new float [N_image_pixels];
	//float *object = new float [N_object_voxels];
	//
	//float *d_image, *d_object;
	//LoadRawData<float> readfloat;
	//SaveRawData<float> savefloat;
	//
	//CUDA_SAFE_CALL( cudaMalloc((void**)&d_image, N_image_pixels*sizeof(float)) );
	//CUDA_SAFE_CALL( cudaMalloc((void**)&d_object, N_object_voxels*sizeof(float)) );
	//CUDA_SAFE_CALL( cudaMemset(d_object, 0, N_object_voxels*sizeof(float) ) );

	//for (int n = 0; n < 180; n++)
	//{
	//	float angle_degree = (float)n*2;
	//	recon.siddon_var.a_calc_initial_limits(angle_degree, blocks_image, threads_image);

	//	//readfloat(image, N_image_pixels*sizeof(float), "");
	//	//CUDA_SAFE_CALL( cudaMemcpy(d_image, image, N_image_pixels*sizeof(float), cudaMemcpyHostToDevice) );

	//	set_ones<<<blocks_image, threads_image>>>(d_image);
	//	recon.a2_bp_per_angle(d_image, recon.siddon_var, blocks_image, threads_image, blocks_object, threads_object, blocks_atomic, threads_atomic);
	//	siddon_add_bp<<<blocks_object, threads_object>>>(d_object, recon.a_bpdevicepointer());

	//	CUDA_SAFE_CALL( cudaMemcpy(object, recon.a_bpdevicepointer(), N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );
	//	savefloat(object, N_object_voxels*sizeof(float), 
	//		create_filename("H:/Visual Studio 2010/CTSolution/Siddon/bp_ones/", "bp_ones32", n, ".bin") );
	//	std::cout << "angle = " << angle_degree << std::endl;
	//}
	//
	//
	//CUDA_SAFE_CALL( cudaMemcpy(object, d_object, N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );
	//savefloat(object, N_object_voxels*sizeof(float), 
	//	create_filename("H:/Visual Studio 2010/CTSolution/Siddon/bp_ones/", "total_bp_ones32.bin") );
	//CUDA_SAFE_CALL( cudaFree(d_image) );
	//CUDA_SAFE_CALL( cudaFree(d_object) );
	//delete [] image, object;	
	//-----------------------------------------------------------------------------------------------------------------------------------------------------


	sdkStopTimer(&timer);
	totalTime = sdkGetTimerValue(&timer)*1e-3;
	printf("calculation time = %f seconds \n", totalTime);

	cudaProfilerStop();

	
	
	system("PAUSE");
}