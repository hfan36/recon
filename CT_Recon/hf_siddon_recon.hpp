#pragma once


#ifndef HF_SIDDON_RECON_HPP
#define HF_SIDDON_RECON_HPP


#include "hf_siddon_class.hpp"
#include "RawData.hpp"

class siddon_recon : public forward, public backward, public WriteParameterFile
{
public:
	void a1_recon_initiate(std::string configurationfilename);
	void a1_recon_initiate(std::string folder, std::string configurationfilename);
	void a1_forward_projection(std::string folder, std::string configurationfilename);

	void a_MLEM(std::string outputparameterfilename);
	void a_MLEM();

	float *a_pull_recon_pointer();
	float *a_dsensitivity_pointer();
	void a_pull_kernel_threads();
	void a_pull_kernel_threads(dim3 &blocks_image, dim3 &threads_image, dim3 &blocks_object, dim3 &threads_object, dim3 &blocks_atomic, dim3 &threads_atomic);

	dim3 a_blocks_image, a_threads_image;
	dim3 a_blocks_object, a_threads_object;
	dim3 a_blocks_atomic, a_threads_atomic;
	int N_image_pixels;
	int N_object_voxels;
	siddon_commons siddon_var;

	~siddon_recon();
	
protected:
	CalibrationParam m_params;
	Detector m_det;
	VoxVolParam m_vox;
	ScanParameters m_scanparam;
	FilePaths m_filepath;

	LoadRawData <float> readfloat;
	SaveRawData <float> savefloat;

	std::vector <float> m_image;
	std::vector <float> m_object;
	float *md_f, *md_Ht_fscale, *md_g_data;

	dim3 m_blocks_image, m_threads_image;
	dim3 m_blocks_object, m_threads_object;
	dim3 m_blocks_atomic, m_threads_atomic;

	void m_recon_initiate(std::string configurationfilename);
	void m_calc_sensitivity();
	void m_calc_kernel_threads();
	void m_correct_filepaths(FilePaths &filepaths);

	void m_mlem_single_iteration(int c2x, int c2y);
	void m_mlem_all_iterations();
	void m_forward_projection();

private:
	bool allocate_check;
	float *pd_sensitivity;
};

void siddon_recon::m_calc_sensitivity()
{
	int cursorX, cursorY;
	CUDA_SAFE_CALL( cudaMemset( this->pd_sensitivity, 0, this->N_object_voxels*sizeof(float) ) );
	set_ones<<<this->m_blocks_image, this->m_threads_image>>>(md_g_data);
	std::cout << "calculating sensitivity..." << std::endl;
	std::cout << "angle = ";
	CursorGetXY(cursorX, cursorY);

	float angle_degree;
	for (unsigned int n = 0; n < this->m_scanparam.NumProj; n++) // mlem_values.total_projection_images
	{
		angle_degree = this->m_scanparam.DeltaAng*(float)n;

		this->siddon_var.a_calc_initial_limits(angle_degree, this->m_blocks_image, this->m_threads_image);
		this->a2_bp_per_angle(md_g_data, siddon_var, this->m_blocks_image, this->m_threads_image, this->m_blocks_object, this->m_threads_object, this->m_blocks_atomic, this->m_threads_atomic);
		siddon_add_bp<<<this->m_blocks_object, this->m_threads_object>>>( this->pd_sensitivity, this->a_bpdevicepointer() );
		
		//CUDA_SAFE_CALL( cudaMemcpy(&this->m_object[0], this->pd_sensitivity, this->N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );
		//savefloat(&this->m_object[0], this->N_object_voxels*sizeof(float), create_filename("", "sense", n, ".bin"));
		CursorGotoXY(cursorX, cursorY, "          ");
		CursorGotoXY(cursorX, cursorY);
		std::cout << angle_degree;
	}
	std::cout<< "\n sensitivity calculated! " << std::endl;

}

void siddon_recon::m_calc_kernel_threads()
{
	this->m_threads_image.x = 8;
	this->m_threads_image.y = 8;
	this->m_blocks_image.x = this->m_det.NumAxPixels/m_threads_image.x;
	this->m_blocks_image.y = this->m_det.NumTAxPixels/m_threads_image.y;
	
	this->m_threads_object.x = 4;
	this->m_threads_object.y = 4;
	this->m_threads_object.z = 4;
	this->m_blocks_object.x = this->m_vox.Xdim/m_threads_object.x;
	this->m_blocks_object.y = this->m_vox.Ydim/m_threads_object.y;
	this->m_blocks_object.z = this->m_vox.Zdim/m_threads_object.z;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int N_processor = prop.multiProcessorCount;

	this->m_threads_atomic.x = 256;
	this->m_blocks_atomic.x = 2*N_processor;
}

void siddon_recon::a_pull_kernel_threads()
{
	this->a_blocks_atomic = this->m_blocks_atomic;
	this->a_threads_atomic = this->m_threads_atomic;

	this->a_blocks_image = this->m_blocks_image;
	this->a_threads_image = this->m_threads_image;
	
	this->a_blocks_object = this->m_blocks_object;
	this->a_threads_object = this->m_threads_object;
}

void siddon_recon::a_pull_kernel_threads(dim3 &blocks_image, dim3 &threads_image, dim3 &blocks_object, dim3 &threads_object, dim3 &blocks_atomic, dim3 &threads_atomic)
{
	blocks_atomic = this->m_blocks_atomic;
	threads_atomic = this->m_threads_atomic;

	blocks_image = this->m_blocks_image;
	threads_image = this->m_threads_image;
	
	blocks_object = this->m_blocks_object;
	threads_object = this->m_threads_object;
}

void siddon_recon::m_correct_filepaths(FilePaths &filepaths)
{
	//get all the slashes checked
	check_folder_path(filepaths.ProjFileFolder);

	if (filepaths.ReconFileFolder.empty())
	{
		filepaths.ReconFileFolder = filepaths.ProjFileFolder;
	}
}

float *siddon_recon::a_dsensitivity_pointer()
{
	
	return( this->pd_sensitivity );

}

float *siddon_recon::a_pull_recon_pointer()
{
	return ( &this->m_object[0] );
}

void siddon_recon::m_recon_initiate(std::string configurationfilename)
{
	//load values from configuration file
	this->m_params.LoadFromConfigFile(configurationfilename);
	this->m_det.LoadFromConfigFile(configurationfilename);
	this->m_vox.LoadFromConfigFile(configurationfilename);
	this->m_scanparam.LoadFromConfigFile(configurationfilename);
	this->m_filepath.LoadFromconfigFile(configurationfilename);
	m_correct_filepaths(this->m_filepath);

	//initiate and allocate functions for common siddon variables
	this->siddon_var.a1_initiate(this->m_params, this->m_det, this->m_vox);
	this->siddon_var.a2_allocate();
	this->N_image_pixels = this->siddon_var.N_image_pixels;
	this->N_object_voxels = this->siddon_var.N_object_voxels;

	//initiate and allocate functions to do forward and backward projection
	this->a1_Binitiate(this->N_object_voxels, this->N_image_pixels);
	this->a1_Finitiate(this-> N_image_pixels);

	//allocate space for arrays that will be used 
	this->m_object.resize(this->N_object_voxels, 0);
	this->m_image.resize(this->N_image_pixels, 0);
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_f, this->N_object_voxels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_Ht_fscale, this->N_object_voxels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->pd_sensitivity, this->N_object_voxels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_g_data, this->N_image_pixels*sizeof(float) ) );

	this->m_calc_kernel_threads();
	this->a_pull_kernel_threads();
	allocate_check = true;

}

void siddon_recon::a1_recon_initiate(std::string configurationfilename)
{
	if (file_exist(configurationfilename))
	{
		this->m_recon_initiate(configurationfilename);
	}
}

void siddon_recon::a1_recon_initiate(std::string folder, std::string configurationfilename)
{
	check_folder_path(folder);
	folder.append(configurationfilename);
	
	if (file_exist(folder))
	{
		this->m_recon_initiate(folder);
	}
}

void siddon_recon::m_mlem_single_iteration(int c2x, int c2y)
{	
		CUDA_SAFE_CALL( cudaMemset(this->md_Ht_fscale, 0, this->N_object_voxels*sizeof(float) ) );

		for (unsigned int n = 0; n < this->m_scanparam.NumProj; n++)
		{
			float angle_degrees = (float)n * this->m_scanparam.DeltaAng;

			//load g
			readfloat(&this->m_image[0], this->N_image_pixels*sizeof(float), 
				create_filename(this->m_filepath.ProjFileFolder, this->m_filepath.ProjFileNameRoot, n, this->m_filepath.ProjFileSuffix));
		
				CUDA_SAFE_CALL( cudaMemcpy(this->md_g_data, &this->m_image[0], this->N_image_pixels*sizeof(float), cudaMemcpyHostToDevice) );

			//siddon algorithm initial numbers that'll be used for both forward and backward projections per angle
			this->siddon_var.a_calc_initial_limits(angle_degrees, this->m_blocks_image, this->m_threads_image);

			// Hf
			this->a2_fp_per_angle(this->md_f, this->siddon_var, this->m_blocks_image, this->m_threads_image);

			// g/Hf
			divide1<float><<<this->m_blocks_image, this->m_threads_image>>>(this->md_g_data, this->a_Fdevicepointer());

			// Ht (g/Hf)
			this->a2_bp_per_angle(this->md_g_data, this->siddon_var, this->m_blocks_image, this->m_threads_image, this->m_blocks_object, this->m_threads_object, this->m_blocks_atomic, this->m_threads_atomic);
			siddon_add_bp<<<this->m_blocks_object, this->m_threads_object>>>(this->md_Ht_fscale, this->a_bpdevicepointer() );


			//============== nothing but display on prompt ==========================================================
			CursorGotoXY(c2x, c2y, "          ");
			CursorGotoXY(c2x, c2y);
			std::cout << angle_degrees << "\n";
			//=======================================================================================================
		}
		
		
		multiply1<float><<<this->m_blocks_object, this->m_threads_object>>>(this->md_f, this->md_Ht_fscale);
		divide1<float><<<this->m_blocks_object, this->m_threads_object>>>(this->md_f, this->a_dsensitivity_pointer());

}

void siddon_recon::m_mlem_all_iterations()
{
		int c1x, c1y, c2x, c2y;

		this->m_calc_sensitivity();
		set_ones<<<this->m_blocks_object, this->m_threads_object>>>(this->md_f);

		std::cout << "\n";
		std::cout << "================================="<< std::endl;
		std::cout << "MLEM recon started..." << std::endl;
		std::cout << "MLEM loop = ";
		CursorGetXY(c1x, c1y);

		for (unsigned int iteration = 0; iteration < this->m_scanparam.N_iteration; iteration++)
		{
			//============== nothing but display on prompt ==========================================================
			CursorGotoXY(c1x, c1y, "          ");
			CursorGotoXY(c1x, c1y);
			std::cout << iteration << std::endl;
			std::cout << "angle = ";
			CursorGetXY(c2x, c2y);
			//=======================================================================================================
	
			this->m_mlem_single_iteration(c2x, c2y);

			CUDA_SAFE_CALL( cudaMemcpy(&this->m_object[0], this->md_f, this->N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );

			savefloat(&this->m_object[0], this->N_object_voxels*sizeof(float), 
				create_filename(this->m_filepath.ReconFileFolder, this->m_filepath.ReconFileNameRoot, iteration, this->m_filepath.ReconFileSuffix) );
		}
		std::cout << "completed!" << std::endl;
		std::cout << "================================="<< std::endl;
		std::cout << "\n" <<std::endl;
}


void siddon_recon::a_MLEM()
{
	bool file_check = check_projection_files(this->m_scanparam.NumProj, this->m_det, this->m_filepath.ProjFileFolder, this->m_filepath.ProjFileNameRoot, this->m_filepath.ProjFileSuffix); 

	if (file_check)
	{
		this->m_mlem_all_iterations();
		this->a1_WriteParameters(this->m_params, this->m_det, this->m_vox, this->m_scanparam, this->m_filepath);
	}
}

void siddon_recon::a_MLEM(std::string outputparameterfilename)
{

	bool file_check = check_projection_files(this->m_scanparam.NumProj, this->m_det, this->m_filepath.ProjFileFolder, this->m_filepath.ProjFileNameRoot, this->m_filepath.ProjFileSuffix); 
	fix_configfile_suffix(outputparameterfilename);

	if (file_check)
	{
		this->m_mlem_all_iterations();
		this->a1_WriteParameters(outputparameterfilename, this->m_params, this->m_det, this->m_vox, this->m_scanparam, this->m_filepath);
	}

}

void siddon_recon::a1_forward_projection(std::string folder, std::string configurationfilename)
{
	this->a1_recon_initiate(folder, configurationfilename);
	check_folder_path(this->m_filepath.sim_ObjFileFolder);
	std::string objfilename = this->m_filepath.sim_ObjFileFolder + this->m_filepath.sim_ObjFileName;
	std::ifstream::pos_type objfilesize = this->N_object_voxels*sizeof(float);
	
	if ( !file_exist(objfilename) )
	{
		std::cout << "object file does not exist!!" << std::endl;
		std::cout << "The loaded object filename: " << objfilename << std::endl;
	}
	else if ( objfilesize != file_size( objfilename.c_str() ) )
	{
		std::cout << "object file size does not match input configuration file!" << std::endl;
		std::cout << "file size = " << file_size( objfilename.c_str() ) << "bytes" << std::endl;
		std::cout << "configuration file indicates that it should be = " << objfilesize << "bytes" << std::endl;
	}
	else
	{
		//this->m_forward_projection();
		std::cout << "do forward projection of all angles" << std::endl;
	}
}

siddon_recon::~siddon_recon()
{
	if (allocate_check)
	{
		siddon_var.a3_deallocate();
		this->m_image.clear();
		this->m_object.clear();
		CUDA_SAFE_CALL( cudaFree(this->md_f) );
		CUDA_SAFE_CALL( cudaFree(this->md_g_data) );
		CUDA_SAFE_CALL( cudaFree(this->md_Ht_fscale) );
		CUDA_SAFE_CALL( cudaFree(this->pd_sensitivity) );
	}
}

#endif


//void siddon_recon::a_MLEM(mlem_input_values &mlem_values)
//{
//
//	//Cursor positions
//	int c1x, c1y, c2x, c2y;
//
//	//calculate sensitivity first
//	m_calc_sensitivity();
//
//	set_ones<<<this->m_blocks_object, this->m_threads_object>>>(this->md_f);
//
//	std::string recon_filename_root;
//	recon_filename_root = "recon_" + mlem_values.filename_root;
//	float angle_degrees;
//
//	std::cout << "====================================================================="<< std::endl;
//	std::cout << "MLEM recon started..." << std::endl;
//	std::cout << "MLEM loop = ";
//	CursorGetXY(c1x, c1y);
//	for (int iteration = 0; iteration < mlem_values.N_iterations; iteration++)
//	{
//		//============== nothing but display on prompt ==========================================================
//		CursorGotoXY(c1x, c1y, "          ");
//		CursorGotoXY(c1x, c1y);
//		std::cout << iteration << std::endl;
//		std::cout << "angle = ";
//		CursorGetXY(c2x, c2y);
//		//=======================================================================================================
//
//		
//		CUDA_SAFE_CALL( cudaMemset(this->md_Ht_fscale, 0, this->N_object_voxels*sizeof(float) ) );
//		for (int n = 0; n < mlem_values.total_projection_images; n++)
//		{
//			angle_degrees = (float)n* mlem_values.delta_angle_deg;
//
//			//load g
//			readfloat(&this->m_image[0], this->N_image_pixels*sizeof(float), 
//				create_filename(mlem_values.folder_root, mlem_values.filename_root, n, mlem_values.suffix));
//			CUDA_SAFE_CALL( cudaMemcpy(this->md_g_data, &this->m_image[0], this->N_image_pixels*sizeof(float), cudaMemcpyHostToDevice) );
//
//			//siddon algorithm initial numbers that'll be used for both forward and backward projections per angle
//			this->siddon_var.a_calc_initial_limits(angle_degrees, this->m_blocks_image, this->m_threads_image);
//			//std::cout << "loopN = " << siddon_var.loopN << "\t angle = " << angle_degrees << std::endl; 
//
//			// Hf
//			this->a2_fp_per_angle(this->md_f, this->siddon_var, this->m_blocks_image, this->m_threads_image);
//
//			// g/Hf
//			divide1<float><<<this->m_blocks_image, this->m_threads_image>>>(this->md_g_data, this->a_Fdevicepointer());
//
//			// Ht (g/Hf)
//			this->a2_bp_per_angle(this->md_g_data, this->siddon_var, this->m_blocks_image, this->m_threads_image, this->m_blocks_object, this->m_threads_object, this->m_blocks_atomic, this->m_threads_atomic);
//			siddon_add_bp<<<this->m_blocks_object, this->m_threads_object>>>(this->md_Ht_fscale, this->a_bpdevicepointer() );
//
//
//			//============== nothing but display on prompt ==========================================================
//			CursorGotoXY(c2x, c2y, "          ");
//			CursorGotoXY(c2x, c2y);
//			std::cout << angle_degrees;
//			//=======================================================================================================
//		}
//		multiply1<float><<<this->m_blocks_object, this->m_threads_object>>>(this->md_f, this->md_Ht_fscale);
//		divide1<float><<<this->m_blocks_object, this->m_threads_object>>>(this->md_f, this->a_dsensitivity_pointer());
//
//		CUDA_SAFE_CALL( cudaMemcpy(&this->m_object[0], this->md_f, this->N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );
//
//
//		savefloat(&this->m_object[0], this->N_object_voxels*sizeof(float), 
//			create_filename( mlem_values.folder_root, recon_filename_root, iteration, mlem_values.suffix) );
//			
//	}
//
//	std::cout << "\n";
//	std::cout << "completed! \n" << std::endl;
//
//}