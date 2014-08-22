//written by Helen Fan

#pragma once

#ifndef HF_SIDDON_RECON_HPP
#define HF_SIDDON_RECON_HPP


#include "hf_siddon_class.hpp"
#include "RawData.hpp"
#include <strsafe.h>

//a0_*: are functions that are used for reconstruction
//a1_*: are functions that are either for forward projection using configuration file or inherited functions
//b_* : are public variables
//m_* : are protected functions and variables
//p_* : are private functions and variables
class siddon_recon : public forward, public backward, public WriteParameterFile
{
public:
	//main public functions that does stuff
	void a0_RECON_MLEM(std::string configurationfilename, std::string outputparameterfilename);
	void a0_RECON_MLEM(std::string configurationfilename);
	void a0_RECON_MLEM(std::string configurationfilefolder, std::string configruationfilename,
						std::string outputparameterfilename);

	void a1_FORWARD_PROJECTION(std::string configurationfilefolder, std::string configurationfilename);
	void a1_BACKWARD_PROJECTION(std::string configurationfilefolder, std::string configurationfilename, bool write_files_per_angle);

	//public functions to retrieve stuff
	float *a_pull_recon_pointer();
	float *a_dsensitivity_pointer();
	void a_pull_kernel_threads();
	void a_pull_kernel_threads(dim3 &blocks_image, dim3 &threads_image, dim3 &blocks_object, dim3 &threads_object, dim3 &blocks_atomic, dim3 &threads_atomic);

	//public values
	dim3 b_blocks_image, b_threads_image;
	dim3 b_blocks_object, b_threads_object;
	dim3 b_blocks_atomic, b_threads_atomic;
	int b_N_image_pixels;
	int b_N_object_voxels;

	//destructor
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
	siddon_commons m_siddon_var;

	void m_recon_initiate(std::string configurationfilename);
	void m_correct_filepaths(FilePaths &filepaths);
	inline bool m_check_files(std::string configurationfilename);
	inline bool m_check_forward_projection(std::string configurationfilename);
	inline bool m_check_backward_projection(std::string configurationfilename);
	bool m_check_reconstruction(std::string configurationfilename);

	void m_calc_sensitivity();
	void m_calc_kernel_threads();


	void m_mlem_single_iteration(int c2x, int c2y);
	void m_mlem_all_iterations();
	void m_forward_projection();
	void m_backward_projection(bool write_files_per_angle);


private:
	bool pb_allocate_check;
	bool pb_initiate_check;
	bool pb_sim_obj_file;

	float *pd_sensitivity;
};

// main public functions
void siddon_recon::a0_RECON_MLEM(std::string configurationfilefolder, std::string configurationfilename)
{
	correct_folder_path(configurationfilefolder);
	std::string config_file = configurationfilefolder + configurationfilename;

	if (this->m_check_reconstruction(config_file))
	{
		this->m_recon_initiate(config_file);
		this->m_mlem_all_iterations();
		this->a1_WriteReconParameters(this->m_params, this->m_det, this->m_vox, this->m_scanparam, this->m_filepath);
		std::cout << "Do reconstruction" << std::endl;
	}
	else
	{
		std::cout << "Encountered error, so did nothing" << std::endl;
	}

}

void siddon_recon::a0_RECON_MLEM(std::string configurationfilename)
{
	correct_folder_path(configurationfilename);
	if (this->m_check_reconstruction(configurationfilename))
	{
		this->m_recon_initiate(configurationfilename);
		this->m_mlem_all_iterations();
		this->a1_WriteReconParameters(this->m_params, this->m_det, this->m_vox, this->m_scanparam, this->m_filepath);
		std::cout << "Do reconstruction" << std::endl;
	}
	else
	{
		std::cout << "Encountered error, so did nothing" << std::endl;
	}
}

void siddon_recon::a0_RECON_MLEM(std::string configurationfilefolder, std::string configurationfilename, std::string outputparameterfilename)
{
	correct_folder_path(configurationfilefolder);
	std::string config_file = configurationfilefolder + configurationfilename;

	if (this->m_check_reconstruction(config_file))
	{
		this->m_recon_initiate(config_file);
		this->m_mlem_all_iterations();
		fix_configfile_suffix(outputparameterfilename);
		this->a1_WriteReconParameters(outputparameterfilename, this->m_params, this->m_det, this->m_vox, this->m_scanparam, this->m_filepath);
		std::cout << "Do reconstruction" << std::endl;
	}
	else
	{
		std::cout << "Encountered error, so did nothing" << std::endl;
	}

}

void siddon_recon::a1_FORWARD_PROJECTION(std::string configurationfilefolder, std::string configurationfilename)
{
	correct_folder_path(configurationfilefolder);
	std::string config_name = configurationfilefolder + configurationfilename;

	if (this->m_check_forward_projection(config_name))
	{
		this->m_recon_initiate(config_name);
		this->m_forward_projection();
		this->a1_WriteForwardParameters(this->m_params, this->m_det, this->m_vox, this->m_scanparam, this->m_filepath);
		std::cout << "Do forward projection" << std::endl;
	}
	else
	{
		std::cout << "Encountered error, so did nothing" << std::endl;
	}

}

void siddon_recon::a1_BACKWARD_PROJECTION(std::string configurationfilefolder, std::string configurationfilename, bool write_files_per_angle)
{
	correct_folder_path(configurationfilefolder);
	std::string config_file = configurationfilefolder + configurationfilename;

	if ( this->m_check_backward_projection(config_file) )
	{
		this->m_recon_initiate(config_file);
		this->m_backward_projection(write_files_per_angle);
		this->a1_WriteBackwardParameters(this->m_params, this->m_det, this->m_vox, this->m_scanparam, this->m_filepath);
		std::cout << "do m_backward_projection()" << std::endl;
	}
	else
	{ 
		std::cout << "Encountered error, so did nothing" << std::endl;
	}

}













inline bool siddon_recon::m_check_files(std::string configurationfilename)
{

	bool b_config_file = file_exist(configurationfilename);
	if (b_config_file)
	{
		//load values from configuration file
		this->m_params.LoadFromConfigFile(configurationfilename);
		this->m_det.LoadFromConfigFile(configurationfilename);
		this->m_vox.LoadFromConfigFile(configurationfilename);
		this->m_scanparam.LoadFromConfigFile(configurationfilename);
		this->m_filepath.LoadFromconfigFile(configurationfilename);
		this->m_correct_filepaths(this->m_filepath);

		//initiate and allocate functions for common siddon variables
		this->m_siddon_var.a1_initiate(this->m_params, this->m_det, this->m_vox);
		this->m_siddon_var.a2_allocate();
		this->b_N_image_pixels = this->m_siddon_var.N_image_pixels;
		this->b_N_object_voxels = this->m_siddon_var.N_object_voxels;
		
		std::string sim_obj_filename = this->m_filepath.sim_ObjFileFolder + this->m_filepath.sim_ObjFileName;
		this->pb_sim_obj_file = file_exist(sim_obj_filename);
		return true;
	}
	else
	{
		std::cout << "Error: Nothing was done" << std::endl;
		return false;
	}

}

inline bool siddon_recon::m_check_forward_projection(std::string configurationfilename)
{
	if (this->m_check_files(configurationfilename) )
	{
		std::string sim_obj_filename = this->m_filepath.sim_ObjFileFolder + this->m_filepath.sim_ObjFileName;
		std::ifstream::pos_type objfilesize = this->b_N_object_voxels*sizeof(float);
		bool b_create_dir = create_directory(this->m_filepath.ProjFileFolder.c_str());

		if ( pb_sim_obj_file == false )
		{
			std::cout << "Did NOT load object volume from: " << sim_obj_filename << std::endl;
			return false;
		}
		else if ( objfilesize != file_size( sim_obj_filename.c_str() ) )
		{
			std::cout << "object file size does not match input configuration file!" << std::endl;
			std::cout << "file size = " << file_size( sim_obj_filename.c_str() ) << "bytes" << std::endl;
			std::cout << "configuration file indicates that it should be = " << objfilesize << "bytes" << std::endl;
			return false;
		}
		
		else if ( !b_create_dir )
		{			
			std::cout << "New directory was not created" << std::endl;
			std::cout << "Forward projection not calculated" << std::endl;
			return false;
		}
		else if ( b_create_dir && pb_sim_obj_file && objfilesize & file_size( sim_obj_filename.c_str() ) )
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}

}

inline bool siddon_recon::m_check_backward_projection(std::string configurationfilename)
{
	if (this->m_check_files(configurationfilename))
	{
		//check_projection_files makes sure all projection image files exists, and its size are valid
		bool file_check = check_projection_files(this->m_scanparam.NumProj, this->m_det, this->m_filepath.ProjFileFolder, this->m_filepath.ProjFileNameRoot, this->m_filepath.ProjFileSuffix); 
		if ( file_check )
		{
			return true;
		}
		else
		{
			return false;
		}

	}
	else
	{
		return false;
	}

}

bool siddon_recon::m_check_reconstruction(std::string configurationfilename)
{
	if (this->m_check_files(configurationfilename) )
	{
		bool file_check = check_projection_files(this->m_scanparam.NumProj, this->m_det, this->m_filepath.ProjFileFolder, this->m_filepath.ProjFileNameRoot, this->m_filepath.ProjFileSuffix); 
		if (file_check)
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}

}


float *siddon_recon::a_pull_recon_pointer()
{
	return ( &this->m_object[0] );
}



void siddon_recon::a_pull_kernel_threads()
{
	this->b_blocks_atomic = this->m_blocks_atomic;
	this->b_threads_atomic = this->m_threads_atomic;

	this->b_blocks_image = this->m_blocks_image;
	this->b_threads_image = this->m_threads_image;
	
	this->b_blocks_object = this->m_blocks_object;
	this->b_threads_object = this->m_threads_object;
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
	correct_folder_path(filepaths.ProjFileFolder);
	correct_folder_path(filepaths.sim_ObjFileFolder);

	if (filepaths.ReconFileFolder.empty())
	{
		filepaths.ReconFileFolder = filepaths.ProjFileFolder;
	}
}

float *siddon_recon::a_dsensitivity_pointer()
{
	
	return( this->pd_sensitivity );

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

void siddon_recon::m_calc_sensitivity()
{
	int cursorX, cursorY;
	CUDA_SAFE_CALL( cudaMemset( this->pd_sensitivity, 0, this->m_siddon_var.N_object_voxels*sizeof(float) ) );
	set_ones<<<this->m_blocks_image, this->m_threads_image>>>(md_g_data);
	std::cout << "calculating sensitivity..." << std::endl;
	std::cout << "angle = ";
	CursorGetXY(cursorX, cursorY);

	float angle_degree;
	for (unsigned int n = 0; n < this->m_scanparam.NumProj; n++) // mlem_values.total_projection_images
	{
		angle_degree = this->m_scanparam.DeltaAng*(float)n;

		this->m_siddon_var.a_calc_initial_limits(angle_degree, this->m_blocks_image, this->m_threads_image);
		this->a2_bp_per_angle(md_g_data, m_siddon_var, this->m_blocks_image, this->m_threads_image, this->m_blocks_object, this->m_threads_object, this->m_blocks_atomic, this->m_threads_atomic);
		siddon_add_bp<<<this->m_blocks_object, this->m_threads_object>>>( this->pd_sensitivity, this->a_bpdevicepointer() );
		
		//CUDA_SAFE_CALL( cudaMemcpy(&this->m_object[0], this->pd_sensitivity, this->N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );
		//savefloat(&this->m_object[0], this->N_object_voxels*sizeof(float), create_filename("", "sense", n, ".bin"));
		CursorGotoXY(cursorX, cursorY, "          ");
		CursorGotoXY(cursorX, cursorY);
		std::cout << angle_degree;
	}
	std::cout << "\n";
	std::cout << "sensitivity calculated! " << std::endl;

}

void siddon_recon::m_recon_initiate(std::string configurationfilename)
{
	//initiate and allocate functions to do forward and backward projection
	this->a1_Binitiate(this->b_N_object_voxels, this->b_N_image_pixels);
	this->a1_Finitiate(this-> b_N_image_pixels);

	//allocate space for arrays that will be used 
	this->m_object.resize(this->b_N_object_voxels, 0);
	this->m_image.resize(this->b_N_image_pixels, 0);
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_f, this->b_N_object_voxels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_Ht_fscale, this->b_N_object_voxels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->pd_sensitivity, this->b_N_object_voxels*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&this->md_g_data, this->b_N_image_pixels*sizeof(float) ) );

	this->m_calc_kernel_threads();
	this->a_pull_kernel_threads();
	this->pb_allocate_check = true;

}

void siddon_recon::m_mlem_single_iteration(int c2x, int c2y)
{	
		CUDA_SAFE_CALL( cudaMemset(this->md_Ht_fscale, 0, this->b_N_object_voxels*sizeof(float) ) );

		for (unsigned int n = 0; n < this->m_scanparam.NumProj; n++)
		{
			float angle_degrees = (float)n * this->m_scanparam.DeltaAng;

			//load g
			readfloat(&this->m_image[0], this->b_N_image_pixels*sizeof(float), 
				create_filename(this->m_filepath.ProjFileFolder, this->m_filepath.ProjFileNameRoot, n, this->m_filepath.ProjFileSuffix));
		
				CUDA_SAFE_CALL( cudaMemcpy(this->md_g_data, &this->m_image[0], this->b_N_image_pixels*sizeof(float), cudaMemcpyHostToDevice) );

			//siddon algorithm initial numbers that'll be used for both forward and backward projections per angle
			this->m_siddon_var.a_calc_initial_limits(angle_degrees, this->m_blocks_image, this->m_threads_image);

			// Hf
			this->a2_fp_per_angle(this->md_f, this->m_siddon_var, this->m_blocks_image, this->m_threads_image);

			// g/Hf
			divide1<float><<<this->m_blocks_image, this->m_threads_image>>>(this->md_g_data, this->a_Fdevicepointer());

			// Ht (g/Hf)
			this->a2_bp_per_angle(this->md_g_data, this->m_siddon_var, this->m_blocks_image, this->m_threads_image, this->m_blocks_object, this->m_threads_object, this->m_blocks_atomic, this->m_threads_atomic);
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

			CUDA_SAFE_CALL( cudaMemcpy(&this->m_object[0], this->md_f, this->b_N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );

			savefloat(&this->m_object[0], this->b_N_object_voxels*sizeof(float), 
				create_filename(this->m_filepath.ReconFileFolder, this->m_filepath.ReconFileNameRoot, iteration, this->m_filepath.ReconFileSuffix) );
		}
		std::cout << "completed!" << std::endl;
		std::cout << "================================="<< std::endl;
		std::cout << "\n" <<std::endl;
}



void siddon_recon::m_forward_projection()
{
	readfloat( &this->m_object[0], this->b_N_object_voxels*sizeof(float), 
		create_filename(this->m_filepath.sim_ObjFileFolder, this->m_filepath.sim_ObjFileName) );
	CUDA_SAFE_CALL( cudaMemcpy( this->md_f, &this->m_object[0], this->b_N_object_voxels*sizeof(float), cudaMemcpyHostToDevice) );

	int x, y;
	std::cout << "================================="<< std::endl;
	std::cout << "Forward Projection started" << std::endl;
	std::cout << "Projection angle (degrees) = \t";
	CursorGetXY(x, y);
	for (unsigned int n = 0; n < this->m_scanparam.NumProj; n++)
	{
		float angle_degree = (float)n * this->m_scanparam.DeltaAng;
		this->m_siddon_var.a_calc_initial_limits(angle_degree, this->m_blocks_image, this->m_threads_image);
		this->a2_fp_per_angle(this->md_f, this->m_siddon_var, this->m_blocks_image, this->m_threads_image);
		CUDA_SAFE_CALL( cudaMemcpy( &this->m_image[0], this->a_Fdevicepointer(), this->b_N_image_pixels*sizeof(float), cudaMemcpyDeviceToHost) );
		savefloat( &this->m_image[0], this->b_N_image_pixels*sizeof(float), 
			create_filename( this->m_filepath.ProjFileFolder, this->m_filepath.ProjFileNameRoot, n, this->m_filepath.ProjFileSuffix) );
		
		//for display to prompt
		CursorGotoXY(x, y, "           ");
		CursorGotoXY(x, y);
		std::cout << angle_degree << std::endl;
	}
	std::cout << "completed!" << std::endl;
	std::cout << "================================="<< std::endl;
	std::cout << "\n" <<std::endl;

}

void siddon_recon::m_backward_projection(bool write_files_per_angle)
{
	int x, y;
	std::cout << "================================="<< std::endl;
	std::cout << "Backward Projection started" << std::endl;
	std::cout << "Backprojection angle (degrees) = \t";
	CursorGetXY(x, y);

	for (unsigned int n = 0; n < this->m_scanparam.NumProj; n++)
	{
		float angle_degree = (float)n * this->m_scanparam.DeltaAng;
		this->m_siddon_var.a_calc_initial_limits(angle_degree, this->m_blocks_image, this->m_threads_image);

		//load the projection images to host memory
		readfloat(&this->m_image[0], this->b_N_image_pixels*sizeof(float), 
					create_filename( this->m_filepath.ProjFileFolder, this->m_filepath.ProjFileNameRoot, n, this->m_filepath.ProjFileSuffix) );

		//copy projection host memory to device memory
		CUDA_SAFE_CALL( cudaMemcpy( this->md_g_data, &this->m_image[0], this->b_N_image_pixels*sizeof(float), cudaMemcpyHostToDevice));


		this->a2_bp_per_angle(this->md_g_data, this->m_siddon_var, this->m_blocks_image, this->m_threads_image, 
								this->m_blocks_object, this->m_threads_object, this->m_blocks_atomic, this->m_threads_atomic);

		siddon_add_bp<<<this->m_blocks_object, this->m_threads_object>>>(this->md_f, this->a_bpdevicepointer());

		if (write_files_per_angle)
		{
			CUDA_SAFE_CALL( cudaMemcpy(&this->m_object[0], this->a_bpdevicepointer(), this->b_N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );
			savefloat( &this->m_object[0], this->b_N_object_voxels*sizeof(float), 
				create_filename(this->m_filepath.ProjFileFolder, this->m_filepath.sim_BpFileNameRoot, n, this->m_filepath.sim_BpFileSuffix) );
		}

		//for display to prompt
		CursorGotoXY(x, y, "           ");
		CursorGotoXY(x, y);
		std::cout << angle_degree << std::endl;
	}
	CUDA_SAFE_CALL( cudaMemcpy(&this->m_object[0], this->md_f, this->b_N_object_voxels*sizeof(float), cudaMemcpyDeviceToHost) );
	std::string filename = this->m_filepath.sim_BpFileNameRoot + this->m_filepath.sim_BpFileSuffix;
	savefloat( &this->m_object[0], this->b_N_object_voxels*sizeof(float), 
		create_filename(this->m_filepath.ProjFileFolder, filename) );

	std::cout << "completed!" << std::endl;
	std::cout << "================================="<< std::endl;
	std::cout << "\n" <<std::endl;
}

siddon_recon::~siddon_recon()
{
	if (pb_allocate_check)
	{
		m_siddon_var.a3_deallocate();
		this->m_image.clear();
		this->m_object.clear();
		CUDA_SAFE_CALL( cudaFree(this->md_f) );
		CUDA_SAFE_CALL( cudaFree(this->md_g_data) );
		CUDA_SAFE_CALL( cudaFree(this->md_Ht_fscale) );
		CUDA_SAFE_CALL( cudaFree(this->pd_sensitivity) );
	}
}

#endif
