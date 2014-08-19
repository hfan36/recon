//Written by Helen Fan, modified from Jared Moore's work


#ifndef CTSYSTEMELEMENTS_HF_H_
#define CTSYSTEMELEMENTS_HF_H_


#include "ConfigFile.h"
#include <map>
#include <string>


struct CalibrationParam
{
	float R;		//focal spot to phosphor screen (Rf + Rd)
	float Rf;		//x-ray source to center of rotation (mm)
	float Ry;		//focal spot to axis of rotation
	float Dx;		//detector y-offset (misalignment)
	float Dz;		//detector z-offset (misalignment)
	float xangle;	//detector angle about x-axis (misalignment)
	float yangle;	//detector angle about y-axis (misalignment)
	float zangle;	//detector angle about z-axis (misalignment)
	float M;		// magnification of the camera-lens system

	int LoadFromConfigFile(std::string FilePath)
	{
		ConfigFile configfile(FilePath);
		
		this->R  = configfile.read<float>("R");
		this->Rf = configfile.read<float>("Rf");
		this->Ry = configfile.read<float>("Ry");
		this->Dx = configfile.read<float>("Dx");
		this->Dz = configfile.read<float>("Dz");
		this->xangle = configfile.read<float>("xangle");
		this->yangle = configfile.read<float>("yangle");
		this->zangle = configfile.read<float>("zangle");
		this->M	= configfile.read<float>("M");
		return 0;
	}
};

struct Detector
{
	unsigned int NumTAxPixels;
	unsigned int NumAxPixels;
	float TAxPixDimMm;
	float AxPixDimMm;

	int LoadFromConfigFile(std::string FileName)
	{
		ConfigFile configfile(FileName);
		this->NumTAxPixels	= configfile.read<unsigned int>("TAxPixels");
		this->NumAxPixels	= configfile.read<unsigned int>("AxPixels");
		this->TAxPixDimMm	= configfile.read<float>("TAxPixDim");
		this->AxPixDimMm	= configfile.read<float>("AxPixDim");
		return 0;
	}
};

struct ScanParameters
{
	unsigned int N_iteration;
	unsigned int NumProj;
	float DeltaAng;
	

	int LoadFromConfigFile(std::string FilePath)
	{
		ConfigFile configfile(FilePath);

		this->N_iteration	= configfile.read<unsigned int>("N_iteration");
		this->NumProj	= configfile.read<unsigned int>("NumProj");
		this->DeltaAng	= configfile.read<float>("DeltaAng");
		return 0;
	}
};

struct VoxVolParam
{
	unsigned int Xdim; // Number of x voxels
	unsigned int Ydim; // Number of y voxels
	unsigned int Zdim; // Number of z voxels
	float XVoxSize;		// voxel size in x direction (inches)
	float YVoxSize;		// voxel size in y direction (inches)
	float ZVoxSize;		// voxel size in z direction (inches)

	int LoadFromConfigFile(std::string FilePath)
	{
		ConfigFile configfile(FilePath);

		this->Xdim		= configfile.read<unsigned int>("Xdim");
		this->Ydim		= configfile.read<unsigned int>("Ydim");
		this->Zdim		= configfile.read<unsigned int>("Zdim");
		this->XVoxSize	= configfile.read<float>("XVoxSize");
		this->YVoxSize	= configfile.read<float>("YVoxSize");
		this->ZVoxSize	= configfile.read<float>("ZVoxSize");
		return 0;
	}
};

struct ObjectParam
{
	float ObjX;
	float ObjY;
	float ObjZ;
	float ObjXangle;
	float ObjYangle;
	float ObjZangle;

	int LoadFromConfigFile(std::string FilePath)
	{
		ConfigFile configfile(FilePath);
		this->ObjX		= configfile.read<float>("ObjX");
		this->ObjY		= configfile.read<float>("ObjY");
		this->ObjZ		= configfile.read<float>("ObjZ");
		this->ObjXangle	= configfile.read<float>("ObjXangle");
		this->ObjYangle	= configfile.read<float>("ObjYangle");
		this->ObjZangle	= configfile.read<float>("ObjZangle");
		return 0;
	}
};

struct FilePaths
{
	std::string ProjFileFolder;
	std::string ProjFileNameRoot;
	std::string ProjFileSuffix;
	std::string ReconFileFolder;
	std::string ReconFileNameRoot;
	std::string ReconFileSuffix;
	std::string sim_ObjFileFolder;
	std::string sim_ObjFileName;
	std::string sim_BpFileNameRoot;
	std::string sim_BpFileSuffix;

	int LoadFromconfigFile(std::string FilePath)
	{
		ConfigFile configfile(FilePath);
		this->ProjFileFolder	= configfile.read<std::string>("ProjFileFolder");
		this->ProjFileNameRoot	= configfile.read<std::string>("ProjFileNameRoot");
		this->ProjFileSuffix	= configfile.read<std::string>("ProjFileSuffix");
		this->ReconFileFolder	= configfile.read<std::string>("ReconFileFolder");
		this->ReconFileNameRoot	= configfile.read<std::string>("ReconFileNameRoot");
		this->ReconFileSuffix	= configfile.read<std::string>("ReconFileSuffix");
		this->sim_ObjFileFolder	= configfile.read<std::string>("ObjFileFolder");
		this->sim_ObjFileName	= configfile.read<std::string>("ObjFileName");
		this->sim_BpFileNameRoot= configfile.read<std::string>("BpFileNameRoot");
		this->sim_BpFileSuffix	= configfile.read<std::string>("BpFileSuffix");
		return 0;
	}
};

struct CTParameters
{
	CalibrationParam CalParam;
	Detector Det;
	//ScanParameters ScanParam;
	VoxVolParam VoxParam;
	ObjectParam ObjParam;
	FilePaths filenames;

	int LoadFromConfigFile(std::string FilePath)
	{
		this->CalParam.LoadFromConfigFile(FilePath);
		this->Det.LoadFromConfigFile(FilePath);
		//this->ScanParam.LoadFromConfigFile(FilePath);
		this->VoxParam.LoadFromConfigFile(FilePath);
		this->ObjParam.LoadFromConfigFile(FilePath);
		this->filenames.LoadFromconfigFile(FilePath);

		return 0;
	}

};

struct CTangles
{
	float initial_angle;
	float delta_angle;
	int num_proj;

	int create(float initial_angle, float delta_angle, int num_proj)
	{
		this->initial_angle = initial_angle;
		this->delta_angle = delta_angle;
		this->num_proj = num_proj;

		return 0;
	}

};

struct objR
{
	float thetaXobj;
	float thetaYobj;
	float thetaZobj;
};

struct detR
{
	float thetaXdet;
	float thetaYdet;
	float thetaZdet;
};

struct WriteToParamFile
{
	CalibrationParam CalParam;
	ObjectParam ObjParam;

	void WriteToFile(std::string outputfilename)
	{
		std::ofstream file;
		file.precision(3);
		file.setf( std::ios::fixed, std:: ios::floatfield ); // floatfield set to fixed
		file.open(outputfilename);

		file << "############################################### \n";
		file << "####### Calibration Parameters ################ \n";
		file << "############################################### \n\n";
		file << "R	= " << CalParam.R << "\n";
		file << "Rf = " << CalParam.Rf << "\n";
		file << "Ry	= " << CalParam.Ry<< "\n";
		file << "Dx = " << CalParam.Dx << "\n";
		file << "Dz = " << CalParam.Dz << "\n";
		file << "xangle = " << CalParam.xangle << "\n";
		file << "yangle = " << CalParam.yangle << "\n";
		file << "zangle = " << CalParam.zangle << "\n";
		file << "M = " << CalParam.M << "\n";
		file << "\n";
		file.close();
	}
};



#endif
