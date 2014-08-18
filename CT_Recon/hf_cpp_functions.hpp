#ifndef HF_CPP_FUNCTIONS_HPP_
#define HF_CPP_FUNCTIONS_HPP_

#include <Windows.h>
#include <string>

HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
COORD CursorPosition;
CONSOLE_SCREEN_BUFFER_INFO csbi;
PCONSOLE_SCREEN_BUFFER_INFO pcsbi=&csbi;


//got it online somewhere...
void CursorGotoXY(int x, int y) 
{ 
	CursorPosition.X = x; 
	CursorPosition.Y = y; 
	SetConsoleCursorPosition(console,CursorPosition); 
}

//got it online somewhere...
void CursorGotoXY(int x, int y, string text) 
{ 

	CursorPosition.X = x; 
	CursorPosition.Y = y; 
	SetConsoleCursorPosition(console,CursorPosition);
	std::cout << text;
}

void CursorGetXY(int &x, int &y)
{
	GetConsoleScreenBufferInfo(console, pcsbi);
	x = csbi.dwCursorPosition.X;
	y = csbi.dwCursorPosition.Y;
}

struct cursor_position
{
	int cursorX, cursorY;

	void CursorGotoXY(int cursorX, int cursorY)
	{
		CursorPosition.X = cursorX; 
		CursorPosition.Y = cursorY; 
		SetConsoleCursorPosition(console,CursorPosition); 
	}

	void CursorGotoXY(int cursorX, int cursorY, std::string text) 
	{ 
		CursorPosition.X = cursorX; 
		CursorPosition.Y = cursorY; 
		SetConsoleCursorPosition(console,CursorPosition);
		std::cout << text;
	}

	void CursorGetXY()
	{
		GetConsoleScreenBufferInfo(console, pcsbi);
		this->cursorX = csbi.dwCursorPosition.X;
		this->cursorY = csbi.dwCursorPosition.Y;
	}
};


template<class T> 
class linspace //same as matlab's linspace function
{
public:
	linspace(){};
	void operator() (T* larray, T initialValue, T endValue, int size);
	~linspace(){};
	
private:
};

template<class T>
void linspace<T>::operator()(T *larray, T initialValue, T endValue, int size)
{
	T increment;
	int i;
	if (size >=2)
	{
		larray[0] = initialValue;
		increment = (endValue - initialValue)/((T)size - (T)1.0);
		for (i = 0; i < size; i++)
		{
			larray[i] = larray[0] + i*increment;
		}
	}
	else if(size == 1)
	{
		larray[0] = initialValue;
	}
	else if(size == 0)
	{
		larray[0] = (T)0.0;
	}
	else
	{
		std::cout << "cannot make array, check input parameters" << std::endl;
	}
}

float degree_to_rad(float degrees)
{
	float rad = degrees * 3.14f/180.0f;		
	return rad;
}


string create_filename(std::string folder_root, std::string filename, int image_number)
{
	std::ostringstream num;
	num << image_number;
	return (folder_root+filename+"_"+num.str() +".bin");
}

string create_filename(std::string folder_root, std::string filename)
{
	return (folder_root+filename);
}

string create_filename(std::string folder_root, std::string filename, int image_number, std::string suffix)
{
	std::ostringstream num;
	num << image_number;
	return (folder_root+filename+"_"+num.str() + suffix);
}

enum output_type {write_to_file, do_not_write}; 

inline bool file_exist(const std::string filename)
{
	if (FILE *fp = fopen(filename.c_str(), "rb"))
	{
		fclose(fp);
		return true;
	}
	else
	{
		std::cout << "Error: " << filename << " does not exist!" << std::endl;
		return false;
	}
}

std::ifstream::pos_type file_size(const char* filename)
{
    std::ifstream in(filename, std::ifstream::in | std::ifstream::binary);
    in.seekg(0, std::ifstream::end);
    return in.tellg(); 
}



struct mlem_input_values
{
	unsigned int N_iterations;
	unsigned int total_projection_images;
	float delta_angle_deg;
	std::string filename_root;
	std::string suffix;
	std::string folder_root;

	void create(unsigned int N_iterations, unsigned int total_projection_images, float delta_angle_deg,
				std::string folder_root, std::string filename_root, std::string suffix)
	{
		this->N_iterations = N_iterations;
		this->total_projection_images = total_projection_images;
		this->delta_angle_deg = delta_angle_deg;
		this->filename_root = filename_root;
		this->folder_root = folder_root;
		this->suffix = suffix;
	}
};



struct osem_input_values
{
	unsigned int N_iterations;
	unsigned int N_subsets;
	unsigned int total_projection_images;
	std::string filename_root;
	std::string folder_root;
	std::string suffix;
	
	void create(unsigned int N_iterations, unsigned int N_subsets, unsigned int total_projection_images,
				std::string filename_root, std::string suffix)
	{
		this->N_iterations = N_iterations;
		this->N_subsets = N_subsets;
		this->total_projection_images = total_projection_images;
		this->filename_root = filename_root;
		this->suffix = suffix;
	}
};


struct recon_values
{
	unsigned int N_iterations;
	unsigned int N_subsets;
	CTangles subset_angles;

	unsigned int total_projection_images;
	std::string filename_root;
	std::string suffix;
	std::string folder_root;


	void create(unsigned int N_iterations, unsigned int N_subsets, 
				float subset_initial_angle_deg, float subset_delta_angle_deg,
				unsigned int subset_N_projections, unsigned int total_projection_images, 
				std::string filename_root, std::string suffix)
	{
		this->N_iterations = N_iterations;
		this->N_subsets = N_subsets;
		this->subset_angles.initial_angle = subset_initial_angle_deg;
		this->subset_angles.delta_angle = subset_delta_angle_deg;
		this->subset_angles.num_proj = subset_N_projections;

		this->total_projection_images = total_projection_images;
		this->filename_root = filename_root;
		this->suffix = suffix;
	}

};

struct osem_subset_inputs
{
	float subset_initial_angle_deg;
	float subset_delta_angle_deg;
	unsigned int subset_N_projections;
	unsigned int subset_iterator;
};

inline bool check_projection_files(unsigned int total_projection_images, Detector det, std::string folder_root, std::string filename_root, std::string suffix)
{
	int image_size = det.NumAxPixels*det.NumTAxPixels*sizeof(float);
	std::string filename;
	for (unsigned int i = 0; i < total_projection_images; i++)
	{
		filename = create_filename(folder_root, filename_root, i, suffix);

		if (!file_exist(filename))
		{
			std::cout << "error: file => " << filename << "does not exist!" << std::endl;
			return false;
		}
		if ( image_size != file_size(filename.c_str()) )
		{
			std::cout << "error: file size does not match with parameter file!" << std::endl;
			return false;
		}
	}

	return true;
}


void correct_folder_path(std::string &folderpath)
{
	if ( !folderpath.empty() )
	{
		std::size_t pos = folderpath.find("\\");
		while(pos != std::string::npos)
		{
			folderpath.replace(pos, 1, "/");
			pos = folderpath.find("\\");
		}

		if (folderpath.compare(folderpath.size()-1, 1, "/") != 0)
			folderpath.append("/");
	}
	
}

bool directory_exist(const std::string directory_name)
{
	DWORD ftyp = GetFileAttributes(directory_name.c_str());
	if (ftyp == -1 )
	{
		//std::cout << "This directory does NOT exist: " << directory_name << std::endl;
		return false;
	}
	else if (ftyp & FILE_ATTRIBUTE_DIRECTORY)
	{
		return true;
	}
	else
	{
		//std::cout << "This directory does NOT exist: " << directory_name << std::endl;
		return false;
	}
}



bool create_directory(const std::string directory_name)
{
	int tf = directory_exist(directory_name);
	SECURITY_ATTRIBUTES sa;
	sa.nLength = sizeof(SECURITY_ATTRIBUTES);
	sa.lpSecurityDescriptor = NULL;
	sa.bInheritHandle = FALSE;

	char ans;
	if (tf)
	{
		//If The file exist, then return false (directory was not created)
		return (true);
	}
	else 
	{
		std::cout << "This directory does NOT exist: " << directory_name << std::endl;
		std::cout << "Do you want to create it? (y/n): ";
		std::cin >> ans;

		if ( !std::cin.fail() && ans == 'Y' || ans == 'y' )
		{
			CreateDirectory( (LPCSTR)directory_name.c_str(), &sa );
			std::cout << "directory created" << std::endl;
			return (true);
		}
		else
		{
			std::cout << "directory was not created" << std::endl;
			return (false);
		}
	}
}


std::string fix_configfile_suffix(std::string &filename)
{
	std::size_t pos = filename.find(".");
	if (pos != std::string::npos)
	{
		filename.replace(pos+1, filename.size()-pos-1, "cfg");
	}
	else
	{
		filename.append(".cfg");
	}

	return(filename);
}

bool check_recon_inputs(osem_subset_inputs &osem_subset, osem_input_values osem_inputs)
{
	int check = osem_inputs.total_projection_images % osem_inputs.N_subsets;
	if (check == 0)
	{
		osem_subset.subset_N_projections = osem_inputs.total_projection_images / osem_inputs.N_subsets;
		osem_subset.subset_delta_angle_deg = 360.0f / ( (float)osem_subset.subset_N_projections* (float)osem_inputs.N_subsets );
		osem_subset.subset_initial_angle_deg = 0.0f;
		osem_subset.subset_iterator = 0;

		return true;
	}
	else
	{
		std::cout << "Error: The input values does not agree with each other." << std::endl;
		return false;
	}
}

bool check_recon_inputs(mlem_input_values mlem_inputs)
{
	float calculated_delta_angle = 360.0f / (float)mlem_inputs.total_projection_images;


	if (calculated_delta_angle != mlem_inputs.delta_angle_deg)
	{
		return false;
	}
	else
	{
		return true;
	}
}


inline bool empty_hf(CTParameters &params, CTangles &angles)
{
	if (angles.num_proj != 0 && params.Det.NumAxPixels != 0 && params.Det.NumTAxPixels != 0)
	{
		return false; //if everything is not zeros, return false, so stuff is NOT empty
	}
	else
	{
		return true; //else, return true, so stuff IS empty
	}


}


#endif