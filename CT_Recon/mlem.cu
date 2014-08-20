#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <Windows.h>
#include <cuda_profiler_api.h>
#include "hf_siddon_recon.hpp"

//maybe I should include how long it took to run the program in the parameter file???
//call it .info file instead of .cfg
//.info - generated using siddon_recon class, saves the information that was used to create files
//.cfg - used as input for the siddon_recon class for recon/projection/backprojection
//it's nice to distinguish the two.  Can you make a file that's read only? so I don't try to modify it??
int main()
{
	cudaProfilerStart();

	double totalTime;
	StopWatchInterface *timer;
	sdkCreateTimer(&timer);
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
	
	//---------------- how to use siddon_recon class --------------------------------
	siddon_recon recon;
	//recon.a1_FORWARD_PROJECTION("H:\\Visual Studio 2010\\CT_Recon\\CT_Recon", "CTParameters_042414.h");
	recon.a0_RECON_MLEM("H:\\Visual Studio 2010\\CT_Recon\\CT_Recon\\", "CTParameters_042414.h");
	//recon.a1_backward_projection("H:\\Visual Studio 2010\\CT_Recon\\CT_Recon", "MasterParameterFile.cfg", true);
	//-------------------------------------------------------------------
	
	sdkStopTimer(&timer);
	totalTime = sdkGetTimerValue(&timer)*1e-3;
	printf("calculation time = %f seconds \n", totalTime);

	cudaProfilerStop();

	
	
	system("PAUSE");
}