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
	
	//---------------- how to use siddon_recon class --------------------------------
	siddon_recon recon;
	//recon.a1_FORWARD_PROJECTION("H:\\Visual Studio 2010\\CT_Recon\\CT_Recon", "MasterParameterFile.cfg");
	recon.a0_RECON_MLEM("H:\\Visual Studio 2010\\CT_Recon\\CT_Recon\\", "MasterParameterFile.cfg");
	//recon.a1_backward_projection("H:\\Visual Studio 2010\\CT_Recon\\CT_Recon", "MasterParameterFile.cfg", true);
	//-------------------------------------------------------------------
	
	sdkStopTimer(&timer);
	totalTime = sdkGetTimerValue(&timer)*1e-3;
	printf("calculation time = %f seconds \n", totalTime);

	cudaProfilerStop();

	
	
	system("PAUSE");
}