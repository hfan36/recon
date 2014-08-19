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
	recon.a1_forward_projection("C:\\Users\\Ryan-Helen\\Documents\\Visual Studio 2010\\Projects\\recon\\CT_Recon", "CTParameters_042414.h");
	//recon.a0_recon_mlem("H:\\Visual Studio 2010\\CT_Recon\\CT_Recon\\ball", "fp_ball64.cfg");
	//recon.a1_backward_projection("H:\\Visual Studio 2010\\CT_Recon\\CT_Recon\\ball", "fp_ball64.cfg", true);
	//-------------------------------------------------------------------
	

	sdkStopTimer(&timer);
	totalTime = sdkGetTimerValue(&timer)*1e-3;
	printf("calculation time = %f seconds \n", totalTime);

	cudaProfilerStop();

	
	
	system("PAUSE");
}