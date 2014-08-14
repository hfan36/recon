//Written by Helen Fan
//some codes are taken from cuda SDK examples, because I don't feel like rewriting code that's already there
//use the template "sum" , use "allocate_for_sum" first to allocate the necessary arrays required to do gpu summation
//then once the arrays are allocated, you can use gpu_sum to insert the device array and it'll return the result.
//the timer isn't actually used, but since the example code called for it, and I didn't want to modify it too much, i left it there
//do your timing outside the class if you want.
//you will use allocate_for_sum first, then if you want to 
//add multiple arrays with in a loop, it doesn't allocate over and over again.


#ifndef GPU_SUM_H
#define GPU_SUM_H

#include "helper_cuda.h"
#include "helper_functions.h"

#include <algorithm>
#include "reduction.h"
#include <vector>

extern "C"
bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

//copied from cuda SDK 5.5
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{

    //get device capability, to avoid block/grid size excceed the upbound
    cudaDeviceProp prop;
    int device;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&prop, device));

    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }

    if ((float)threads*blocks > (float)prop.maxGridSize[0] * prop.maxThreadsPerBlock)
    {
        printf("n is too large, please choose a smaller number!\n");
    }

    if (blocks > prop.maxGridSize[0])
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, prop.maxGridSize[0], threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    if (whichKernel == 6)
    {
        blocks = MIN(maxBlocks, blocks);
    }
}

////////////////////////////////////////////////////////////////////////////////
// This function performs a reduction of the input data multiple times and
// measures the average reduction time.
// also copied from cuda SDK 5.5
////////////////////////////////////////////////////////////////////////////////
template <class T>
T benchmarkReduce(int  n,
                  int  numThreads,
                  int  numBlocks,
                  int  maxThreads,
                  int  maxBlocks,
                  int  whichKernel,
                  int  testIterations,
                  bool cpuFinalReduction,
                  int  cpuFinalThreshold,
                  StopWatchInterface *timer,
                  T *h_odata,
                  T *d_idata,
                  T *d_odata)
{
    T gpu_result = 0;
    bool needReadBack = true;

    for (int i = 0; i < testIterations; ++i)
    {
        gpu_result = 0;

        cudaDeviceSynchronize();
        //sdkStartTimer(&timer);

        // execute the kernel
        reduce<T>(n, numThreads, numBlocks, whichKernel, d_idata, d_odata);

        // check if kernel execution generated an error
        getLastCudaError("Kernel execution failed");

        if (cpuFinalReduction)
        {
            // sum partial sums from each block on CPU
            // copy result from device to host
            checkCudaErrors(cudaMemcpy(h_odata, d_odata, numBlocks*sizeof(T), cudaMemcpyDeviceToHost));

            for (int i=0; i<numBlocks; i++)
            {
                gpu_result += h_odata[i];
            }

            needReadBack = false;
        }
        else
        {
            // sum partial block sums on GPU
            int s=numBlocks;
            int kernel = whichKernel;

            while (s > cpuFinalThreshold)
            {
                int threads = 0, blocks = 0;
                getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

                reduce<T>(s, threads, blocks, kernel, d_odata, d_odata);

                if (kernel < 3)
                {
                    s = (s + threads - 1) / threads;
                }
                else
                {
                    s = (s + (threads*2-1)) / (threads*2);
                }
            }

            if (s > 1)
            {
                // copy result from device to host
                checkCudaErrors(cudaMemcpy(h_odata, d_odata, s * sizeof(T), cudaMemcpyDeviceToHost));

                for (int i=0; i < s; i++)
                {
                    gpu_result += h_odata[i];
                }

                needReadBack = false;
            }
        }

        cudaDeviceSynchronize();
        //sdkStopTimer(&timer);
    }

    if (needReadBack)
    {
        // copy final sum from device to host
        checkCudaErrors(cudaMemcpy(&gpu_result, d_odata, sizeof(T), cudaMemcpyDeviceToHost));
    }

    return gpu_result;
}


//written by Helen Fan using the above code copied from cuda SDK 5.5
// sum device array, for detail, refer to the project: "reduction", from 
// cuda examples v5.5
template <class T>
class gpu_sum
{
public:
	void allocate_for_sum(unsigned int size); //use this first, 
	void reset_memory(unsigned int size);
	T sum(unsigned int size, T *d_idata, StopWatchInterface *timer); //actual summing code, timer doesn't matter, but since I 
															//the example code called for it, just initialize it. 
															//timer is disabled in the code, use outside to time function;

	~gpu_sum(); //delete class

protected:
	int m_blocks, m_threads;
	static const int m_maxBlocks = 40000; //can use, but I dont' care so i just set a large enough number, change it if you need more
	static const int m_maxThreads = 256; // i use 256 just coz i can, so sue me.  
	static const int m_whichKernel = 6; //6 is the fastest method, 1 is the worst.
	static const int testIterations = 1; //just doing 1, why do I need to do multiple when I'm not timing how fast it executes over and over?
	static const bool cpuFinalReduction = false; //use gpu to sum the blocks, 
	static const int cpuFinalThreshold = 1; //just 1 because the example said so,
	
	T *h_odata;
	T *d_odata;
};

template <class T>
T gpu_sum<T>::sum(unsigned int size, T *d_idata, StopWatchInterface *timer)
{
	T gpu_result = 0;
	gpu_result = benchmarkReduce<T>(size, m_threads, m_blocks, m_maxThreads, m_maxBlocks, m_whichKernel, testIterations, cpuFinalReduction,
						cpuFinalThreshold, timer, h_odata, d_idata, d_odata);
	return gpu_result;
}

template <class T>
void gpu_sum<T>::allocate_for_sum(unsigned int size)
{
	getNumBlocksAndThreads(m_whichKernel, size, m_maxBlocks, m_maxThreads, m_blocks, m_threads);
	h_odata = (T*)malloc(m_blocks*sizeof(T));
	checkCudaErrors( cudaMalloc( (void**)&d_odata, size*sizeof(T)) );
}

template <class T>
void gpu_sum<T>::reset_memory(unsigned int size)
{
	checkCudaErrors( cudaMemset(d_odata, 0, size*sizeof(T)) );
	memset(h_odata, 0, m_blocks*sizeof(T));
}


template <class T>
gpu_sum<T>::~gpu_sum()
{
	free(h_odata);
	checkCudaErrors(cudaFree(d_odata));
}

#endif
