#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

void CheckCudaError(cudaError_t err, char const* errMsg);
#define CUDA_CHECK_RETURN(value1, value2) CheckCudaError(value1, value2)

#define SAMPLES_PATH "data/burglar.sample"
#define NODEINFO_PATH "data/burglar.info"
#define ALPHA 10.0
#define GAMMA 0.2
#define CONSTRAINTS 2

__device__ int findIndex_kernel(int k, int* combi, int nodesNum);

__device__ long C_kernel(int n, int m);

__device__ void recoverComb_kernel(int vi, int* combi, int size);

__device__ void findComb_kernel(int nodesNum, int index, int* size, int* combi);

__device__ double calLocalScore_kernel(int * dev_valuesRange,
		int *dev_samplesValues, int samplesNum, int size, int* parentSet,
		int curNode, int nodesNum, int valuesMaxNum);

__global__ void calAllLocalScore_kernel(int *dev_valuesRange,
		int *dev_samplesValues, int *dev_N, double * dev_lsTable,
		int samplesNum, int nodesNum, int allParentSetNumPerNode, int valuesMaxNum);

__global__ void calOrderScore_kernel(double * dev_lsTable, int * dev_order,
		double * dev_nodeScore, int * dev_bestParentSet, int allParentSetNumPerNode,
		int nodesNum);

__device__ void sortArray_kernel(int * s, int n);

void BNSL_init();

void BNSL_calcLocalScore();

void BNSL_start();

void BNSL_finish();

void readNodeInfo();

void readSamples();

int compare(const void*a, const void*b);

long C(int n, int m);

bool nextCombination(int* s, int n, int r);

void randInitOrder(int * s);

void selectTwoNodeToSwap(int *n1, int *n2);

void randSwapTwoNode(int *order);

// calculate max different values number for all pair of child and parent set
int calcValuesMaxNum();

void calcCPUTimeStart(char const *message);
void calcCPUTimeEnd();
