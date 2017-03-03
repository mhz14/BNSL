#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

void CheckCudaError(cudaError_t err, char const* errMsg);
#define CUDA_CHECK_RETURN(value1, value2) CheckCudaError(value1, value2)

#define SAMPLES_PATH "data/alarm.sample"
#define NODEINFO_PATH "data/alarm.info"
#define ALPHA 10.0
#define GAMMA 0.2
#define CONSTRAINTS 4
#define PARENT_VALUE_MAX_NUM 512

__device__ int findIndex_kernel(int k, int* combi, int nodesNum);

__device__ long C_kernel(int n, int m);

__device__ void recoverComb_kernel(int vi, int* combi, int size);

__device__ void findComb_kernel(int nodesNum, int index, int* size, int* combi);

__device__ double calLocalScore_kernel(int * dev_valuesRange, int *dev_samplesValues, int samplesNum, int size, int* parentSet, int curNode, int nodesNum);

__global__ void calAllLocalScore_kernel(int *dev_valuesRange, int *dev_samplesValues, int *dev_N, double * dev_lsTable, int samplesNum, int nodesNum, int parentSetNum);

__global__ void calTopologyScore_kernel(double * dev_lsTable, int * dev_order, double * dev_nodeScore, int * dev_bestParentSet, int parentSetNum, int nodesNum);

__device__ void sortArray_kernel(int * s, int n);

void BNSL_init();

void BNSL_calLocalScore();

void BNSL_start();

void BNSL_finish();

void readNodeInfo();

void readSamples();

long C(int n, int m);

bool nextCombination(int* s, int n, int r);

void randInitOrder(int * s);

void calcTime_start(int type);
void calcTime_end(int type);
