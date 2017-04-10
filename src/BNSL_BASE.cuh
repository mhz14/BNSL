/*
 * BNSL_BASE.cuh
 *
 *  Created on: 2017年4月6日
 *      Author: mark
 */

#ifndef BNSL_BASE_CUH_
#define BNSL_BASE_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <float.h>

__host__ void CheckCudaError(cudaError_t err, char const* errMsg);
#define CUDA_CHECK_RETURN(value1, value2) CheckCudaError(value1, value2)

#define SAMPLES_PATH "data/alarm.sample"
#define NODEINFO_PATH "data/alarm.info"
#define ALPHA 10.0
#define GAMMA 0.2
#define CONSTRAINTS 4
#define MAX_THREAD_NUM 1024

__device__ int findIndex_kernel(int k, int* combi, int nodesNum);
__device__ long C_kernel(int n, int m);
__device__ void recoverComb_kernel(int vi, int* combi, int size);
__device__ void findComb_kernel(int nodesNum, int index, int* size, int* combi);
__device__ void sortArray_kernel(int * s, int n);
__device__ double calcLocalScore_kernel(int * dev_valuesRange,
		int *dev_samplesValues, int *dev_N, int samplesNum, int size,
		int* parentSet, int curNode, int nodesNum, int valuesMaxNum);
__global__ void calcAllLocalScore_kernel(int *dev_valuesRange,
		int *dev_samplesValues, int *dev_N, double * dev_lsTable,
		int samplesNum, int nodesNum, int allParentSetNumPerNode,
		int valuesMaxNum);
__global__ void calcOrderScore_kernel(double * dev_lsTable, int * dev_order,
		double * dev_bestNodeScore, int * dev_bestParentSet,
		int allParentSetNumPerNode, int nodesNum);

__host__ void BNSL_init();
__host__ void BNSL_calcLocalScore();
__host__ void BNSL_start();
__host__ void BNSL_printResult();
__host__ void BNSL_finish();
__host__ void readNodeInfo();
__host__ void readSamples();
__host__ int compare(const void*a, const void*b);
__host__ long C(int n, int m);
__host__ bool nextCombination(int* s, int n, int r);
__host__ void randInitOrder(int * s);
__host__ void selectTwoNodeToSwap(int *n1, int *n2);
__host__ void randSwapTwoNode(int *order);
// calculate max different values number for all pair of child and parent set
__host__ int calcValuesMaxNum();
__host__ void calcCPUTimeStart(char const *message);
__host__ void calcCPUTimeEnd();

__host__ int getBlockNum(int parentSetNum);
__host__ int getParentSetNumInOrder(int curPos);

#endif /* BNSL_BASE_CUH_ */
