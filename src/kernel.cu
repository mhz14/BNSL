#include "BNSL_GPU.cuh"

int * valuesRange;

int nodesNum = 0;

int * samplesValues;

int samplesNum;

int parentSetNum;

double * dev_lsTable;

int* globalBestGraph;
int* topSort;
double globalBestScore;

int begin = 0;
cudaEvent_t start, stop;

int main() {
	calcTime_start(2);
	BNSL_init();
	calcTime_end(2);
	calcTime_start(2);
	BNSL_calLocalScore();
	calcTime_end(2);
	calcTime_start(2);
	BNSL_start();
	calcTime_end(2);
	printf("Bayesian Network learned:\n");
	for (int i = 0; i < nodesNum; i++) {
		for (int j = 0; j < nodesNum; j++) {
			printf("%d ", globalBestGraph[i * nodesNum + j]);
		}
		printf("\n");
	}
	printf("Best Score: %f \n", globalBestScore);
	printf("Best Topology: ");
	for (int i = 0; i < nodesNum; i++) {
		printf("%d ", topSort[i]);
	}
	printf("\n");

	calcTime_start(2);
	BNSL_finish();
	calcTime_end(2);
	return 0;
}

void CheckCudaError(cudaError_t err, char const* errMsg) {
	if (err == cudaSuccess)
		return;
	printf("%s\nError Message: %s.\n", errMsg, cudaGetErrorString(err));
	system("pause");
	exit(EXIT_FAILURE);
}

void calcTime_start(int type) {
	if (type == 1) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, NULL);
	} else {
		begin = clock();
	}
}

void calcTime_end(int type) {
	if (type == 1) {
		float time = 0;
		cudaEventRecord(stop, NULL);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		printf("Elapsed time is %fms\n", time);
	} else {
		printf("Elapsed time is %dms\n", (int) clock() - begin);
	}
}

void BNSL_init() {
	readNodeInfo();
	readSamples();
}

void BNSL_calLocalScore() {

	int i;
	parentSetNum = 0;
	for (i = 0; i <= CONSTRAINTS; i++) {
		parentSetNum = parentSetNum + C(i, nodesNum - 1);
	}

	int* dev_valuesRange;
	int* dev_samplesValues;
	int* dev_N;

	CUDA_CHECK_RETURN(cudaDeviceReset(), "reset error.");

	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_lsTable, nodesNum * parentSetNum * sizeof(double)),
			"dev_lsTable cudaMalloc failed.");
	CUDA_CHECK_RETURN(cudaMalloc(&dev_valuesRange, nodesNum * sizeof(int)),
			"dev_valuesRange cudaMalloc failed.");
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_samplesValues, samplesNum * nodesNum * sizeof(int)),
			"dev_samplesValues cudaMalloc failed.");
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_N, nodesNum * parentSetNum * PARENT_VALUE_MAX_NUM * sizeof(int)),
			"dev_N cudaMalloc failed.");

	CUDA_CHECK_RETURN(
			cudaMemcpy(dev_valuesRange, valuesRange, nodesNum * sizeof(int),
					cudaMemcpyHostToDevice),
			"valuesRange -> dev_valuesRange failed.");
	CUDA_CHECK_RETURN(
			cudaMemcpy(dev_samplesValues, samplesValues,
					samplesNum * nodesNum * sizeof(int),
					cudaMemcpyHostToDevice),
			"samplesValues -> dev_samplesValues failed.");
	CUDA_CHECK_RETURN(cudaGetLastError(),
			"calAllLocalScore_kernel launch failed.");
	CUDA_CHECK_RETURN(cudaDeviceSynchronize(),
			"calAllLocalScore_kernel failed on running.");

	calAllLocalScore_kernel<<<1, 256>>>(dev_valuesRange, dev_samplesValues,
			dev_N, dev_lsTable, samplesNum, nodesNum, parentSetNum);
	CUDA_CHECK_RETURN(cudaGetLastError(),
			"calAllLocalScore_kernel launch failed.");
	CUDA_CHECK_RETURN(cudaDeviceSynchronize(),
			"calAllLocalScore_kernel failed on running.");
	/*
	 double * lsTable = (double *)malloc(nodesNum * parentSetNum * sizeof(double));
	 CUDA_CHECK_RETURN(cudaMemcpy(lsTable, dev_lsTable, nodesNum * parentSetNum * sizeof(double), cudaMemcpyDeviceToHost), "dev_lsTable -> lsTable");
	 double sum = 0.0;
	 for (int i = 0; i < nodesNum; i++){
	 for (int j = 0; j < parentSetNum; j++){
	 sum += lsTable[i * parentSetNum + j];
	 }
	 }
	 printf("%f\n", sum);
	 */
	// �ͷ���GPU�з�����ڴ�ռ�
	CUDA_CHECK_RETURN(cudaFree(dev_valuesRange),
			"dev_valuesRange cudaFree failed.");
	CUDA_CHECK_RETURN(cudaFree(dev_samplesValues),
			"dev_samplesValues cudaFree failed.");
	CUDA_CHECK_RETURN(cudaFree(dev_N), "dev_N cudaFree failed.");

	// ���մ���������ݵ��ڴ�
	free(valuesRange);
	free(samplesValues);
}

void BNSL_start() {
	int i, j, swap, r1, r2, iter;
	double oldScore = -DBL_MAX;
	globalBestScore = -DBL_MAX;
	globalBestGraph = (int *) malloc(sizeof(int) * nodesNum * nodesNum);

	topSort = (int *) malloc(sizeof(int) * nodesNum);

	double newScore = 0.0;

	double * nodeScore = (double *) malloc(nodesNum * sizeof(double));
	int * bestParentSet = (int *) malloc(
			(CONSTRAINTS + 1) * nodesNum * sizeof(int));

	int * dev_order;
	int * dev_bestParentSet;
	double * dev_nodeScore;

	CUDA_CHECK_RETURN(cudaMalloc(&dev_order, nodesNum * sizeof(int)),
			"dev_order cudaMalloc failed.");
	CUDA_CHECK_RETURN(cudaMalloc(&dev_nodeScore, nodesNum * sizeof(double)),
			"dev_nodeScore cudaMalloc failed.");
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_bestParentSet, nodesNum * (CONSTRAINTS + 1) * sizeof(int)),
			"dev_bestParentSet cudaMalloc failed.");

	// ���ѡ��һ����������
	int * oldOrder = (int *) malloc(sizeof(int) * nodesNum);
	randInitOrder(oldOrder);

	int * newOrder = (int *) malloc(sizeof(int) * nodesNum);

	// MCMCѭ��
	int maxIterNum = 1;
	for (iter = 0; iter < maxIterNum; iter++) {
		// ����������������Ϊ�µ���������
		for (i = 0; i < nodesNum; i++) {
			newOrder[i] = oldOrder[i];
		}
		r1 = rand() % nodesNum;
		if (r1 == nodesNum - 1) {
			r2 = rand() % (nodesNum - 1);
			swap = newOrder[r1];
			newOrder[r1] = newOrder[r2];
			newOrder[r2] = swap;
		} else {
			swap = newOrder[r1];
			newOrder[r1] = newOrder[nodesNum - 1];
			newOrder[nodesNum - 1] = swap;

			r2 = rand() % (nodesNum - 1);
			if (r1 != r2) {
				swap = newOrder[r1];
				newOrder[r1] = newOrder[r2];
				newOrder[r2] = swap;

				swap = newOrder[r2];
				newOrder[r2] = newOrder[nodesNum - 1];
				newOrder[nodesNum - 1] = swap;
			}
		}

		newOrder[0] = 9;
		newOrder[1] = 8;
		newOrder[2] = 3;
		newOrder[3] = 5;
		newOrder[4] = 11;
		newOrder[5] = 4;
		newOrder[6] = 2;
		newOrder[7] = 1;
		newOrder[8] = 10;
		newOrder[9] = 7;
		newOrder[10] = 6;

		// ��ʼ��newOrder�ĵ÷��Լ��洢newOrder����õ�ͼ�ṹ�ľ���
		newScore = 0.0;

		// �����������е�ÿһ���ڵ㣬������������ĵ÷�
		CUDA_CHECK_RETURN(
				cudaMemcpy(dev_order, newOrder, nodesNum * sizeof(int),
						cudaMemcpyHostToDevice),
				"newOrder -> dev_order failed.");

		dim3 blockNum(nodesNum);
		dim3 threadNumInBlock(256);
		size_t dynamicSharedMemory = 256 * 8;

		calTopologyScore_kernel<<<blockNum, threadNumInBlock,
				dynamicSharedMemory>>>(dev_lsTable, dev_order, dev_nodeScore,
				dev_bestParentSet, parentSetNum, nodesNum);

		CUDA_CHECK_RETURN(
				cudaMemcpy(nodeScore, dev_nodeScore, nodesNum * sizeof(double),
						cudaMemcpyDeviceToHost),
				"dev_nodeScore -> nodeScore failed.");
		CUDA_CHECK_RETURN(
				cudaMemcpy(bestParentSet, dev_bestParentSet, nodesNum * (CONSTRAINTS + 1) * sizeof(int), cudaMemcpyDeviceToHost),
				"dev_bestParentSet -> bestParentSet failed.");

		for (i = 0; i < nodesNum; i++) {
			newScore += nodeScore[i];
		}

		// ʹ��Metropolis-Hastings rule
		srand((unsigned int) time(NULL));
		double u = rand() / (double) RAND_MAX;
		if (log(u) < newScore - oldScore) {
			oldScore = newScore;
			for (j = 0; j < nodesNum; j++) {
				oldOrder[j] = newOrder[j];
			}
		}
		if (newScore > globalBestScore) {
			globalBestScore = newScore;
			for (i = 0; i < nodesNum; i++) {
				for (j = 0; j < nodesNum; j++) {
					globalBestGraph[i * nodesNum + j] = 0;
				}
			}
			for (i = 0; i < nodesNum; i++) {
				for (j = 0; j < bestParentSet[i * (CONSTRAINTS + 1)]; j++) {
					globalBestGraph[(bestParentSet[i * (CONSTRAINTS + 1) + j + 1]
							- 1) * nodesNum + i] = 1;
				}
			}
			for (i = 0; i < nodesNum; i++) {
				topSort[i] = newOrder[i];
			}
		}
	}

	CUDA_CHECK_RETURN(cudaFree(dev_lsTable), "dev_lsTable cudaFree failed.");
	CUDA_CHECK_RETURN(cudaFree(dev_order), "dev_order cudaFree failed.");
	CUDA_CHECK_RETURN(cudaFree(dev_nodeScore),
			"dev_nodeScore cudaFree failed.");
	CUDA_CHECK_RETURN(cudaFree(dev_bestParentSet),
			"dev_bestParentSet cudaFree failed.");
	free(nodeScore);
	free(newOrder);
	free(oldOrder);
	free(bestParentSet);
}

void BNSL_finish() {
	free(topSort);
	free(globalBestGraph);
}

void readNodeInfo() {
	FILE * inFile = fopen(NODEINFO_PATH, "r");

	char cur = fgetc(inFile);
	while (cur != EOF) {
		if (cur == '\n')
			nodesNum++;
		cur = fgetc(inFile);
	}
	nodesNum++;

	rewind(inFile);
	valuesRange = (int *) malloc(sizeof(int) * nodesNum);
	int i;
	for (i = 0; i < nodesNum; i++) {
		fscanf(inFile, "%d", &(valuesRange[i]));
	}

	fclose(inFile);

}

void readSamples() {
	FILE * inFile = fopen(SAMPLES_PATH, "r");
	int i, j, value;

	// ��ȡ��������
	samplesNum = 0;
	char cur = fgetc(inFile);
	while (cur != EOF) {
		if (cur == '\n')
			samplesNum++;
		cur = fgetc(inFile);
	}
	samplesNum++;

	samplesValues = (int *) malloc(sizeof(int) * samplesNum * nodesNum);
	rewind(inFile);
	for (i = 0; i < samplesNum; i++) {
		for (j = 0; j < nodesNum; j++) {
			fscanf(inFile, "%d", &value);
			samplesValues[i * nodesNum + j] = value - 1;
		}
	}

	fclose(inFile);
}

long C(int n, int m) {

	if (n > m || n < 0 || m < 0)
		return -1;

	int k, res = 1;
	for (k = 1; k <= n; k++) {
		res = (res * (m - n + k)) / k;
	}
	return res;
}

void randInitOrder(int * s) {
	for (int i = 0; i < nodesNum; i++) {
		s[i] = i + 1;
	}
	int swap, r;
	srand((unsigned int) time(NULL));
	for (int i = nodesNum - 1; i > 0; i--) {
		r = rand() % i;
		swap = s[r];
		s[r] = s[i];
		s[i] = swap;
	}
}

__global__ void calTopologyScore_kernel(double * dev_lsTable, int * dev_order,
		double * dev_nodeScore, int * dev_bestParentSet, int parentSetNum,
		int nodesNum) {

	int parentSetNumInBlock = 0;
	int i, s;
	int curPos = blockIdx.x;
	int curNode = dev_order[curPos];
	for (i = 0; i <= CONSTRAINTS && i < curPos + 1; i++) {
		parentSetNumInBlock += C_kernel(i, curPos);
	}

	extern __shared__ double result[];
	result[threadIdx.x] = -DBL_MAX;
	__syncthreads();
	int combi[CONSTRAINTS];
	int size = 0;
	if (threadIdx.x < parentSetNumInBlock) {
		findComb_kernel(curPos + 1, threadIdx.x, &size, combi);

		int parentSet[CONSTRAINTS];
		for (i = 0; i < size; i++) {
			parentSet[i] = dev_order[combi[i] - 1];
		}

		sortArray_kernel(parentSet, size);

		for (i = 0; i < size; i++) {
			if (parentSet[i] > curNode) {
				parentSet[i] -= 1;
			}
		}

		int index = 0;
		if (size > 0) {
			index = findIndex_kernel(size, parentSet, nodesNum);
		}

		result[threadIdx.x] = dev_lsTable[(curNode - 1) * parentSetNum + index];
	}

	__syncthreads();

	s = blockDim.x / 2;
	if (threadIdx.x < s) {
		if (result[threadIdx.x] >= result[threadIdx.x + s]) {
			result[threadIdx.x + s] = threadIdx.x;
		} else {
			result[threadIdx.x] = result[threadIdx.x + s];
			result[threadIdx.x + s] = threadIdx.x + s;
		}
	}

	__syncthreads();

	for (s = blockDim.x / 4; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			if (result[threadIdx.x] >= result[threadIdx.x + s]) {
				result[threadIdx.x + s] = result[threadIdx.x + 2 * s];
			} else {
				result[threadIdx.x] = result[threadIdx.x + s];
				result[threadIdx.x + s] = result[threadIdx.x + 3 * s];
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0) {
		dev_nodeScore[curPos] = result[0];
	}

	if (threadIdx.x == result[1]) {
		dev_bestParentSet[(curNode - 1) * (CONSTRAINTS + 1)] = size;
		for (i = 0; i < size; i++) {
			dev_bestParentSet[(curNode - 1) * (CONSTRAINTS + 1) + i + 1] =
					dev_order[combi[i] - 1];
		}
	}
}

__device__ double calLocalScore_kernel(int *dev_valuesRange,
		int *dev_samplesValues, int *dev_N, int samplesNum, int size,
		int* parentSet, int curNode, int nodesNum) {

	int curNodeValuesNum = dev_valuesRange[curNode];
	int valuesNum = 1;
	int i, j;
	for (i = 0; i < size; i++) {
		valuesNum = valuesNum * dev_valuesRange[parentSet[i] - 1];
	}

	int *N = dev_N
			+ ((blockIdx.x * blockDim.x + threadIdx.x) * nodesNum + curNode)
					* PARENT_VALUE_MAX_NUM;
	int pvalueIndex = 0;
	for (i = 0; i < samplesNum; i++) {
		pvalueIndex = 0;
		for (j = 0; j < size; j++) {
			pvalueIndex = pvalueIndex * dev_valuesRange[parentSet[j] - 1]
					+ dev_samplesValues[i * nodesNum + parentSet[j] - 1];
		}

		N[pvalueIndex * curNodeValuesNum
				+ dev_samplesValues[i * nodesNum + curNode]]++;
	}

	double alpha = ALPHA / (curNodeValuesNum * valuesNum);
	double localScore = size * log(GAMMA);
	for (i = 0; i < valuesNum; i++) {
		int sum = 0;
		for (j = 0; j < curNodeValuesNum; j++) {
			int cur = i * curNodeValuesNum + j;
			if (N[cur] != 0) {
				localScore = localScore + lgamma(N[cur] + alpha)
						- lgamma(alpha);
				sum = sum + N[cur];
			}
		}
		localScore = localScore + lgamma(alpha * curNodeValuesNum)
				- lgamma(alpha * curNodeValuesNum + sum);
	}

	return localScore;
}

__global__ void calAllLocalScore_kernel(int *dev_valuesRange,
		int *dev_samplesValues, int *dev_N, double *dev_lsTable, int samplesNum,
		int nodesNum, int parentSetNum) {
	/*
	 int id = blockIdx.x * blockDim.x + threadIdx.x;
	 if (id < parentSetNum) {
	 int size = 0;
	 int combination[CONSTRAINTS], parentSet[CONSTRAINTS];
	 findComb_kernel(nodesNum, id, &size, combination);
	 int i, curNode;
	 for (curNode = 0; curNode < nodesNum; curNode++) {
	 for (i = 0; i < size; i++) {
	 parentSet[i] = combination[i];
	 }
	 recoverComb_kernel(curNode, parentSet, size);
	 dev_lsTable[curNode * parentSetNum + id] = calLocalScore_kernel(
	 dev_valuesRange, dev_samplesValues, dev_N, samplesNum, size,
	 parentSet, curNode, nodesNum);
	 //dev_lsTable[id * nodesNum + curNode] = -1.0;
	 }
	 }
	 */

	if (threadIdx.x < parentSetNum) {
		int curNode = blockIdx.x;
		int index = threadIdx.x;
		int parentSet[CONSTRAINTS];
		int size = 0;

		findComb_kernel(nodesNum, index, &size, parentSet);

		recoverComb_kernel(curNode, parentSet, size);

		dev_lsTable[curNode * parentSetNum + index] = calLocalScore_kernel(
				dev_valuesRange, dev_samplesValues, dev_N, samplesNum, size,
				parentSet, curNode, nodesNum);
	}
}

__device__ long C_kernel(int n, int m) {

	if (n > m || n < 0 || m < 0)
		return -1;

	int k, res = 1;
	for (k = 1; k <= n; k++) {
		res = (res * (m - n + k)) / k;
	}
	return res;
}

__device__ void recoverComb_kernel(int curNode, int* combi, int size) {

	for (int i = 0; i < size; i++) {
		if (combi[i] >= curNode + 1) {
			combi[i] = combi[i] + 1;
		}
	}
}

__device__ void findComb_kernel(int nodesNum, int index, int* size,
		int* combi) {

	if (index == 0) {
		*size = 0;
	} else {
		int k = 1;
		int limit = C_kernel(k, nodesNum - 1);
		while (index > limit) {
			k++;
			limit = limit + C_kernel(k, nodesNum - 1);
		}
		index = index - limit + C_kernel(k, nodesNum - 1);
		*size = k;

		int base = 0;
		int n = nodesNum - 1;
		int i, sum, shift;
		int sum_new = 0;

		for (i = 1; i < k; i++) {
			sum = 0;
			for (shift = 1; shift <= n; shift++) {
				sum_new = sum + C_kernel(k - i, n - shift);
				if (sum_new < index) {
					sum = sum_new;
				} else {
					break;
				}
			}
			combi[i - 1] = base + shift;
			n = n - shift;
			index = index - sum;
			base = combi[i - 1];
		}
		combi[k - 1] = base + index;
	}
}

__device__ void sortArray_kernel(int * s, int n) {
	int min;
	int id = -1;
	for (int i = 0; i < n - 1; i++) {
		min = INT_MAX;
		id = -1;
		for (int j = i; j < n; j++) {
			if (s[j] < min) {
				min = s[j];
				id = j;
			}
		}
		int swap = s[i];
		s[i] = s[id];
		s[id] = swap;
	}
}

__device__ int findIndex_kernel(int k, int* combi, int nodesNum) {
	int index = 1;
	int i, j;
	int * newCombi = (int *) malloc(sizeof(int) * (k + 1));
	newCombi[0] = 0;
	for (i = 1; i <= k; i++) {
		newCombi[i] = combi[i - 1];
	}
	for (i = 1; i <= k; i++) {
		for (j = newCombi[i - 1] + 1; j < newCombi[i]; j++) {
			index = index + C_kernel(k - i, nodesNum - 1 - j);
		}
	}

	free(newCombi);

	for (i = 1; i < k; i++) {
		index = index + C_kernel(i, nodesNum - 1);
	}

	return index;
}
