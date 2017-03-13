#include "BNSL_GPU.cuh"

int * valuesRange;

int nodesNum = 0;

int * samplesValues;

int samplesNum;

int allParentSetNumPerNode;

double * dev_lsTable;

int* globalBestGraph;
int* topSort;
double globalBestScore;

int begin = 0;

int main() {
	calcCPUTimeStart("init.");
	BNSL_init();
	calcCPUTimeEnd();

	calcCPUTimeStart("calcLS.");
	BNSL_calcLocalScore();
	calcCPUTimeEnd();

	calcCPUTimeStart("start.");
	BNSL_start();
	calcCPUTimeEnd();

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

	calcCPUTimeStart("finish.");
	BNSL_finish();
	calcCPUTimeEnd();

	return 0;
}

void CheckCudaError(cudaError_t err, char const* errMsg) {
	if (err == cudaSuccess)
		return;
	printf("%s\nError Message: %s.\n", errMsg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}

void calcCPUTimeStart(char const *message) {
	begin = clock();
	printf("%s", message);
}

void calcCPUTimeEnd() {
	printf("Elapsed CPU time is %dms\n", (clock() - begin) / 1000);
}

void BNSL_init() {
	readNodeInfo();
	readSamples();
}

void BNSL_calcLocalScore() {

	int i;
	allParentSetNumPerNode = 0;
	for (i = 0; i <= CONSTRAINTS; i++) {
		allParentSetNumPerNode = allParentSetNumPerNode + C(i, nodesNum - 1);
	}

	int* dev_valuesRange;
	int* dev_samplesValues;
	int* dev_N;

	// calculate max different values number for all pair of child and parent set
	int valuesMaxNum = calcValuesMaxNum();

	// malloc in GPU global mem.
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_lsTable,
					nodesNum * allParentSetNumPerNode * sizeof(double)),
			"dev_lsTable cudaMalloc failed.");
	CUDA_CHECK_RETURN(cudaMalloc(&dev_valuesRange, nodesNum * sizeof(int)),
			"dev_valuesRange cudaMalloc failed.");
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_samplesValues, samplesNum * nodesNum * sizeof(int)),
			"dev_samplesValues cudaMalloc failed.");
	CUDA_CHECK_RETURN(
			cudaMalloc(&dev_N,
					allParentSetNumPerNode * valuesMaxNum * sizeof(int)),
			"dev_N cudaMalloc failed.");

	// copy data from CPU mem to GPU mem.
	CUDA_CHECK_RETURN(
			cudaMemcpy(dev_valuesRange, valuesRange, nodesNum * sizeof(int),
					cudaMemcpyHostToDevice),
			"valuesRange -> dev_valuesRange failed.");
	CUDA_CHECK_RETURN(
			cudaMemcpy(dev_samplesValues, samplesValues,
					samplesNum * nodesNum * sizeof(int),
					cudaMemcpyHostToDevice),
			"samplesValues -> dev_samplesValues failed.");
	CUDA_CHECK_RETURN(
			cudaMemset(dev_N, 0,
					allParentSetNumPerNode * valuesMaxNum * sizeof(int)),
			"dev_N cudaMemset failed.");

	int blockNum = (allParentSetNumPerNode + 1) / 256 + 1;
	calAllLocalScore_kernel<<<blockNum, 256>>>(dev_valuesRange,
			dev_samplesValues, dev_N, dev_lsTable, samplesNum, nodesNum,
			allParentSetNumPerNode, valuesMaxNum);
	CUDA_CHECK_RETURN(cudaDeviceSynchronize(),
			"calAllLocalScore_kernel failed on running.");

	CUDA_CHECK_RETURN(cudaFree(dev_valuesRange),
			"dev_valuesRange cudaFree failed.");
	CUDA_CHECK_RETURN(cudaFree(dev_samplesValues),
			"dev_samplesValues cudaFree failed.");
	CUDA_CHECK_RETURN(cudaFree(dev_N), "dev_N cudaFree failed.");

	free(valuesRange);
	free(samplesValues);
}

void BNSL_start() {
	int i, j, iter, offset;
	double oldScore = -DBL_MAX, newScore = 0.0;
	globalBestScore = -DBL_MAX;
	globalBestGraph = (int *) malloc(sizeof(int) * nodesNum * nodesNum);
	topSort = (int *) malloc(sizeof(int) * nodesNum);

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

	int * newOrder = (int *) malloc(sizeof(int) * nodesNum);
	randInitOrder(newOrder);
	int * oldOrder = (int *) malloc(sizeof(int) * nodesNum);

	int maxIterNum = 1;
	for (iter = 0; iter < maxIterNum; iter++) {

		//randSwapTwoNode(newOrder);
		newOrder[0] = 1;
		newOrder[1] = 2;
		newOrder[2] = 3;
		newOrder[3] = 4;
		newOrder[4] = 5;

		CUDA_CHECK_RETURN(
				cudaMemcpy(dev_order, newOrder, nodesNum * sizeof(int),
						cudaMemcpyHostToDevice),
				"newOrder -> dev_order failed.");

		// use GPU to calculate order score
		calOrderScore_kernel<<<nodesNum, 256, 256 * 8>>>(dev_lsTable, dev_order,
				dev_nodeScore, dev_bestParentSet, allParentSetNumPerNode,
				nodesNum);

		CUDA_CHECK_RETURN(
				cudaMemcpy(nodeScore, dev_nodeScore, nodesNum * sizeof(double),
						cudaMemcpyDeviceToHost),
				"dev_nodeScore -> nodeScore failed.");
		CUDA_CHECK_RETURN(
				cudaMemcpy(bestParentSet, dev_bestParentSet, nodesNum * (CONSTRAINTS + 1) * sizeof(int), cudaMemcpyDeviceToHost),
				"dev_bestParentSet -> bestParentSet failed.");

		newScore = 0.0;
		for (i = 0; i < nodesNum; i++) {
			newScore += nodeScore[i];
		}

		// use Metropolis-Hastings rule
		srand((unsigned int) time(NULL));
		double u = rand() / (double) RAND_MAX;
		if (log(u) < newScore - oldScore) {
			oldScore = newScore;
			memcpy(oldOrder, newOrder, nodesNum * sizeof(int));
		}

		// search for best graph
		if (newScore > globalBestScore) {
			globalBestScore = newScore;
			memset(globalBestGraph, 0, nodesNum * nodesNum * sizeof(int));

			for (i = 0; i < nodesNum; i++) {
				for (j = 1, offset = i * (CONSTRAINTS + 1);
						j <= bestParentSet[offset]; j++) {
					globalBestGraph[(bestParentSet[offset + j] - 1) * nodesNum
							+ i] = 1;
				}
			}

			memcpy(topSort, newOrder, nodesNum * sizeof(int));
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

int compare(const void*a, const void*b) {
	return *(int*) a - *(int*) b;
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

void selectTwoNodeToSwap(int *n1, int *n2) {
	*n1 = rand() % nodesNum;
	*n2 = rand() % nodesNum;
	if (*n1 == *n2) {
		*n2 = rand() % (nodesNum - 1);
		if (*n2 >= *n1) {
			*n2++;
		}
	}
}

void randSwapTwoNode(int *order) {
	int n1 = 0, n2 = 0, temp;
	selectTwoNodeToSwap(&n1, &n2);
	temp = order[n1];
	order[n1] = order[n2];
	order[n2] = temp;
}

int calcValuesMaxNum() {
	int * valuesRangeToSort = (int *) malloc(nodesNum * sizeof(int));
	memcpy(valuesRangeToSort, valuesRange, nodesNum * sizeof(int));
	qsort(valuesRangeToSort, nodesNum, sizeof(int), compare);
	int valuesMaxNum = 1;
	for (int i = nodesNum - CONSTRAINTS - 1; i < nodesNum; i++) {
		valuesMaxNum *= valuesRangeToSort[i];
	}
	free(valuesRangeToSort);
	return valuesMaxNum;
}

__global__ void calOrderScore_kernel(double * dev_lsTable, int * dev_order,
		double * dev_nodeScore, int * dev_bestParentSet, int allParentSetNumPerNode,
		int nodesNum) {

	int parentSetNumInOrder = 0;
	int i, s;
	int curPos = blockIdx.x;
	int curNode = dev_order[curPos];
	for (i = 0; i <= CONSTRAINTS && i < curPos + 1; i++) {
		parentSetNumInOrder += C_kernel(i, curPos);
	}

	extern __shared__ double result[];
	result[threadIdx.x] = -DBL_MAX;
	__syncthreads();
	int combi[CONSTRAINTS];
	int size = 0;
	if (threadIdx.x < parentSetNumInOrder) {
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

		result[threadIdx.x] = dev_lsTable[(curNode - 1) * allParentSetNumPerNode + index];
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
		int* parentSet, int curNode, int nodesNum, int valuesMaxNum) {

	int curNodeValuesNum = dev_valuesRange[curNode];
	int valuesNum = 1;
	int i, j;
	for (i = 0; i < size; i++) {
		valuesNum = valuesNum * dev_valuesRange[parentSet[i] - 1];
	}

	int *N = dev_N + (blockIdx.x * blockDim.x + threadIdx.x) * valuesMaxNum;
	for (i = 0; i < valuesMaxNum; i++) {
		N[i] = 0;
	}
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
		int nodesNum, int allParentSetNumPerNode, int valuesMaxNum) {

	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < allParentSetNumPerNode) {
		int size = 0;
		int combination[CONSTRAINTS], parentSet[CONSTRAINTS];
		findComb_kernel(nodesNum, id, &size, combination);
		int i, curNode;
		for (curNode = 0; curNode < nodesNum; curNode++) {
			for (i = 0; i < size; i++) {
				parentSet[i] = combination[i];
			}
			recoverComb_kernel(curNode, parentSet, size);
			double result = calLocalScore_kernel(dev_valuesRange,
					dev_samplesValues, dev_N, samplesNum, size, parentSet,
					curNode, nodesNum, valuesMaxNum);
			dev_lsTable[curNode * allParentSetNumPerNode + id] = result;
		}
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
		min = s[i];
		id = i;
		for (int j = i + 1; j < n; j++) {
			if (s[j] < min) {
				min = s[j];
				id = j;
			}
		}
		if (i != id) {
			int swap = s[i];
			s[i] = s[id];
			s[id] = swap;
		}
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
