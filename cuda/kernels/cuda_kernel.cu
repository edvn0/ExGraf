__global__ void add(float *A, float *B, float *C, int N) {
	int i = threadIdx.x;
	int j = threadIdx.y;
	int idx = i * N + j; // Compute flattened index
	C[idx] = A[idx] + B[idx];
}
