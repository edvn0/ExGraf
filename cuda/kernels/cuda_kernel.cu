#define N 10

extern "C" __global__ void add(float *A, float *B, float *C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) {
		int idx = i * N + j;
		C[idx] = A[idx] + B[idx];
	}
}

extern "C" __global__ void matmul(float *A, float *B, float *C,
																	unsigned int ARows, unsigned int ACols,
																	unsigned int BRows, unsigned int BCols,
																	bool col_major) {
	if (ACols != BRows)
		return;

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < ARows && col < BCols) {
		float sum = 0.0f;

		for (int k = 0; k < ACols; ++k) {
			int A_idx = col_major ? (k * ARows + row) : (row * ACols + k);
			int B_idx = col_major ? (col * BRows + k) : (k * BCols + col);

			sum += A[A_idx] * B[B_idx];
		}

		int C_idx = col_major ? (col * ARows + row) : (row * BCols + col);
		C[C_idx] = sum;
	}
}
