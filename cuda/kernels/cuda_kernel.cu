#define N 10
#define uint unsigned int

extern "C" __global__ void add(float *A, float *B, float *C) {
	uint i = blockIdx.x * blockDim.x + threadIdx.x;
	uint j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < N && j < N) {
		uint idx = i * N + j;
		C[idx] = A[idx] + B[idx];
	}
}

extern "C" __global__ void matmul(float *A, float *B, float *C,
																	unsigned int ARows, unsigned int ACols,
																	unsigned int BRows, unsigned int BCols,
																	bool col_major) {
	if (ACols != BRows)
		return;

	uint row = blockIdx.y * blockDim.y + threadIdx.y;
	uint col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < ARows && col < BCols) {
		float sum = 0.0f;

		for (uint k = 0; k < ACols; ++k) {
			uint A_idx = col_major ? (k * ARows + row) : (row * ACols + k);
			uint B_idx = col_major ? (col * BRows + k) : (k * BCols + col);

			sum += A[A_idx] * B[B_idx];
		}

		uint C_idx = col_major ? (col * ARows + row) : (row * BCols + col);
		C[C_idx] = sum;
	}
}
