
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <random>

#define CHECK(call)														\
{																		\
	const cudaError_t error	= call;										 \
	if (error != cudaSuccess)											 \
	{																	 \
		printf("Error: %s:%d, ", __FILE__, __LINE__);						\
		printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
		exit(1);															\
	}																		\
}

double cpuSecond()
{
	clock_t start = clock();
	double res = (double)start / CLOCKS_PER_SEC;
	return res;
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
	double epsilon = 1.0E-8;
	bool match = 1;
	for (int i = 0; i < N; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip, int size)
{
	time_t t;
	srand((unsigned)time(&t));
	for (int i = 0; i < size; i++)
		ip[i] = (float)(rand() & 0xFF) / 10.0f;
}

void initialInt(int *ip, int size) {
	for (int i = 0; i < size; i++) {
		ip[i] = i;
	}
}
void printMatrix(int *C, const int nx, const int ny) {
	int *ic = C;
	printf("\nMatrix: (%d.%d)\n", nx, ny);
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			printf("%3d", ic[ix]);
		}
		ic += nx;
		printf("\n");
	}
	printf("\n");
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny) {
	float *ia = A;
	float *ib = B;
	float *ic = C;
	for (int iy = 0; iy < ny; iy++) {
		for (int ix = 0; ix < nx; ix++) {
			ic[ix] = ia[ix] + ib[ix];
		}
		ia += nx; ib += nx; ic += nx;
	}
}

template<typename T>
void multiplyOnHost(T *x, T *y, T *out, int dim_m, int dim_k, int dim_n, int N)
{
	for (int i = 0; i < dim_m; i++) {
		for (int j = 0; j < dim_n; j++) {
			out[i * dim_n + j] = 0;
			for (int k = 0; k < dim_k; k++) {

				out[i * dim_n + j] += x[i * dim_k + k] * y[k * dim_n + j];
			}
		}
	}

	N--;
	while (N--) {
		x = x + dim_m * dim_k;
		y = y + dim_k * dim_n;
		out = out + dim_m * dim_n;
		for (int i = 0; i < dim_m; i++) {
			for (int j = 0; j < dim_n; j++) {
				out[i * dim_n + j] = 0;
				for (int k = 0; k < dim_k; k++) {

					out[i * dim_n + j] += x[i * dim_k + k] * y[k * dim_n + j];
				}
			}
		}//for
		
	}//while
}

template<typename T>
__global__ void sumMatrixOnGPU2D(T *MatA, T *MatB, T *MatC, int nx, int ny) {
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = iy*nx + ix;
	if (ix < nx && iy < ny)
		MatC[idx] = MatA[idx] + MatB[idx];
}

// 获取矩阵A的(row, col)元素
template<typename T>
__device__ T getElement(T *A, int heigth, int width, int row, int col)
{
	return A[row * width + col];
}

// 为矩阵A的(row, col)元素赋值
template<typename T>
__device__ void setElement(T *A, int height, int width, int row, int col, T value)
{
	A[row * width + col] = value;
}

// 矩阵相乘kernel，2-D，每个线程计算一个元素 (m*k) * (k*n) = (m*n)
template<typename T>
__global__ void matMulKernel(T *A, T *B, T *C, int dim_m, int dim_k, int dim_n, int N)
{
	T Cvalue = 0.0;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	for (int i = 0; i < dim_k; ++i)
	{
		Cvalue += getElement(A, dim_m, dim_k, row, i) * getElement(B, dim_k, dim_n, i, col);
	}
	setElement(C, dim_m, dim_n, row, col, Cvalue);
	//batch
	N--;
	while (N--)
	{
		A = A + dim_m * dim_k;
		B = B + dim_k * dim_n;
		C = C + dim_m * dim_n;
		T Cvalue = 0.0;
		for (int i = 0; i < dim_k; ++i)
		{
			Cvalue += getElement(A, dim_m, dim_k, row, i) * getElement(B, dim_k, dim_n, i, col);
		}
		setElement(C, dim_m, dim_n, row, col, Cvalue);
	}

}

template<typename T>
void matrix_multiply(T *x, T *y, T *z, int dim_m, int dim_k, int dim_n, int N)
{
	//kernel
	T *d_MatA, *d_MatB, *d_MatC;
	cudaMalloc((void **)&d_MatA, sizeof(T) * N * dim_m * dim_k);
	cudaMalloc((void **)&d_MatB, sizeof(T) * N * dim_n * dim_k);
	cudaMalloc((void **)&d_MatC, sizeof(T) * N * dim_m * dim_n);
	// transfer data from host to device
	cudaMemcpy(d_MatA, x, sizeof(T) * N  * dim_m * dim_k, cudaMemcpyHostToDevice);
	cudaMemcpy(d_MatB, y, sizeof(T) * N  * dim_n * dim_k, cudaMemcpyHostToDevice);

	dim3 blockSize(32, 32);
	dim3 gridSize((dim_n + blockSize.x - 1) / blockSize.x,
		(dim_m + blockSize.y - 1) / blockSize.y);

	double iStart = cpuSecond();
	matMulKernel<T> << < gridSize, blockSize >> >(d_MatA, d_MatB, d_MatC, dim_m, dim_k, dim_n, N);
	cudaDeviceSynchronize();
	double iElaps = cpuSecond() - iStart;
	printf("matMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", gridSize.x,
		gridSize.y, blockSize.x, blockSize.y, iElaps * 1000);
	// copy kernel result back to host side
	cudaMemcpy(z, d_MatC, sizeof(T)* N * dim_m * dim_n, cudaMemcpyDeviceToHost);

	cudaFree(d_MatA);
	cudaFree(d_MatB);
	cudaFree(d_MatC);

	cudaDeviceReset();	
}

extern "C" 
void matrix_multiply_double(double *x, double *y, double *z, int dim_m, int dim_k, int dim_n, int N)
{
	matrix_multiply<double>(x, y, z, dim_m, dim_k, dim_n, N);
}

//int main() {
//	std::default_random_engine e;
//	printf("Hi there, this is the cuda add experiment program\n\n");
//	int N = 1;
//	int dim_m = 1 << 9, dim_k = 1 << 10, dim_n = 1 << 8;
//	float *x = new float[N * dim_m * dim_k];
//	float *y = new float[N * dim_n * dim_k];
//	float *z = new float[N * dim_m * dim_n];
//	float *z_cpu = new float[N * dim_m * dim_n];
//
//	clock_t start, finish;
//	start = clock();
//	for (int i = 0; i < N * dim_m * dim_k; i++) {
//		x[i] = ((float)e()) / 1000000;
//	}
//	for (int i = 0; i < N * dim_n * dim_k; i++) {
//		y[i] = ((float)e()) / 1000000;
//	}
//	finish = clock();
//	printf("init: %f ms\n", 1000 * (float)(finish - start) / CLOCKS_PER_SEC);
//
//	//kernel
//	float *d_MatA, *d_MatB, *d_MatC;
//	cudaMalloc((void **)&d_MatA, sizeof(float) * N * dim_m * dim_k);
//	cudaMalloc((void **)&d_MatB, sizeof(float) * N * dim_n * dim_k);
//	cudaMalloc((void **)&d_MatC, sizeof(float) * N * dim_m * dim_n);
//	// transfer data from host to device
//	cudaMemcpy(d_MatA, x, sizeof(float) * N * dim_m * dim_k, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_MatB, y, sizeof(float) * N * dim_n * dim_k, cudaMemcpyHostToDevice);
//
//	dim3 blockSize(32, 32);
//	dim3 gridSize((dim_n + blockSize.x - 1) / blockSize.x,
//		(dim_m + blockSize.y - 1) / blockSize.y);
//
//	double iStart = cpuSecond();
//	//multiplyOnGPU<float> << < gridSize, blockSize >> >(d_MatA, d_MatB, d_MatC, dim_m, dim_k, dim_n, N);
//	matMulKernel<float> << < gridSize, blockSize >> >(d_MatA, d_MatB, d_MatC, dim_m, dim_k, dim_n);
//	cudaDeviceSynchronize();
//	double iElaps = cpuSecond() - iStart;
//	printf("matMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", gridSize.x,
//		gridSize.y, blockSize.x, blockSize.y, iElaps * 1000);
//	// copy kernel result back to host side
//	cudaMemcpy(z, d_MatC, sizeof(float) * N * dim_m * dim_n, cudaMemcpyDeviceToHost);
//
//	// cpu compute
//	start = clock();
//	multiplyOnHost<float>(x, y, z_cpu, dim_m, dim_k, dim_n);
//	finish = clock();
//	printf("CPU compute: %f ms\n", 1000 * (float)(finish - start) / CLOCKS_PER_SEC);
//
//	// check device results
//	checkResult(z, z_cpu, N * dim_n * dim_m);
//
//	cudaFree(d_MatA);
//	cudaFree(d_MatB);
//	cudaFree(d_MatC);
//
//	delete[] x;
//	delete[] y;
//	delete[] z;
//	delete[] z_cpu;
//	cudaDeviceReset();
//	system("pause");
//	return 0;
//}

int main() {
	std::default_random_engine e;
	printf("Hi there, this is the cuda add experiment program\n\n");
	int N = 2;
	int dim_m = 1 << 8, dim_k = 1 << 9, dim_n = 1 << 10;
	float *x = new float[N * dim_m * dim_k];
	float *y = new float[N * dim_n * dim_k];
	float *z = new float[N * dim_m * dim_n];
	float *z_cpu = new float[N * dim_m * dim_n];

	clock_t start, finish;
	start = clock();
	for (int i = 0; i < N * dim_m * dim_k; i++) {
		x[i] = ((float)e()) / 1000000;
	}
	for (int i = 0; i < N * dim_n * dim_k; i++) {
		y[i] = ((float)e()) / 1000000;
	}
	finish = clock();
	printf("init: %f ms\n", 1000 * (float)(finish - start) / CLOCKS_PER_SEC);

	//kernel
	matrix_multiply<float>(x, y, z, dim_m, dim_k, dim_n, N);

	// cpu compute
	start = clock();
	multiplyOnHost<float>(x, y, z_cpu, dim_m, dim_k, dim_n, N);
	finish = clock();
	printf("CPU compute: %f ms\n", 1000 * (float)(finish - start) / CLOCKS_PER_SEC);

	// check device results
	checkResult(z_cpu, z, N * dim_n * dim_m);

	delete[] x;
	delete[] y;
	delete[] z;
	delete[] z_cpu;
	system("pause");
	return 0;
}

