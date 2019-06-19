
#include <cuda_runtime.h>

#include <random>
#include <stdio.h>
#include <time.h>
#include <string>
#include <vector>

using namespace std;


// 改进可以有， 把输入放到 constant前缀，  输出放入shared 部分去。 IO增速

__constant__ int dev_sum;
__constant__ int dev_bias_x;
__constant__ int dev_bias_y;
__constant__ int dev_bias_out;

template<typename T>
__global__ void multiply(T *x, T *y, T *out, int dim_m, int dim_k, int dim_n, int N) {
	// 获取全局索引
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	// 步长
	int stride = blockDim.x * gridDim.x;
	int pos, m, n, part_pos_x, part_pos_y;
	T res;
	for (int i = index; i < dev_sum; i += stride)
	{
		pos = i / N;
		m = (i - N * pos) / dim_m;
		n = i - N * pos - dim_m * m;
		part_pos_x = pos * dev_bias_x + m * dim_k;
		part_pos_y = pos * dev_bias_y + n;
		res = 0;
		for (int j = 0; j < dim_k; j++)
			res += x[part_pos_x + j] * y[part_pos_y + j * dim_n];

		out[pos * dev_bias_out + m * dim_n + n] = res;
	}
}


template<typename T>
void mat_mul(vector<T *> ins, T *out, int dim_m, int dim_k, int dim_n, int N) {
	cudaEvent_t start, stop;
	float elapsedTime;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	T *dev_x, *dev_y, *dev_out;
	cudaMalloc((void **)&dev_x, sizeof(T) * N * dim_m * dim_k);
	cudaMalloc((void **)&dev_y, sizeof(T) * N * dim_n * dim_k);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("cuda malloc : %.3f ms\n", elapsedTime);
	cudaEventRecord(start, 0);
	cudaMemcpy(dev_x, (T *)ins[0], sizeof(T) * N * dim_m * dim_k, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y, (T *)ins[1], sizeof(T) * N * dim_n * dim_k, cudaMemcpyHostToDevice);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("cuda memcpy : %.3f ms\n", elapsedTime);

	cudaMalloc((void **)&dev_out, sizeof(T) * N * dim_m * dim_n);

	// 定义kernel的执行配置
	dim3 blockSize(16384);
	dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
	cudaEventRecord(start, 0);
	// 执行kernel
	multiply<T> << <gridSize, blockSize >> > (dev_x, dev_y, dev_out, dim_m, dim_k, dim_n, N);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("cuda compte: %.3f ms\n", elapsedTime);

	cudaEventRecord(start, 0);
	cudaMemcpy(out, dev_out, sizeof(T) * N * dim_m * dim_n, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("cuda mempcpy back : %.3f ms\n", elapsedTime);

	cudaFree(dev_x);
	cudaFree(dev_y);
	cudaFree(dev_out);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}

extern "C"
void mat_mul_float(vector<float *> ins, float *out, int dim_m, int dim_k, int dim_n, int N)
{
	mat_mul<float>(ins, out, dim_m, dim_k, dim_n, N);
}


/*extern "C" int matrix_multiply_shell(std::vector<void *> inputs, void *output, int dim_m, int dim_k, int dim_n, int N, std::string T){

int sum = dim_m * dim_n * N;
int bias_x = dim_m * dim_k;
int bias_y = dim_k * dim_n;
int bias_out = dim_m * dim_n;

cudaMemcpyToSymbol(dev_sum, &sum, sizeof(int));
cudaMemcpyToSymbol(dev_bias_x, &bias_x, sizeof(int));
cudaMemcpyToSymbol(dev_bias_y, &bias_y, sizeof(int));
cudaMemcpyToSymbol(dev_bias_out, &bias_out, sizeof(int));

if(T == "short")
return matrix_multiply<short>(inputs, (short *)output, dim_m, dim_k, dim_n, N);
if(T == "int")
return matrix_multiply<int>(inputs, (int *)output, dim_m, dim_k, dim_n, N);
if(T == "long")
return matrix_multiply<long>(inputs, (long *)output, dim_m, dim_k, dim_n, N);
if(T == "long long")
return matrix_multiply<long long>(inputs, (long long *)output, dim_m, dim_k, dim_n, N);
if(T == "double")
return matrix_multiply<double>(inputs, (double *)output, dim_m, dim_k, dim_n, N);
if(T == "bool")
return matrix_multiply<bool>(inputs, (bool *)output, dim_m, dim_k, dim_n, N);

return matrix_multiply<float>(inputs, (float *)output, dim_m, dim_k, dim_n, N);
}*/


/*
template<typename T>
__global__ void add(T **x, T *z, int dim_in, int N){
// 获取全局索引
int index = threadIdx.x + blockIdx.x * blockDim.x;
// 步长
T stride = blockDim.x * gridDim.x;
for (int i = index; i < N; i += stride)
{
int res = 0;
for(int j = 0; j <= dim_in; j++)
res += x[j][i];
z[i] = res;
}
}

template<typename T>
int matrix_add(T **ins, T *out, int dim_in, int N){
T *dev_ins[dim_in], *dev_out;
int n_bytes = N *sizeof(T);
int in_bytes = dim_in *sizeof(T);

for(int i = 0; i <dim_in; i++){
T* dev_x;
cudaMalloc((void **)&dev_x, in_bytes);
dev_ins[i] = dev_x;
cudaMemcpy(dev_ins[i], ins[i], in_bytes, cudaMemcpyHostToDevice);
}
cudaMalloc((void **)&dev_out, n_bytes);

// 定义kernel的执行配置
dim3 blockSize(256);
dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

// 执行kernel
add<T><<<gridSize, blockSize>>>(dev_x, dev_y, dev_z, dim_in, N);
cudaMemcpy(out, dev_out, n_bytes, cudaMemcpyDeviceToHost);

for(int i = 0; i < dim_in; i++)
cudaFree(dev_ins[i]);
cudaFree(dev_out);
return 0;
}

int matrix_add_shell(void *inputs, void *output, int dim_in, int N, string T){
if(T == "short")
return matrix_add<short>((short **)inputs, (short *)output, dim_in, N);
if(T == "int")
return matrix_add<int>((int **)inputs, (int *)output, dim_in, N);
if(T == "long")
return matrix_add<long>((long **)inputs, (long *)output, dim_in, N);
if(T == "long long")
return matrix_add<long long>((long long **)inputs, (long long *)output, dim_in, N);
if(T == "double")
return matrix_add<double>((double **)inputs, (double *)output, dim_in, N);
if(T == "bool")
return matrix_add<bool>((bool **)inputs, (bool *)output, dim_in, N);

return matrix_add<float>((float **)inputs, (float *)output, dim_in, N);
}

template<typename T>
__global__ void dot(T **x, T *z, int dim_in, int N){
// 获取全局索引
int index = threadIdx.x + blockIdx.x * blockDim.x;
// 步长
T stride = blockDim.x * gridDim.x;
for (int i = index; i < N; i += stride)
{
int res = x[j][0];
for(int j = 1; j <= dim_in; j++)
res *= x[j][i];
z[i] = res;
}
}

template<typename T>
int matrix_dot(T **ins, T *out, int dim_in, int N){
T *dev_ins[dim_in], *dev_out;
int n_bytes = N *sizeof(T);
int in_bytes = dim_in *sizeof(T);

for(int i = 0; i <dim_in; i++){
T* dev_x;
cudaMalloc((void **)&dev_x, in_bytes);
dev_ins[i] = dev_x;
cudaMemcpy(dev_ins[i], ins[i], in_bytes, cudaMemcpyHostToDevice);
}
cudaMalloc((void **)&dev_out, n_bytes);

// 定义kernel的执行配置
dim3 blockSize(256);
dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

// 执行kernel
add<T><<<gridSize, blockSize>>>(dev_x, dev_y, dev_z, dim_in, N);
cudaMemcpy(out, dev_out, n_bytes, cudaMemcpyDeviceToHost);

for(int i = 0; i < dim_in; i++)
cudaFree(dev_ins[i]);
cudaFree(dev_out);
return 0;
}

int matrix_dot_shell(void *inputs, void *output, int dim_in, int N, string T){
if(T == "short")
return matrix_dot<short>((short **)inputs, (short *)output, dim_in, N);
if(T == "int")
return matrix_dot<int>((int **)inputs, (int *)output, dim_in, N);
if(T == "long")
return matrix_dot<long>((long **)inputs, (long *)output, dim_in, N);
if(T == "long long")
return matrix_dot<long long>((long long **)inputs, (long long *)output, dim_in, N);
if(T == "double")
return matrix_dot<double>((double **)inputs, (double *)output, dim_in, N);
if(T == "bool")
return matrix_dot<bool>((bool **)inputs, (bool *)output, dim_in, N);

return matrix_dot<float>((float **)inputs, (float *)output, dim_in, N);
}
*/

//int main() {
//	std::default_random_engine e;
//	printf("Hi there, this is the cuda add experiment program\n\n");
//	int N = 1 << 18;
//	int dim_m = 1 << 5, dim_k = 1 << 4, dim_n = 1 << 6;
//	float *x = new float[N * dim_m * dim_k];
//	float *y = new float[N * dim_n * dim_k];
//	float *z = new float[N * dim_m * dim_n];
//
//	clock_t start, finish;
//	start = clock();
//	for (int i = 0; i < N * dim_m * dim_k; i++) {
//		x[i] = ((float)e()) / 1000;
//	}
//	for (int i = 0; i < N * dim_n * dim_k; i++) {
//		y[i] = ((float)e()) / 1000;
//	}
//	finish = clock();
//	printf("init: %f ms\n", 1000 * (float)(finish - start) / CLOCKS_PER_SEC);
//	std::vector<float *>ins;
//	ins.push_back(x);
//	ins.push_back(y);
//
//	mat_mul<float>(ins, z, dim_m, dim_k, dim_n, N);
//
//	start = clock();
//	int sum = dim_m * dim_n * N;
//	int bias_x = dim_m * dim_k;
//	int bias_y = dim_k * dim_n;
//	int bias_out = dim_m * dim_n;
//	int pos, m, n, part_pos_x, part_pos_y;
//	float res;
//	for (int i = 0; i < sum; i++)
//	{
//		pos = i / N;
//		m = (i - N * pos) / dim_m;
//		n = i - N * pos - dim_m * m;
//		part_pos_x = pos * bias_x + m * dim_k;
//		part_pos_y = pos * bias_y + n;
//		res = 0;
//		for (int j = 0; j < dim_k; j++)
//			res += x[part_pos_x + j] * y[part_pos_y + j * dim_n];
//
//		z[pos * bias_out + m * dim_n + n] = res;
//	}
//	finish = clock();
//	printf("CPU compute: %f ms\n", 1000 * (float)(finish - start) / CLOCKS_PER_SEC);
//
//	delete[] x;
//	delete[] y;
//	delete[] z;
//	system("pause");
//	return 0;
//}
