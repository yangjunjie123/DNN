
//  nn_utils.hpp
//  automl
//
//  Created by Jack on 2019/2/24.
//  Copyright © 2019年 PA. All rights reserved.
//

#ifndef nn_utils_h
#define nn_utils_h

#define ACTIVATION_NONE 0
#define ACTIVATION_SIGMOID 1
#define ACTIVATION_TANH 2
#define ACTIVATION_RELU 3
#define ACTIVATION_SOFTMAX 4

#include <iostream>
#include <stdio.h>
#include <vector>
#include <time.h>
#include <assert.h>
#include <random>

using namespace std;
namespace pa_nn {

	typedef vector<double>    DoubleVector;
	typedef vector<DoubleVector>    DoubleVector2D;
	typedef vector<DoubleVector2D> DoubleVector3D;
	typedef vector<DoubleVector3D > DoubleVector4D;

	typedef struct {
		int w;
		int h;
	}MATRIX_SIZE;

	// 输入输出数据
	typedef struct{
	int width;
	int height;
	DoubleVector2D data;
	}StCnnData;


	class NN_Utils {
	public:

		static void resize_vector2D(DoubleVector2D &in_vec, int height, int width) {
			in_vec.resize(height);
			for (int h = 0; h<height; h++) {
				in_vec[h].resize(width);
			}
		}

		static void resize_vector3D(DoubleVector3D &in_vec, int channel, int height, int width) {
			in_vec.resize(channel);
			for (int c = 0; c<channel; c++) {
				in_vec[c].resize(height);
				for (int h = 0; h<height; h++) {
					in_vec[c][h].resize(width);
				}
			}
		}

		static void resize_vector4D(DoubleVector4D &in_vec, int batch_size, int channel, int width, int height) {
			in_vec.resize(batch_size);
			for (int b = 0; b<batch_size; b++) {
				in_vec[b].resize(channel);
				for (int c = 0; c<channel; c++) {
					in_vec[b][c].resize(height);
					for (int h = 0; h<height; h++) {
						in_vec[b][c][h].resize(width);
					}
				}
			}
		}

		static void vector3D_2_array(DoubleVector3D in_vec, int channel, int height, int width, DoubleVector &out_array) {
			for (int c = 0; c<channel; c++) {
				for (int h = 0; h<height; h++) {
					for (int w = 0; w<width; w++) {
						out_array.push_back(in_vec[c][h][w]);
					}
				}
			}
		}

		static DoubleVector2D matrix_transpose(DoubleVector2D vec_src) {
			int src_h = (int)vec_src.size();
			int src_w = (int)vec_src[0].size();
			DoubleVector2D vec_out;
			NN_Utils::resize_vector2D(vec_out, src_w, src_h);

			for (int i = 0; i<src_w; i++) {
				for (int j = 0; j<src_h; j++) {
					vec_out[i][j] = vec_src[j][i];
				}
			}
			return vec_out;
		}


		static void util_get_rand_array(DoubleVector &out, int len, double start, double end)
		{
			default_random_engine e;
			e.seed((unsigned)time(NULL));
			uniform_real_distribution<double> u(start, end); //随机数分布对象
			//int i;
			for (int i = 0; i < len; i++) {
				out[i] = u(e);
			}
		}

		static double util_gen_one_rand_dicimal(int decimal_len = 0)
		{
			//        int64_t N = pow(100.0, decimal_len) - 1;
			//        double out = 2 * (rand()%(N+1)- N/2)/(double)N;
			//        return out;
			default_random_engine e;
			e.seed((unsigned)time(NULL));
			uniform_real_distribution<double> u(0, 1); //随机数分布对象

			double res = u(e);

			return res;
		}

		static void gen_normal_distribution_array(int len, double x_left, double x_right, DoubleVector &out) {
			random_device rd;
			mt19937 gen(rd());
			//normal(0,1)中0为均值，1为方差

			normal_distribution<double> normal(0, 1);

			int idx = 0;

			while (1) {
				double rand_num = normal(gen);
				if ((x_left < rand_num) && (rand_num < x_right)) {
					out[idx] = rand_num;
					idx++;
					if (idx >= len) {
						break;
					}
				}
			}
		}

		static void print_debug_data_2_file(char *filename, DoubleVector2D data, int w, int h, char *tag, bool active = true);
		static double activation_op(int activation, double input);
		static double activation_op_back(int activation, double input, double output_delta);

		static int util_sigmoid(const double* src, double* dst, unsigned long length)
		{
			for (int i = 0; i < length; ++i) {
				dst[i] = (double)(1. / (1. + exp(-src[i])));
			}
			return 0;
		}


		static double util_sigmoid(const double src)
		{
			return (double)(1. / (1. + exp(-src)));
		}

		static int util_sigmoid_grad(const double* src, double* dst, unsigned long length)
		{
			for (int i = 0; i < length; ++i) {
				dst[i] = (double)(exp(-src[i]) / pow((1. + exp(-src[i])), 2));
			}
			return 0;
		}

		static double util_sigmoid_grad(const double src)
		{
			return (double)(exp(-src) / pow((1. + exp(-src)), 2));
		}

		static double util_tanh(const double src)
		{
			return 2 * (double)(1. / (1 + exp(-2 * src))) - 1;
		}

		static double util_relu(const double src)
		{
			return src> 0 ? src : 0;
		}

		static double util_mean_array(double *src, int len)
		{
			assert(len>0);
			double sum = 0;
			for (int i = 0; i<len; i++)
				sum += src[i];
			return sum / double(len);
		}

		static DoubleVector2D matrix_dot(DoubleVector2D matrix_a, DoubleVector2D matrix_b) {
			int row_a = (int)matrix_a.size();
			int col_a = (int)matrix_a[0].size();
			int row_b = (int)matrix_b.size();
			int col_b = (int)matrix_b[0].size();

			DoubleVector2D matrix_res;
			resize_vector2D(matrix_res, row_a, col_a);

			assert(row_a == row_b && col_a == col_b);

			for (int i = 0; i<row_a; i++) {
				for (int j = 0; j<col_a; j++) {
					matrix_res[i][j] = matrix_a[i][j] * matrix_b[i][j];
				}
			}
			return matrix_res;
		}

		static DoubleVector2D matrix_add(DoubleVector2D matrix_a, DoubleVector2D matrix_b) {
			int row_a = (int)matrix_a.size();
			int col_a = (int)matrix_a[0].size();
			int row_b = (int)matrix_b.size();
			int col_b = (int)matrix_b[0].size();

			DoubleVector2D matrix_res;
			resize_vector2D(matrix_res, row_a, col_a);

			assert(row_a == row_b && col_a == col_b);

			for (int i = 0; i<row_a; i++) {
				for (int j = 0; j<col_a; j++) {
					matrix_res[i][j] = matrix_a[i][j] + matrix_b[i][j];
				}
			}
			return matrix_res;
		}

		static DoubleVector2D matrix_sub(DoubleVector2D matrix_a, DoubleVector2D matrix_b) {
			int row_a = (int)matrix_a.size();
			int col_a = (int)matrix_a[0].size();
			int row_b = (int)matrix_b.size();
			int col_b = (int)matrix_b[0].size();

			DoubleVector2D matrix_res;
			resize_vector2D(matrix_res, row_a, col_a);

			assert(row_a == row_b && col_a == col_b);

			for (int i = 0; i<row_a; i++) {
				for (int j = 0; j<col_a; j++) {
					matrix_res[i][j] = matrix_a[i][j] - matrix_b[i][j];
				}
			}
			return matrix_res;
		}

		static DoubleVector2D util_matrix_multiply(DoubleVector2D arr_a, DoubleVector2D arr_b)
		{
			auto row_a = arr_a.size();
			auto col_a = arr_a[0].size();
			auto row_b = arr_b.size();
			auto col_b = arr_b[0].size();

			DoubleVector2D res;
			if (col_a != row_b) {
				return res;
			}

			res.resize(row_a);
			for (int i = 0; i < row_a; ++i) {
				res[i].resize(col_b);
			}

			for (int i = 0; i < row_a; ++i) {
				for (int j = 0; j < col_b; ++j) {
					for (int k = 0; k < col_a; ++k) {
						res[i][j] += arr_a[i][k] * arr_b[k][j];
					}
				}
			}
			return res;
		}

		//rotate a matrix 180 degrees,
		//input: original matrix,
		//output: rotated matrix,
		//height: original matrix 1st dimension length
		//width: original matrix 2nd dimension length
		void static rotate_180(DoubleVector2D &input, DoubleVector2D &output, int height, int width);

		//padding around a matrix,
		//input: original matrix,
		//output: output matrix with padding,
		//height: original matrix 1st dimension length
		//width: original matrix 2nd dimension length
		//padding_size: padding size for very edge
		void static
			padding_round_zero(DoubleVector2D &input, DoubleVector2D &output, int height, int width, int padding_size);

		//soft max part
		void static softmax(DoubleVector &input, DoubleVector &output, int n);
		void static softmax(DoubleVector2D &input, DoubleVector2D &output, int m, int n);

		//input represents forward propagation input, output represents forward propagation output
		void static softmax_back_real(DoubleVector2D &input, DoubleVector2D &real_flag, DoubleVector2D &input_delta, int m, int n);
		void static softmax_back_real(DoubleVector &output, DoubleVector &real_flag, DoubleVector &input_delta, int n);

		//ReLU part
		void static ReLU(double &input, double &output);

		void static ReLU(DoubleVector &input, DoubleVector &output, int w);

		void static ReLU(DoubleVector2D &input, DoubleVector2D &output, int h, int w);

		void static ReLU(DoubleVector3D &input, DoubleVector3D &output, int n, int h, int w);

		void static ReLU(DoubleVector4D &input, DoubleVector4D &output, int m, int n, int h, int w);

		//input represents forward propagation input, output represents forward propagation output

		void static ReLU_back(double &input, double &output_delta, double &input_deta);

		void static ReLU_back(DoubleVector &input, DoubleVector &output_delta, DoubleVector &input_delta, int w);

		void static
			ReLU_back(DoubleVector2D &input, DoubleVector2D &output_delta, DoubleVector2D &input_delta, int h, int w);

		void static
			ReLU_back(DoubleVector3D &input, DoubleVector3D &output_delta, DoubleVector3D &input_delta, int n, int h,
				int w);

		void static
			ReLU_back(DoubleVector4D &input, DoubleVector4D &output_delta, DoubleVector4D &input_delta, int m, int n, int h,
				int w);

		//sigmoid part
		void static sigmoid(double &input, double &output);

		void static sigmoid(DoubleVector &input, DoubleVector &output, int w);

		void static sigmoid(DoubleVector2D &input, DoubleVector2D &output, int h, int w);

		void static sigmoid(DoubleVector3D &input, DoubleVector3D &output, int n, int h, int w);

		void static sigmoid(DoubleVector4D &input, DoubleVector4D &output, int m, int n, int h, int w);

		//input represents forward propagation input, output represents forward propagation output
		void static sigmoid_back(double &input, double &output_delta, double &input_delta);

		void static sigmoid_back(DoubleVector &input, DoubleVector &output_delta, DoubleVector &input_delta, int w);

		void static
			sigmoid_back(DoubleVector2D &input, DoubleVector2D &output_delta, DoubleVector2D &input_delta, int h, int w);

		void static
			sigmoid_back(DoubleVector3D &input, DoubleVector3D &output_delta, DoubleVector3D &input_delta, int n, int h,
				int w);

		void static
			sigmoid_back(DoubleVector4D &input, DoubleVector4D &output_delta, DoubleVector4D &input_delta, int m, int n,
				int h,
				int w);

		void static d_sigmoid(double in, double &out);
		void static d_sigmoid(DoubleVector in, DoubleVector &out, int w);
		void static d_sigmoid(DoubleVector2D in, DoubleVector2D &out, int h, int w);

		//tanh part
		void static tanh(double &input, double &output);

		void static tanh(DoubleVector &input, DoubleVector &output, int w);

		void static tanh(DoubleVector2D &input, DoubleVector2D &output, int h, int w);

		void static tanh(DoubleVector3D &input, DoubleVector3D &output, int n, int h, int w);

		void static tanh(DoubleVector4D &input, DoubleVector4D &output, int m, int n, int h, int w);

		//input represents forward propagation input, output represents forward propagation output
		void static tanh_back(double &input, double &output_delta, double &input_delta);

		void static tanh_back(DoubleVector &input, DoubleVector &output_delta, DoubleVector &input_delta, int w);

		void static
			tanh_back(DoubleVector2D &input, DoubleVector2D &output_delta, DoubleVector2D &input_delta, int h, int w);

		void static
			tanh_back(DoubleVector3D &input, DoubleVector3D &output_delta, DoubleVector3D &input_delta, int n, int h,
				int w);

		void static
			tanh_back(DoubleVector4D &input, DoubleVector4D &output_delta, DoubleVector4D &input_delta, int m, int n,
				int h,
				int w);

		void static d_tanh(double in, double &out);

		void static d_tanh(DoubleVector in, DoubleVector &out, int w);

		void static d_tanh(DoubleVector2D in, DoubleVector2D &out, int h, int w);

		// one hot part
		void static one_hot(double input, DoubleVector &output, int n);

		void static one_hot(DoubleVector &input, DoubleVector2D &output, int m, int n);

		void static array_to_4_dimension(DoubleVector array, DoubleVector4D &output, vector<int>dimension);
		void static array_to_3_dimension(DoubleVector array, DoubleVector3D &output, int ch, int w, int h);
		void static array_to_2_dimension(DoubleVector array, DoubleVector2D &output, int w, int h);


		// gradient check intuition, get input, delta input, output, delta output and the layer compute function
		// step 1: new input = input + delta input, new output = output + delta output
		// step 2; new real output = compute(new input)
		// step 3: check the values (new real output) and (new output) is close or not, if they are close, it means
		// that the back propagation computation is correct
		//gradient check: output = input1 + input2
		void static a_add_b_matrix(double input, double input_delta, double &output);

		void static a_add_b_matrix(DoubleVector input, DoubleVector input_delta, DoubleVector &output, int w);

		void static
			a_add_b_matrix(DoubleVector2D input, DoubleVector2D input_delta, DoubleVector2D &output, int h, int w);

		void static
			a_add_b_matrix(DoubleVector3D input, DoubleVector3D input_delta, DoubleVector3D &output, int n, int h, int w);

		void static
			a_add_b_matrix(DoubleVector4D input, DoubleVector4D input_delta, DoubleVector4D &output, int m, int n, int h,
				int w);

		//gradient check: new output close to (output = layer.compute(input + delta input)
		bool static gradient_check(double output_real, double output_new);

		void static gradient_check(DoubleVector output_real, DoubleVector output_new, int w);

		void static gradient_check(DoubleVector2D output_real, DoubleVector2D output_new, int h, int w);

		void static gradient_check(DoubleVector3D output_real, DoubleVector3D output_new, int n, int h, int w);

		void static gradient_check(DoubleVector4D output_real, DoubleVector4D output_new, int m, int n, int h, int w);

	};

}
#endif /* nn_utils_h */
