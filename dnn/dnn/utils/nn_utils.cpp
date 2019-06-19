//
//  nn_utils.cpp
//  automl
//
//  Created by Jack on 2019/3/6.
//  Copyright © 2019年 PA. All rights reserved.
//

#include "nn_utils.hpp"
#include <stdio.h>
using namespace pa_nn;
using namespace std;

double NN_Utils::activation_op(int activation, double input) {
	double ret = 0.0;
	switch (activation) {
	case ACTIVATION_SIGMOID:
		sigmoid(input, ret);
		break;
	case ACTIVATION_RELU:
		ReLU(input, ret);
		break;
	case ACTIVATION_TANH:
		tanh(input, ret);
		break;

	default:
		break;
	}
	return ret;
}

double NN_Utils::activation_op_back(int activation, double input, double output_delta) {
	double ret = 0.0;
	switch (activation) {
	case ACTIVATION_SIGMOID:
		sigmoid_back(input, output_delta, ret);
		break;
	case ACTIVATION_RELU:
		ReLU_back(input, output_delta, ret);
		break;
	case ACTIVATION_TANH:
		tanh_back(input, output_delta, ret);
		break;
	default:
		break;
	}
	return ret;
}


void NN_Utils::rotate_180(DoubleVector2D &input, DoubleVector2D &output, int height, int width) {
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			output[j][i] = input[width - 1 - j][height - 1 - i];
	}
}

void NN_Utils::padding_round_zero(DoubleVector2D &input, DoubleVector2D &output, int height, int width,
	int padding_size) {
	for (int i = 0; i < height + 2 * padding_size; i++) {
		int posx = i - padding_size;
		for (int j = 0; j < width + 2 * padding_size; j++) {
			int posy = j - padding_size;
			if (posx < 0 || posy < 0 || posx >= height || posy >= width) {
				output[i][j] = 0;
			}
			else
				output[i][j] = input[posx][posy];
		}
	}
}
/*
void NN_Utils::print_debug_data_2_file(char *filename, DoubleVector2D data, int w, int h, char *tag, bool active) {

	if (!active) {
		return;
	}
	FILE *fi = fopen(filename, "a+");
	assert(fi);
	char title[32];
	sprintf_s(title, "\n\n %s\n", tag); //这里改了sprintf
	std::fputs(title, fi);
	for (int i = 0; i<h; i++) {
		fputs("\n", fi);
		for (int j = 0; j<w; j++) {
			fprintf(fi, "%.3f ", data[i][j]);
		}
	}
	fclose(fi);
}
*/

void NN_Utils::softmax(DoubleVector &input, DoubleVector &output, int n) {
	double sum = 0;

	sum = 0;
	for (int j = 0; j < n; j++) {
		sum += exp(input[j]);
	}
	for (int j = 0; j < n; j++) {
		output[j] = exp(input[j]) / sum;
	}

}

void NN_Utils::softmax(DoubleVector2D &input, DoubleVector2D &output, int m, int n) {
	double sum = 0;
	for (int i = 0; i < m; i++) {
		sum = 0;
		for (int j = 0; j < n; j++) {
			sum += exp(input[i][j]);
		}
		for (int j = 0; j < n; j++) {
			output[i][j] = exp(input[i][j]) / sum;
		}
	}
}

void NN_Utils::softmax_back_real(DoubleVector &output, DoubleVector &real_flag,
	pa_nn::DoubleVector &input_delta, int n) {
	for (int j = 0; j < n; j++) {
		input_delta[j] = output[j] - real_flag[j];
	}

}

void NN_Utils::softmax_back_real(DoubleVector2D &output, DoubleVector2D &real_flag,
	pa_nn::DoubleVector2D &input_delta, int m, int n) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			input_delta[i][j] = output[i][j] - real_flag[i][j];
		}
	}
}

void NN_Utils::ReLU(double &input, double &output) {
	output = input > 0 ? input : 0;
}

void NN_Utils::ReLU(DoubleVector &input, DoubleVector &output, int w) {
	for (int i = 0; i < w; i++) {
		ReLU(input[i], output[i]);
	}
}

void NN_Utils::ReLU(DoubleVector2D &input, DoubleVector2D &output, int h, int w) {
	for (int i = 0; i < h; i++) {
		ReLU(input[i], output[i], w);
	}
}

void NN_Utils::ReLU(DoubleVector3D &input, DoubleVector3D &output, int n, int h, int w) {
	for (int i = 0; i < n; i++) {
		ReLU(input[i], output[i], h, w);
	}
}

void NN_Utils::ReLU(DoubleVector4D &input, DoubleVector4D &output, int m, int n, int h, int w) {
	for (int i = 0; i < m; i++) {
		ReLU(input[i], output[i], n, h, w);
	}
}

void NN_Utils::ReLU_back(double &input, double &output_delta, double &input_delta) {
	input_delta = input >= 0 ? output_delta : 0;
}

void NN_Utils::ReLU_back(DoubleVector &input, DoubleVector &output_delta, DoubleVector &input_delta, int w) {
	for (int i = 0; i < w; i++) {
		input_delta[i] = input[i] >= 0 ? output_delta[0] : 0;
	}
};

void NN_Utils::ReLU_back(DoubleVector2D &input, DoubleVector2D &output_delta, DoubleVector2D &input_delta, int h, int w) {
	for (int i = 0; i < h; i++)
		ReLU_back(input[i], output_delta[i], input_delta[i], w);
}

void NN_Utils::ReLU_back(DoubleVector3D &input, DoubleVector3D &output_delta, DoubleVector3D &input_delta, int n, int h,
	int w) {
	for (int i = 0; i < n; i++)
		ReLU_back(input[i], output_delta[i], input_delta[i], h, w);
}

void NN_Utils::ReLU_back(DoubleVector4D &input, DoubleVector4D &output_delta, DoubleVector4D &input_delta, int m, int n,
	int h,
	int w) {
	for (int i = 0; i < m; i++)
		ReLU_back(input[i], output_delta[i], input_delta[i], n, h, w);
}

void NN_Utils::sigmoid(double &input, double &output) {
	output = 1.0 / (1.0 + exp(-input));
}

void NN_Utils::sigmoid(DoubleVector &input, DoubleVector &output, int w) {
	for (int i = 0; i < w; i++) {
		sigmoid(input[i], output[i]);
	}
}

void NN_Utils::sigmoid(DoubleVector2D &input, DoubleVector2D &output, int h, int w) {
	for (int i = 0; i < h; i++) {
		sigmoid(input[i], output[i], w);
	}
}

void NN_Utils::sigmoid(DoubleVector3D &input, DoubleVector3D &output, int n, int h, int w) {
	for (int i = 0; i < n; i++) {
		sigmoid(input[i], output[i], h, w);
	}
}

void NN_Utils::sigmoid(DoubleVector4D &input, DoubleVector4D &output, int m, int n, int h, int w) {
	for (int i = 0; i < m; i++) {
		sigmoid(input[i], output[i], n, h, w);
	}
}

void NN_Utils::sigmoid_back(double &input, double &output_delta, double &input_delta) {
	//input_delta = input * (1 - input) * output_delta;
	input_delta = (double)(exp(-input) / pow((1. + exp(-input)), 2)) * output_delta;
}

void NN_Utils::sigmoid_back(DoubleVector &input, DoubleVector &output_delta, DoubleVector &input_delta, int w) {
	for (int i = 0; i < w; i++) {
		sigmoid_back(input[i], output_delta[i], input_delta[i]);
	}
};

void NN_Utils::sigmoid_back(DoubleVector2D &input, DoubleVector2D &output_delta, DoubleVector2D &input_delta, int h, int w) {
	for (int i = 0; i < h; i++)
		sigmoid_back(input[i], output_delta[i], input_delta[i], w);
}

void NN_Utils::sigmoid_back(DoubleVector3D &input, DoubleVector3D &output_delta, DoubleVector3D &input_delta, int n, int h,
	int w) {
	for (int i = 0; i < n; i++)
		sigmoid_back(input[i], output_delta[i], input_delta[i], h, w);
}

void NN_Utils::sigmoid_back(DoubleVector4D &input, DoubleVector4D &output_delta, DoubleVector4D &input_delta, int m, int n,
	int h,
	int w) {
	for (int i = 0; i < m; i++)
		sigmoid_back(input[i], output_delta[i], input_delta[i], n, h, w);
}

void NN_Utils::d_sigmoid(double in, double &out) {
	out = in * (1 - in);
}

void NN_Utils::d_sigmoid(DoubleVector in, DoubleVector &out, int w) {
	for (int i = 0; i < w; i++) {
		d_sigmoid(in[i], out[i]);
	}
}

void NN_Utils::d_sigmoid(DoubleVector2D in, DoubleVector2D &out, int h, int w) {
	for (int i = 0; i < h; i++) {
		d_sigmoid(in[i], out[i], w);
	}
}

void NN_Utils::tanh(double &input, double &output) {
	output = std::tanh(input);//sinh(input) / cosh(input);
}

void NN_Utils::tanh(DoubleVector &input, DoubleVector &output, int w) {
	for (int i = 0; i < w; i++) {
		tanh(input[i], output[i]);
	}
}

void NN_Utils::tanh(DoubleVector2D &input, DoubleVector2D &output, int h, int w) {
	for (int i = 0; i < h; i++) {
		tanh(input[i], output[i], w);
	}
}

void NN_Utils::tanh(DoubleVector3D &input, DoubleVector3D &output, int n, int h, int w) {
	for (int i = 0; i < n; i++) {
		tanh(input[i], output[i], h, w);
	}
}

void NN_Utils::tanh(DoubleVector4D &input, DoubleVector4D &output, int m, int n, int h, int w) {
	for (int i = 0; i < m; i++) {
		tanh(input[i], output[i], n, h, w);
	}
}

void NN_Utils::tanh_back(double &input, double &output_delta, double &input_delta) {
	//input_delta = output_delta * (1 +input) * (1 - input);
	//input_delta = 1 - pow( (exp(input)-exp(-input)) / (exp(input)+exp(-input)), 2);
	input_delta = (1 - (std::tanh(input) * std::tanh(input))) * output_delta;
}

void NN_Utils::tanh_back(DoubleVector &input, DoubleVector &output_delta, DoubleVector &input_delta, int w) {
	for (int i = 0; i < w; i++) {
		tanh_back(input[i], output_delta[i], input_delta[i]);
	}
};

void NN_Utils::tanh_back(DoubleVector2D &input, DoubleVector2D &output_delta, DoubleVector2D &input_delta, int h, int w) {
	for (int i = 0; i < h; i++)
		tanh_back(input[i], output_delta[i], input_delta[i], w);
}

void NN_Utils::tanh_back(DoubleVector3D &input, DoubleVector3D &output_delta, DoubleVector3D &input_delta, int n, int h,
	int w) {
	for (int i = 0; i < n; i++)
		tanh_back(input[i], output_delta[i], input_delta[i], h, w);
}

void NN_Utils::tanh_back(DoubleVector4D &input, DoubleVector4D &output_delta, DoubleVector4D &input_delta, int m, int n,
	int h,
	int w) {
	for (int i = 0; i < m; i++)
		tanh_back(input[i], output_delta[i], input_delta[i], n, h, w);
}

void NN_Utils::d_tanh(double in, double &out) {
	out = (1 - (std::tanh(in) * std::tanh(in)));
}

void NN_Utils::d_tanh(DoubleVector in, DoubleVector &out, int w) {
	for (int i = 0; i < w; i++) {
		d_tanh(in[i], out[i]);
	}
};

void NN_Utils::d_tanh(DoubleVector2D in, DoubleVector2D &out, int h, int w) {
	for (int i = 0; i < h; i++)
		d_tanh(in[i], out[i], w);
}

void NN_Utils::one_hot(double input, DoubleVector &output, int n) {
	for (int i = 0; i < n; i++)
		output[i] = 0;
	output[(int)input] = 1;
}

void NN_Utils::one_hot(DoubleVector &input, DoubleVector2D &output, int m, int n) {
	for (int i = 0; i < m; i++)
		one_hot(input[i], output[i], n);
}


void NN_Utils::array_to_4_dimension(DoubleVector array, DoubleVector4D &output, vector<int>dimension) {

	auto vec_d = dimension;
	for (int i = 0; i<vec_d[0]; i++) {
		for (int j = 0; j<vec_d[1]; j++) {
			for (int k = 0; k<vec_d[2]; k++) {
				for (int l = 0; l<vec_d[3]; l++) {
					output[i][j][k][l] = array[i*(vec_d[1] * vec_d[2] * vec_d[3]) + j*(vec_d[2] * vec_d[3]) + k*(vec_d[3]) + l];
				}

			}
		}
	}
}

void NN_Utils::array_to_3_dimension(DoubleVector array, DoubleVector3D &output, int ch, int w, int h) {
	for (int c = 0; c<ch; c++) {
		for (int k = 0; k<h; k++) {
			for (int l = 0; l<w; l++) {
				output[c][k][l] = array[c*h*w + k*w + l];
			}
		}
	}
}

void NN_Utils::array_to_2_dimension(DoubleVector array, DoubleVector2D &output, int w, int h) {

	for (int k = 0; k<h; k++) {
		for (int l = 0; l<w; l++) {
			output[k][l] = array[k*w + l];
		}
	}
}



void NN_Utils::a_add_b_matrix(double input, double input_delta, double &output) {
	output = input + input_delta;
}

void NN_Utils::a_add_b_matrix(DoubleVector input, DoubleVector input_delta, DoubleVector &output, int w) {
	for (int i = 0; i < w; i++) {
		a_add_b_matrix(input[i], input_delta[i], output[i]);
	}
};

void
NN_Utils::a_add_b_matrix(DoubleVector2D input, DoubleVector2D input_delta, DoubleVector2D &output, int h, int w) {
	for (int i = 0; i < h; i++)
		a_add_b_matrix(input[i], input_delta[i], output[i], w);
}

void
NN_Utils::a_add_b_matrix(DoubleVector3D input, DoubleVector3D input_delta, DoubleVector3D &output, int n, int h,
	int w) {
	for (int i = 0; i < n; i++)
		a_add_b_matrix(input[i], input_delta[i], output[i], h, w);
}

void
NN_Utils::a_add_b_matrix(DoubleVector4D input, DoubleVector4D input_delta, DoubleVector4D &output, int m, int n,
	int h,
	int w) {
	for (int i = 0; i < m; i++)
		a_add_b_matrix(input[i], input_delta[i], output[i], n, h, w);
}

bool NN_Utils::gradient_check(double output_real_new, double output_new) {
	return (0.05 < fabs(output_real_new - output_new));
}

void NN_Utils::gradient_check(DoubleVector output_real_new, DoubleVector output_new, int w) {
	for (int i = 0; i < w; i++) {
		if (gradient_check(output_real_new[i], output_new[i]))
			cout << output_real_new[i] << " " << output_new[i] << endl;
	}
}

void NN_Utils::gradient_check(DoubleVector2D output_real_new, DoubleVector2D output_new, int h, int w) {
	for (int i = 0; i < h; i++)
		gradient_check(output_real_new[i], output_new[i], w);
}

void NN_Utils::gradient_check(DoubleVector3D output_real_new, DoubleVector3D output_new, int n, int h, int w) {
	for (int i = 0; i < n; i++)
		gradient_check(output_real_new[i], output_new[i], h, w);
}

void NN_Utils::gradient_check(DoubleVector4D output_real_new, DoubleVector4D output_new, int m, int n, int h, int w) {
	for (int i = 0; i < m; i++)
		gradient_check(output_real_new[i], output_new[i], n, h, w);
}



