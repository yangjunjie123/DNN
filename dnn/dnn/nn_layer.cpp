//
//  nn_layer.cpp
//  automl
//
//  Created by Jack on 2019/2/24.
//  Copyright © 2019年 PA. All rights reserved.
//

#include "nn_layer.hpp"

NN_Layer::NN_Layer(int input_size, int neurons_num, double alpha, char* layer_name, int activation) {
	this->m_neurons_num = neurons_num;
	this->m_input_size = input_size;
	this->m_activation = activation;
	this->self = this;
	this->alpha = alpha;
	this->m_next_layer = NULL;

	this->m_inputs.resize(input_size);
	this->m_deltas.resize(input_size);
	strncpy_s(this->layer_name, layer_name, 10);

	// initial weights
	this->m_weights.resize(neurons_num);
	for (int i = 0; i<neurons_num; i++) {
		this->m_weights[i].resize(input_size);
	}
	//    DoubleVector weights;
	//    weights.resize(this->m_neurons_num*this->m_input_size);
	double rand_left, rand_right;
    rand_left = -std::sqrt(6/(double)(this->m_input_size+this->m_neurons_num));
	rand_right = std::sqrt(6 / (double)(this->m_input_size + this->m_neurons_num));
	//NN_Utils::util_get_rand_array(weights, this->m_neurons_num*this->m_input_size, rand_left, rand_right);
	for (int i = 0; i<this->m_neurons_num; i++) {
		DoubleVector weights;
		weights.resize(this->m_input_size);
		NN_Utils::gen_normal_distribution_array(this->m_input_size, 0, rand_right, weights);
		for (int j = 0; j<this->m_input_size; j++) {
			//this->m_weights[i][j] = weights[i*this->m_input_size+j];
			this->m_weights[i][j] = weights[j];
		}
	}
}

NN_Layer::~NN_Layer() {
	// TO BE RELEASE RESOURE...
	printf("NN_Layer deconstruct......\n");
}


NN_Layer* NN_Layer::get_next_layer() {
	return this->m_next_layer;
}

void NN_Layer::set_next_layer(NN_Layer *layer) {
	this->m_next_layer = layer;
}

int NN_Layer::get_out_put_size() {
	return this->m_neurons_num;
}

void NN_Layer::connect(NN_Layer* last_layer) {
	if (last_layer) {
		this->m_input_size = last_layer->get_out_put_size();
		// connet last layer to the new one
		last_layer->set_next_layer(this);
	}
}

vector<double> NN_Layer::compute(vector<double> input) {

	vector<double> vec_ret;

	this->m_inputs.assign(input.begin(), input.end());
	this->m_activate_level.resize(this->m_neurons_num);

	// 将x input转置
	DoubleVector2D x_input;
	x_input.resize(this->m_input_size);
	for (int i = 0; i<this->m_input_size; i++) {
		x_input[i].push_back(this->m_inputs[i]);
	}

	auto y_output = NN_Utils::util_matrix_multiply(this->m_weights, x_input); // 输出应该是 input_size行，单列

																			  // 转置输出
	vec_ret.resize(this->m_neurons_num);

	for (int i = 0; i<this->m_neurons_num; i++) {
		this->m_activate_level[i] = y_output[i][0];
		vec_ret[i] = NN_Utils::activation_op(this->m_activation, this->m_activate_level[i]);
	}

	return vec_ret;
}

void NN_Layer::back_propagation(vector<double> distance, NN_Layer* next_layer) {
}

void NN_Layer::back_propagation_op(vector<double> next_layer_deltas) {

	// 转置weights矩阵
	DoubleVector2D wT;
	wT.resize(this->m_input_size);
	for (int i = 0; i<this->m_input_size; i++) {
		wT[i].resize(this->m_neurons_num);
	}

	for (int i = 0; i<this->m_input_size; i++) {
		for (int j = 0; j<this->m_neurons_num; j++) {
			wT[i][j] = this->m_weights[j][i];
		}
	}

	DoubleVector2D deltaY;
	deltaY.resize(this->m_neurons_num);
	for (int i = 0; i<this->m_neurons_num; i++) {
		deltaY[i].resize(1);
	}

	// 转置后的 weights->wT * next_deltas = delta_X
	for (int i = 0; i<this->m_neurons_num; i++) {
		double delta_y = NN_Utils::activation_op_back(this->m_activation, this->m_activate_level[i], next_layer_deltas[i]);
		//double delta_y = NN_Utils::util_sigmoid_grad(this->m_activate_level[i]) * next_layer_deltas[i];
		deltaY[i][0] = delta_y;
	}

	auto x_deltas = NN_Utils::util_matrix_multiply(wT, deltaY); // 输出应该是 m_input_size行，单列
	for (int i = 0; i<this->m_input_size; i++) {
		this->m_deltas[i] = x_deltas[i][0];
	}
}

void NN_Layer::update(vector<double> next_layer_deltas) {

	//printf("\n\n%s layer update\n", this->layer_name);
	for (int m = 0; m<this->m_neurons_num; m++) {
		double delta_y = NN_Utils::activation_op_back(this->m_activation, this->m_activate_level[m], next_layer_deltas[m]);
		//double delta_y = NN_Utils::util_sigmoid_grad(this->m_activate_level[m]) * next_layer_deltas[m];
		for (int i = 0; i<this->m_input_size; i++) {
			double diff = this->alpha * delta_y * this->m_inputs[i];

			//printf("m_weights[%d][%d]=%f - %f \n", m, i, this->m_weights[m][i], diff);

			this->m_weights[m][i] -= diff;
		}
		//printf("\n\n");
	}

}

