//
//  nn_layer.hpp
//  automl
//
//  Created by Jack on 2019/2/24.
//  Copyright © 2019年 PA. All rights reserved.
//

#ifndef nn_layer_hpp
#define nn_layer_hpp

#include <string.h>
#include "nn_frame.hpp"

using namespace std;

class NN_Layer {
public:
	int m_neurons_num;
	int m_activation;
	NN_Layer* m_next_layer;
	NN_Layer* self;
	DoubleVector2D m_weights; // neurons_num * input_size大小
	DoubleVector m_deltas; // input_size大小
	DoubleVector m_inputs;
	DoubleVector m_activate_level; // 计算输出y值（未经激活函数的y值）
	double alpha;             // 学习率
	char layer_name[10];
public:
	int m_input_size;
	NN_Layer() = default; // 默认构造函数
	NN_Layer(int input_size, int neurons_num, double alpha, char* layer_name, int activation = ACTIVATION_SIGMOID);
	~NN_Layer();
	NN_Layer* get_next_layer();
	void set_next_layer(NN_Layer *layer);
	int get_out_put_size();
	void connect(NN_Layer* last_layer = NULL);
	vector<double> compute(vector<double> input);
	void back_propagation(vector<double> distance, NN_Layer *next_layer = NULL);
	void back_propagation_op(vector<double> distance);
	void update(vector<double> next_layer_deltas);


};

#endif /* nn_layer_hpp */

