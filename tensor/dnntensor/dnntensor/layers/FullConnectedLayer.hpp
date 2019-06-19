//
// Created by yangjunjie on 4/15/2019.
//

#ifndef AUTOML_FULLCONNECTEDLAYER_HPP
#define AUTOML_FULLCONNECTEDLAYER_HPP

#include "../Tensor/Tensor.hpp"
#include "../framework/LSTM.hpp"

template<class T>
static void Relu(Tensor<T> *input, Tensor<T> *output) {
	T *inputs = input->data;
	T *outputs = output->data;
	int sum = input->getSum() - 1;
	try {
		outputs[0] = inputs[0] > 0 ? inputs[0] : 0;
		while (sum--) {
			outputs++;
			inputs++;
			outputs[0] = inputs[0] > 0 ? inputs[0] : 0;
		}
	}
	catch (exception &e) {
		cout << "Standard exception: " << e.what() << endl;
	};
}

template<class T>
static void Relu_back(Tensor<T> *input, Tensor<T> *output_delta, Tensor<T> *input_delta) {
	T *inputs = input->data;
	T *outputs_delta = output_delta->data;
	T *inputs_delta = input_delta->data;
	int sum = input->getSum() - 1;
	try {
		inputs_delta[0] = inputs[0] >= 0 ? outputs_delta[0] : 0;
		while (sum--) {
			outputs_delta++;
			inputs++;
			inputs_delta[0] = inputs[0] >= 0 ? outputs_delta[0] : 0;
		}
	}
	catch (exception &e) {
		cout << "Standard exception: " << e.what() << endl;
	};
}

template<class T>
static void softmax(Tensor<T> *input, Tensor<T> *output) {
	T *inputs = input->data;
	T *outputs = output->data;
	int len = input->getSum() - 1;
	T sum = 0.0;
	try {
		for (int i = 0; i <= len; i++)
			sum += exp(inputs[i]);

		outputs[0] = exp(inputs[0]) / sum;
		while (len--) {
			outputs++;
			inputs++;
			outputs[0] = exp(inputs[0]) / sum;
		}
	}
	catch (exception &e) {
		cout << "Standard exception: " << e.what() << endl;
	};
}


template<class T>
void print(Tensor<T> *in) {
	int sum_count = in->getSum();
	for (int i = 0; i < sum_count; i++) {
		cout << in->data[i] << " ";
	}
	cout << endl;
}


template <class T>
class FullConnectedLayer{

protected:
	int input_dimension, output_dimension, batch_size;
	double alpha;
	Tensor<T> *W;
	Tensor<T> *m_input;  //这个还没使用
	Tensor<T> *m_output; //未经激活函数的y值
public:
	FullConnectedLayer() = delete;

	FullConnectedLayer(int batch_size, int input_dimension, int output_dimension, double alpha)
		: input_dimension(input_dimension)
		, output_dimension(output_dimension)
		, batch_size(batch_size)
		, alpha(alpha)
	{
		auto generator = new RandomGenerator<T>(1);
		int temp1[2] = { output_dimension, input_dimension };
		this->W = new Tensor<T>(temp1, 2);
		generator->XavierInitializer(W->data, output_dimension, input_dimension);
		delete generator;

		int inputdim[2] = { input_dimension,1 };
		this->m_input = new Tensor<T>(inputdim, 2);
		//print(m_input);
		int outputdim[2] = { output_dimension,1 };
		this->m_output = new Tensor<T>(outputdim, 2);
	}

	~FullConnectedLayer()
	{
		delete W;
		delete m_input;
		delete m_output;
	}
	
	Tensor<T>* getW()
	{
		return (this->W);
	}

	void compute(Tensor<T> *inputs, Tensor<T> *outputs) 
	{
		tensor_matmul<T>(this->W, inputs, this->m_output);

		//print(m_output);
		Relu<T>(this->m_output, outputs);
		//print(outputs);
	}

	//output_delta 反向传回的结果，inputs前向传播未激活的结果，outputs结果
	void back_propagation(Tensor<T> *output_delta,  Tensor<T> *outputs)
	{
		//W transpose
		int temp[2] = { input_dimension, output_dimension};
		Tensor<T>* new_W = new Tensor<T>(temp, 2);
		tensor_transpose(this->W, new_W);

		//back active
		Relu_back<T>(this->m_output, output_delta, outputs);

		//vector<int>resnew_W = new_W->getDimension();
		tensor_matmul<T>(new_W, output_delta, outputs);

		delete new_W;
	}
	
	
	//前向传播时的输入和输出
	void update(Tensor<T> *input, Tensor<T> *output) 
	{
		int temp1[2] = { 1, input_dimension };
		Tensor<T>* new_input = new Tensor<T>(temp1, 2);
		tensor_transpose(input, new_input);

		int temp2[2] = {output_dimension, input_dimension};
		Tensor<T>* diff = new Tensor<T>(temp2, 2);

		tensor_matmul<T>(output, new_input, diff);
		//cout << "hha" << endl;
		for (int i = 0; i < diff->getSum(); i++)
		{
			diff->data[i] *= alpha;
			W->data[i] -= diff->data[i];
		}
		
		delete new_input;
		delete diff;
	}

};


#endif //AUTOML_FULLCONNECTEDLAYER_HPP
