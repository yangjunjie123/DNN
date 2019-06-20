//
// Created by shuai on 3/25/2019.
//

#include "Tensor_op.hpp"
#include "../layers/FullConnectedLayer.hpp"
/*
//FullConnect
int main() {
	//
	auto full = new FullConnectedLayer<double>(1, 10, 4, 0.02);
	Tensor<double>* res = full->getW();
	cout << "W initial:" << endl;
	print(res);
	//input
	for (int i=0; i < 100; i++)
	{
		double *data = new double[10]{ 1.0, 1, 2, 1, 1, 1, 1, 2, 1, 1 };
		int *inputdim = new int[2]{ 10,1 };
		Tensor<double>* input = new Tensor<double>(inputdim, 2);
		input->setData(data);
		cout << "input: ";
		print(input);

		//compute
		int temp1[2] = { 4, 1 };
		Tensor<double>* output = new Tensor<double>(temp1, 2);
		full->compute(input, output);
		cout << "output: ";
		print(output);

		//softmax
		softmax(output, output);
		cout << "softoutput: ";
		print(output);
		//反向传播。。。
		//标签
		double *labeldata = new double[4]{ 1.0, 0, 0, 0};
		int labeldim[2]{ 4,1 };
		Tensor<double>* label = new Tensor<double>(labeldim, 2);
		label->setData(labeldata);
		cout << "label: ";
		print(label);

		// diff of Y and label
		for (int i = 0; i < 4; i++)
			output->data[i] -= label->data[i];
		
		cout << "diff of label: ";
		print(output);

		//back
		int temp2[2] = { 10, 1 };
		Tensor<double>* backinput = new Tensor<double>(temp2, 2);
		full->back_propagation(output, backinput);
		for (int i = 0; i < backinput->getSum(); i++)
			cout << backinput->data[i] << " ";
		cout << endl;

		//update
		full->update(input, output);

		cout << "W after update:" << endl;
		Tensor<double>* res1 = full->getW();
		print(res1);

	}//for
	system("pause");
	return 0;
}
*/
// matrix multiply

int min() {
    int *dim1 = new int[3]{2, 2, 1};
    int *dim2 = new int[3]{2, 1, 2};
    int *dim3 = new int[3]{2, 2, 2};
    auto x = new Tensor<double>(dim1, 3);
    auto y = new Tensor<double>(dim2, 3);
    auto z = new Tensor<double>(dim3, 3);
    double *data1 = new double[4]{1, 2, 3, 4};
    double *data2 = new double[4]{3, 4, 5, 6};
    x->setData(data1);
    y->setData(data2);

    tensor_matmul_double(x, y, z);

	//tensor_matmul<double>(x, y, z);

    for(int i = 0; i < z->getSum(); i++)
        cout << z->data[i] << " ";

	system("pause");
	return 0;
}


//matrix dot add minus
/*
int main() {
    int *dim1 = new int[3]{2, 3, 2};
    int *dim2 = new int[3]{2, 3, 2};
    int *dim3 = new int[3]{2, 3, 2};
    auto x = new Tensor<int>(dim1, 3);
    auto y = new Tensor<int>(dim2, 3);
    auto z = new Tensor<int>(dim3, 3);
    int *data1 = new int[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    int *data2 = new int[12]{2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15};
    x->setData(data1);
    y->setData(data2);

    //tensor_dot<int>(x, y, z);
    tensor_add<int>(x, y, z);
    //tensor_minus<int>(x, y, z);
    for(int i = 0; i < z->getSum(); i++)
        cout << z->data[i] << " ";
	system("pause");
	return 0;
} 
*/

//number dot add minus
/*
int main() {
    int *dim1 = new int[3]{2, 3, 2};
    int *dim3 = new int[3]{2, 3, 2};
    auto x = new Tensor<int>(dim1, 3);
    auto z = new Tensor<int>(dim3, 3);
    int *data1 = new int[12]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    x->setData(data1);

    //tensor_dot<int>(12, x, z);
    //tensor_add<int>(10, x, z);
    tensor_minus<int>(10, x, z);
    for(int i = 0; i < z->getSum(); i++)
        cout << z->data[i] << " ";
}
*/

//different data type test
/*int main() {
    int *dim1 = new int[3]{2, 2, 3};
    int *dim2 = new int[3]{2, 3, 2};
    int *dim3 = new int[3]{2, 2, 2};
    auto x = new Tensor<float>(dim1, 3);
    auto y = new Tensor<float>(dim2, 3);
    auto z = new Tensor<float>(dim3, 3);
    auto *data1 = new float[12]{0.1, 1.1, 2.1, 2.4, 2.3, 0,6, -1, 1, 10, 1000,  232.21};
    auto *data2 = new float[12]{3, 2, 4, 6, 9, 10, 1, 2, 1, 0, 281, 42};
    x->setData(data1);
    y->setData(data2);

    tensor_matmul<float>(x, y, z);
    for(int i = 0; i < z->getSum(); i++)
        cout << z->data[i] << " ";
}
 */