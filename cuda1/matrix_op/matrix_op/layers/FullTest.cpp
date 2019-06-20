
#include "FullConnectedLayer.hpp"

#define IMG_W 28
#define IMG_H 28
#define LEARNING_RATE (0.02)

typedef std::vector<double> DoubleVector;
typedef std::vector<DoubleVector> DoubleVector2D;
typedef std::vector<DoubleVector2D> DoubleVector3D;
typedef std::vector<DoubleVector3D> DoubleVector4D;

inline void read_num(FILE *csv_file, int &curr) {
	static int c, sign = 1;
	curr = 0;
	while (~(c = fgetc(csv_file)) && (c < '0' || c > '9') && c != '-');
	c == '-' ? sign = -1 : curr = c - '0';
	while (~(c = fgetc(csv_file)) && (c >= '0' && c <= '9'))
		curr = (curr << 3) + (curr << 1) + c - '0';
}

void read_mnist_dataset(int m, int c, int h, int w, DoubleVector &flag, DoubleVector4D &data, const char *filePath) {
	FILE *csv_file = fopen(filePath, "r");
	int f_input = 0;

	for (int i = 0; i < m; i++) {
		read_num(csv_file, f_input);
		flag[i] = f_input;
		for (int j = 0; j < c; j++) {
			for (int k = 0; k < h; k++) {
				for (int l = 0; l < w; l++) {
					read_num(csv_file, f_input);
					data[i][j][k][l] = f_input;
				}
			}
		}
	}
	fclose(csv_file);
}

void gen_inputs_from_mnist(DoubleVector4D &input_data, DoubleVector &label, int img_num, const char *file_path) {

	unsigned long long c = 1, h = IMG_W, w = IMG_H;
	input_data.resize(img_num);
	label.resize(img_num, 0);
	for (int i = 0; i < img_num; i++) {
		input_data[i].resize(c);
		for (int j = 0; j < c; j++) {
			input_data[i][j].resize(h);
			for (int k = 0; k < h; k++) {
				input_data[i][j][k].resize(w, 0);
			}
		}
	}
	read_mnist_dataset((int)img_num, (int)c, (int)h, (int)w, label, input_data, file_path);

	for (int i = 0; i < img_num; i++) {
		for (int j = 0; j < c; j++) {
			for (int k = 0; k < h; k++) {
				for (int l = 0; l<w; l++) {
					input_data[i][j][k][l] = input_data[i][j][k][l] / double(256.0);
				}
			}
		}
	}
}

int man()
{
	DoubleVector4D mnist_data;
	DoubleVector label;
	int img_total = 10000;
	gen_inputs_from_mnist(mnist_data, label, img_total, "C:/Users/junjie/Desktop/offer/dnn/dnn/data/mnist_test.csv");

	double diff_mean_100 = 0.0;

	int img_w = IMG_W;
	int img_h = IMG_H;

	int labeldim[2] = {10 ,1};
	Tensor<double>* one_hot_label = new Tensor<double>(labeldim, 2);

	auto fullnet1 = new FullConnectedLayer<double>(1, IMG_W*IMG_H, 100, LEARNING_RATE);
	auto fullnet2 = new FullConnectedLayer<double>(1, 100, 10, LEARNING_RATE);

	for (int img_idx = 0; img_idx < 10000; img_idx++)
	{
		std::vector<DoubleVector2D> input_data = mnist_data[img_idx];

		int inputdim[2] = { IMG_W*IMG_H,1 };
		Tensor<double>* fn1_input = new Tensor<double>(inputdim, 2); //new 1
		
		for(int i=0; i< IMG_H; i++)
			for (int j = 0; j < IMG_W; j++)
			{
				fn1_input->data[i*IMG_W + j] = input_data[0][i][j];
			}
	
		//compute
		int fn1outdim[2] = { 100, 1 };
		Tensor<double>* fn1_out = new Tensor<double>(fn1outdim, 2); //new 2
		int fn2outdim[2] = { 10, 1 };
		Tensor<double>* fn2_out = new Tensor<double>(fn2outdim, 2); //new 3

		fullnet1->compute(fn1_input, fn1_out);
		fullnet2->compute(fn1_out, fn2_out);
		//print(fn2_out);

		//softmax
		softmax<double>(fn2_out, fn2_out);
		//cout << "softmax: ";
		//print(fn2_out);
		// 反向传播。。。
		for (int i = 0; i < 10; i++)
			one_hot_label->data[i] = 0;
		int index = label[img_idx];
		one_hot_label->data[index] = 1;

		for (int i = 0; i < 10; i++)
			fn2_out->data[i] -= one_hot_label->data[i];

		//打印结果
		double diff_mean = 0.0;

		for (int i = 0; i<10; i++) {
			diff_mean += fabs(fn2_out->data[i]);
		}

		diff_mean = diff_mean / 10.0;
		//printf("diff_mean:%f\n", diff_mean);
		diff_mean_100 += diff_mean;

		if (img_idx % 100 == 0 && img_idx > 0) {
			printf("img_idx:%d, error_mean:%f\n", img_idx, diff_mean_100 / 100.0);
			diff_mean_100 = 0;
		}

		//back2
		int fn2backdim[2] = {100, 1 };
		Tensor<double>* fn2_back = new Tensor<double>(fn2backdim, 2);  //new 4
		fullnet2->back_propagation(fn2_out, fn2_back);

		fullnet2->update(fn1_out, fn2_out);

		//back1
		int fn1backdim[2] = { 784, 1 };
		Tensor<double>* fn1_back = new Tensor<double>(fn1backdim, 2); // new 5
		fullnet1->back_propagation(fn2_back, fn1_back);

		fullnet1->update(fn1_input, fn2_back);

		//cout << "hha " << endl;

		delete fn1_input;
		delete fn1_out;
		delete fn2_out;
		delete fn2_back;
		delete fn1_back;

	}
	delete fullnet1;
	delete fullnet2;
	delete one_hot_label;

	system("pause");
	return 0;
}

int mai1n() {
	//
	auto full = new FullConnectedLayer<double>(1, 10, 4, 0.02);
	Tensor<double>* res = full->getW();
	cout << "W initial:" << endl;
	print(res);
	//input
	for (int i = 0; i < 1; i++)
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
		double *labeldata = new double[4]{ 1.0, 0, 0, 0 };
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



