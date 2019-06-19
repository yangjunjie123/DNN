//
//  cnn_test.cpp
//  automl
//
//  Created by Jack on 2019/3/5.
//  Copyright © 2019年 PA. All rights reserved.
//


//#include <unistd.h>
//#include "nn_frame.hpp"
//#include "nn_conv_layer.hpp"
//#include "nn_pooling_layer.hpp"
#include "nn_layer.hpp"
//#include "BatchNormalization.hpp"
//#include "../utils/utils.hpp"

using namespace std;
using namespace pa_nn;


#define IMG_W 28
#define IMG_H 28
#define LEARNING_RATE (0.02)

#define CONV_KERNEL_SIZE 3
#define POOL_SIZE 2
#define IN_CHANNEL_NUM 1
#define CHANNEL_NUM 1
#define OUT_PUT_SIZE 10

//产生一个3维的0到1的随机数
void gen_inputs(vector<DoubleVector2D> &input_data, int in_ch, int img_w, int img_h){
    
    srand(time(NULL));
    
    for(int c=0; c<in_ch; c++){
        for(int i=0; i<img_h; i++){
            for(int j=0; j<img_w; j++){
                input_data[c][i][j] = double(rand()%255/256.0);
            }
        }
    }
}

inline void read_num(FILE *csv_file, int &curr) {
    static int c, sign = 1;
    curr = 0;
    while (~(c = fgetc(csv_file)) && (c < '0' || c > '9') && c != '-');
    c == '-' ? sign = -1: curr = c - '0';
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

void gen_inputs_from_mnist(DoubleVector4D &input_data, DoubleVector &label, int img_num, const char *file_path){
    
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
    read_mnist_dataset((int) img_num, (int) c, (int) h, (int) w, label, input_data, file_path);
    
    for (int i = 0; i < img_num; i++) {
        for (int j = 0; j < c; j++) {
            for (int k = 0; k < h; k++) {
                for(int l=0; l<w; l++){
                    input_data[i][j][k][l] = input_data[i][j][k][l] / double(256.0);
                }
            }
        }
    }
}


int find_max_pos_from_array(DoubleVector input, int size){
    double max = input[0];
    int pos = 0;
    for(int i=1; i<size; i++){
        if(input[i] > max){
            max = input[i];
            pos = i;
        }
    }
    return pos;
}



void cnn_train_2()
{
    DoubleVector4D mnist_data;
    DoubleVector label;
    int img_total = 10000; //共一万张图片
    gen_inputs_from_mnist(mnist_data, label, img_total, "C:/Users/junjie/Desktop/offer/dnn/dnn/data/mnist_test.csv");
    
    
    double diff_mean_100 = 0.0; //输出与标签的差值
    
    int img_w = IMG_W;
    int img_h = IMG_H;
    
    auto fullnet1 = new NN_Layer(IMG_W*IMG_H, 100, LEARNING_RATE, (char*)"fn_1", ACTIVATION_RELU);
    
    auto fullnet2 = new NN_Layer(100, 10, LEARNING_RATE, (char*)"fn_2", ACTIVATION_RELU);
//
    //auto fullnet3 = new NN_Layer(32, 10, LEARNING_RATE, (char*)"fn_3");
    
    for(int idx=0; idx<100000; idx++){
        
        auto img_idx = idx%10000;
        vector<DoubleVector2D> input_data = mnist_data[img_idx]; //input_data是3维的
    
        DoubleVector fn1_input; //这里输入是一维的
        for(int i=0; i<IMG_H; i++){
            for(int j=0; j<IMG_W; j++){
                fn1_input.push_back(input_data[0][i][j]);
            }
        }
/*这3行应该是要注释掉的        
        DoubleVector s1_output;
        s1_output.resize(IMG_W*IMG_H);
        NN_Utils::softmax(fn1_input, s1_output, IMG_W*IMG_H);//这里为嘛要用softmax
*/        
        auto fullnet1_out = fullnet1->compute(fn1_input); //然后这里的s1_output应该改为fn1_input
        
//        DoubleVector s1_output;
//        s1_output.resize(fullnet1->m_neurons_num);
//        NN_Utils::softmax(fullnet1_out, s1_output, fullnet1->m_neurons_num);
        
        auto fullnet2_out = fullnet2->compute(fullnet1_out);
//
        //auto fullnet3_out = fullnet3->compute(fullnet2_out);
//
        DoubleVector softmax_out;
        softmax_out.resize(fullnet2->m_neurons_num);
        NN_Utils::softmax(fullnet2_out, softmax_out, fullnet2->m_neurons_num);
        
        // 反向传播 ......
        
        DoubleVector one_hot_label;
        one_hot_label.resize(OUT_PUT_SIZE);
        NN_Utils::one_hot(label[img_idx], one_hot_label, OUT_PUT_SIZE);//一张图片对应长度为10的一维向量
//
//
        DoubleVector out_deltaY;
        out_deltaY.resize(OUT_PUT_SIZE);
        double max_value = 0.0;
        int max_idx = 0;
        for(int i=0; i<OUT_PUT_SIZE; i++ ){
            out_deltaY[i] =   softmax_out[i] - one_hot_label[i];
            if(softmax_out[i] > max_value){
                max_value = softmax_out[i];
                max_idx = i;
            }
        }
        
//        NN_Utils::softmax_back_real(softmax_out, one_hot_label, out_deltaY, fullnet2->m_neurons_num);
//
//
//        // 输出与标签的差值

        double diff_mean = 0.0;

        for(int i=0; i<OUT_PUT_SIZE; i++){
            diff_mean += fabs(out_deltaY[i]);
        }

        diff_mean = diff_mean / 10.0;
        //printf("diff_mean:%f\n", diff_mean);
        diff_mean_100 += diff_mean;

        if(img_idx%100 == 0 && img_idx > 0){
            printf("img_idx:%d, error_mean:%f, y:%d label:%f\n", img_idx, diff_mean_100/100.0, max_idx, label[img_idx]);
            diff_mean_100 = 0;
        }
       
        
        // back propagation
        //fullnet3->back_propagation_op(out_deltaY);
        //fullnet3->update(out_deltaY);
//
        fullnet2->back_propagation_op(out_deltaY);
        fullnet2->update(out_deltaY);
        
        fullnet1->back_propagation_op(fullnet2->m_deltas);
        fullnet1->update(fullnet2->m_deltas);
        
        int end = 0;
        
    
    }
}

/*
void cnn_train_3()
{
    
    DoubleVector4D mnist_data;
    DoubleVector label;
    int img_total = 10000;
    gen_inputs_from_mnist(mnist_data, label, img_total, "/Users/jack/Projects/automl/automl/data/mnist_test.csv");
    double diff_mean_100 = 0.0;
    
    int img_w = IMG_W;
    int img_h = IMG_H;
    
    // 1卷积层 28x28
    auto conv1 = new NN_Conv_layer(CONV_KERNEL_SIZE, IN_CHANNEL_NUM, img_w, img_h, CHANNEL_NUM, ACTIVATION_RELU, LEARNING_RATE);
    int conv1_out_w = img_w - CONV_KERNEL_SIZE + 1;
    int conv1_out_h = img_h - CONV_KERNEL_SIZE + 1;
    
    // 2池化层 26x26
    auto pool2 = new NN_Pooling_layer(POOL_SIZE, CHANNEL_NUM, conv1_out_w, conv1_out_h, POOL_MODE_AVG);
    int pool2_out_w = conv1_out_w / 2;
    int pool2_out_h = conv1_out_h / 2;
    
    // 3卷积层 28x28
    auto conv3 = new NN_Conv_layer(CONV_KERNEL_SIZE, CHANNEL_NUM, pool2_out_w, pool2_out_h, CHANNEL_NUM, ACTIVATION_RELU, LEARNING_RATE);
    int conv3_out_w = pool2_out_w - CONV_KERNEL_SIZE + 1;
    int conv3_out_h = pool2_out_h - CONV_KERNEL_SIZE + 1;
    
    // 4池化层 26x26
    auto pool4 = new NN_Pooling_layer(POOL_SIZE, CHANNEL_NUM, conv3_out_w, conv3_out_h, POOL_MODE_AVG);
    int pool4_out_w = conv3_out_w / 2;
    int pool4_out_h = conv3_out_h / 2;
    
    // batch_normalization 1
    //auto bn2_layer = new NN_batch_normalization(1, pool4_out_w, pool4_out_h, CHANNEL_NUM);
    
    // fully net
    auto fullnet5 = new NN_Layer(pool4_out_w*pool4_out_h*CHANNEL_NUM, OUT_PUT_SIZE, LEARNING_RATE, (char *)"fn_5", ACTIVATION_RELU);
    
    // ############## loop image #############
    for(int img_idx=0; img_idx<img_total; img_idx++){
        
        vector<DoubleVector2D> input_data = mnist_data[0];
        
        vector<DoubleVector2D> conv1_out;
        conv1_out.resize(CHANNEL_NUM);
        for(int i=0; i<CHANNEL_NUM; i++){
            conv1_out[i].resize(conv1_out_h);
            for(int h=0; h<conv1_out_h; h++){
                conv1_out[i][h].resize(conv1_out_w);
            }
        }
        conv1->compute(input_data, conv1_out);
        
        vector<DoubleVector2D> pool2_out;
        pool2_out.resize(CHANNEL_NUM);
        for(int i=0; i<CHANNEL_NUM; i++){
            pool2_out[i].resize(pool2_out_h);
            for(int h=0; h<pool2_out_h; h++){
                pool2_out[i][h].resize(pool2_out_w);
            }
        }
        pool2->compute(conv1_out, pool2_out);
        
        vector<DoubleVector2D> conv3_out;
        conv3_out.resize(CHANNEL_NUM);
        for(int i=0; i<CHANNEL_NUM; i++){
            conv3_out[i].resize(conv3_out_h);
            for(int h=0; h<conv3_out_h; h++){
                conv3_out[i][h].resize(conv3_out_w);
            }
        }
        conv3->compute(pool2_out, conv3_out);
        
        vector<DoubleVector2D> pool4_out;
        pool4_out.resize(CHANNEL_NUM);
        for(int i=0; i<CHANNEL_NUM; i++){
            pool4_out[i].resize(pool4_out_h);
            for(int h=0; h<pool4_out_h; h++){
                pool4_out[i][h].resize(pool4_out_w);
            }
        }
        pool4->compute(conv3_out, pool4_out);
        
        // batch normalization 1
//        int bn1_img_w = pool4_out_w;
//        int bn1_img_h = pool4_out_h;
//        auto bn1_layer = new NN_batch_normalization(1, bn1_img_w, bn1_img_h, CHANNEL_NUM);
//        DoubleVector4D bn_input1, bn_output1;
//        bn_input1.resize(1, pool4_out);
//        bn_output1.resize(1, pool4_out);
//        bn1_layer->compute(bn_input1, bn_output1);
        
        
        
        // 5全连接层
        vector<double> input5;
        for(int c=0; c< CHANNEL_NUM; c++){
            for(int i=0; i<pool4_out_h; i++){
                for(int j=0; j<pool4_out_w; j++){
                    input5.push_back(pool4_out[c][i][j]);
                }
            }
        }

        auto fullnet5_out = fullnet5->compute(input5);
        
        DoubleVector softmax_out;
        softmax_out.resize(fullnet5->m_neurons_num);
        NN_Utils::softmax(fullnet5_out, softmax_out, fullnet5->m_neurons_num);
        
        DoubleVector2D fn_out;
        fn_out.resize(1);
        fn_out[0] = fullnet5_out;
       // NN_Utils::print_debug_data_2_file(DEBUG_FILE_NAME, fn_out, OUT_PUT_SIZE, 1, "fullnet5_out", false);
        
        
        // 反向传播 ......
        
        DoubleVector one_hot_label;
        one_hot_label.resize(OUT_PUT_SIZE);
        NN_Utils::one_hot(label[img_idx], one_hot_label, OUT_PUT_SIZE);
        
        // 输出与标签的差值
        DoubleVector out_diff;
        double diff_mean = 0.0;
        out_diff.resize(OUT_PUT_SIZE);
        for(int i=0; i<OUT_PUT_SIZE; i++){
            out_diff[i] -= one_hot_label[i] - softmax_out[i];
            diff_mean += fabs(out_diff[i]);
        }
        
        diff_mean = diff_mean / 10;
        diff_mean_100 += diff_mean;
        
        if(img_idx%100 == 0 && img_idx >0 ){
            printf("img_idx:%d, 100 pics error mean:%f\n", img_idx, diff_mean_100/100);
            diff_mean_100 = 0;
        }
        
        // 5 fullnet5 back propagation
        fullnet5->back_propagation_op(out_diff);
        fullnet5->update(out_diff);
        
        DoubleVector4D fullnet5_back_delta;
        NN_Utils::resize_vector4D(fullnet5_back_delta, 1, CHANNEL_NUM, pool4_out_w, pool4_out_h);
        vector<int> p4_dimension = {1, CHANNEL_NUM, pool4_out_w, pool4_out_h};
        NN_Utils::array_to_4_dimension(fullnet5->m_deltas, fullnet5_back_delta, p4_dimension);
        
        // bn1_layer back propagation
//        DoubleVector4D bn1_back_output;
//        resize_vector4D(bn1_back_output, 1, CHANNEL_NUM, bn1_img_w, bn1_img_h);
//        bn1_layer->back_propagation(bn_input1, fullnet5_back_delta, bn1_back_output);
        

        // 4池化层反传
        pool4->back_propagation(fullnet5_back_delta[0], pool4_out_w, pool4_out_h);
        
        // 3卷积层反传
        conv3->back_propagation(pool4->delta, pool4->in_w, pool4->in_h);
        
        // 2池化层反传
        pool2->back_propagation(conv3->delta, pool2_out_w, pool2_out_h);
        
        // 1卷积层反传
        conv1->back_propagation(pool2->delta, pool2->in_w, pool2->in_h);
        
        //        printf("%d one image tained..........\n", img_idx+1);
        
    }
}
*/

/*
void cnn_train()
{
    
    DoubleVector all_error_list;
    DoubleVector4D mnist_data;
    DoubleVector label;
    int img_total = 10000;
    gen_inputs_from_mnist(mnist_data, label, img_total, "/Users/jack/Projects/automl/automl/data/mnist_test.csv");
    double diff_mean_100 = 0.0;
    
    int out_channels = 32;
    
    // initialize layers
    auto conv1 = new NN_Conv_layer(CONV_KERNEL_SIZE, IN_CHANNEL_NUM, IMG_W, IMG_H, out_channels, ACTIVATION_RELU, LEARNING_RATE);
    int conv1_out_w = IMG_W - CONV_KERNEL_SIZE + 1;
    int conv1_out_h = IMG_H - CONV_KERNEL_SIZE + 1;
    
    auto conv2 = new NN_Conv_layer(CONV_KERNEL_SIZE, out_channels, conv1_out_w, conv1_out_h, out_channels, ACTIVATION_RELU, LEARNING_RATE);
    int conv2_out_w = conv1_out_w - CONV_KERNEL_SIZE + 1;
    int conv2_out_h = conv1_out_h - CONV_KERNEL_SIZE + 1;
    
//    auto pool3 = new NN_Pooling_layer(POOL_SIZE, 32, conv2_out_w, conv2_out_h, POOL_MODE_AVG);
//    int pool3_out_w = conv2_out_w / 2;
//    int pool3_out_h = conv2_out_h / 2;
//
//    auto conv4 = new NN_Conv_layer(CONV_KERNEL_SIZE, 32, pool3_out_w, pool3_out_h, 32, ACTIVATION_RELU, LEARNING_RATE);
//    int conv4_out_w = pool3_out_w - CONV_KERNEL_SIZE + 1;
//    int conv4_out_h = pool3_out_h - CONV_KERNEL_SIZE + 1;
//
//    auto conv5 = new NN_Conv_layer(CONV_KERNEL_SIZE, 32, conv4_out_w, conv4_out_h, 64, ACTIVATION_RELU, LEARNING_RATE);
//    int conv5_out_w = conv4_out_w - CONV_KERNEL_SIZE + 1;
//    int conv5_out_h = conv4_out_h - CONV_KERNEL_SIZE + 1;
    
    auto pool6 = new NN_Pooling_layer(POOL_SIZE, out_channels, conv2_out_w, conv2_out_h, POOL_MODE_MAX);
    int pool6_out_w = conv2_out_w / 2;
    int pool6_out_h = conv2_out_h / 2;
    
    
    // fully net
    auto fullnet1 = new NN_Layer(pool6_out_w*pool6_out_h*out_channels, OUT_PUT_SIZE, LEARNING_RATE, (char *)"fn_1", ACTIVATION_RELU);
    
    all_error_list.resize(img_total);
    // ############## loop image #############
    for(int img_idx=0; img_idx<img_total; img_idx++){
        
        vector<DoubleVector2D> input_data = mnist_data[img_idx];
        
        vector<DoubleVector2D> conv1_out;
        NN_Utils::resize_vector3D(conv1_out, out_channels, conv1_out_w, conv1_out_h);
        conv1->compute(input_data, conv1_out);
        
        vector<DoubleVector2D> conv2_out;
        NN_Utils::resize_vector3D(conv2_out, out_channels, conv2_out_w, conv2_out_h);
        conv2->compute(conv1_out, conv2_out);
        
//        vector<DoubleVector2D> pool3_out;
//        NN_Utils::resize_vector3D(pool3_out, 32, pool3_out_w, pool3_out_h);
//        pool3->compute(conv2_out, pool3_out);
//
//        vector<DoubleVector2D> conv4_out;
//        NN_Utils::resize_vector3D(conv4_out, 32, conv4_out_w, conv4_out_h);
//        conv4->compute(pool3_out, conv4_out);
//
//        vector<DoubleVector2D> conv5_out;
//        NN_Utils::resize_vector3D(conv5_out, 64, conv5_out_w, conv5_out_h);
//        conv5->compute(conv4_out, conv5_out);
//
        vector<DoubleVector2D> pool6_out;
        NN_Utils::resize_vector3D(pool6_out, out_channels, pool6_out_w, pool6_out_h);
        pool6->compute(conv2_out, pool6_out);
        
        
        DoubleVector fn_inputs;
        NN_Utils::vector3D_2_array(pool6_out, out_channels, pool6_out_w, pool6_out_h, fn_inputs);
        
        // 全连接层
        auto fullnet1_out = fullnet1->compute(fn_inputs);
     
        // softmax
        DoubleVector softmax_out;
        softmax_out.resize(fullnet1->m_neurons_num);
        NN_Utils::softmax(fullnet1_out, softmax_out, fullnet1->m_neurons_num);
        
        
        // 反向传播 ......
        
        DoubleVector one_hot_label;
        one_hot_label.resize(OUT_PUT_SIZE);
        NN_Utils::one_hot(label[img_idx], one_hot_label, OUT_PUT_SIZE);
        
        
        // 输出与标签的差值
        DoubleVector out_diff;
        double diff_mean = 0.0;
        out_diff.resize(OUT_PUT_SIZE);
        for(int i=0; i<OUT_PUT_SIZE; i++){
            out_diff[i] = softmax_out[i] - one_hot_label[i];
            diff_mean += fabs(out_diff[i]);
        }
        
        int max_idx = find_max_pos_from_array(softmax_out, OUT_PUT_SIZE);
        if(int(label[img_idx]) == max_idx){
            all_error_list[img_idx] = 1;
        }else{
            all_error_list[img_idx] = 0;
        }
        
        
        diff_mean = diff_mean / 10;
        diff_mean_100 += diff_mean;
        
        if(img_idx%100 == 0 && img_idx >0 ){
            int sum_count = 0;
            for(int i=0; i<img_idx; i++){
                sum_count += all_error_list[i];
            }
            double right_rate = (double)sum_count / img_idx;
            
            printf("img_idx:%d, 100 pics error mean:%f, right_rate:%f\n", img_idx, diff_mean_100/100, right_rate);
            diff_mean_100 = 0;
        }
        
        // 5 fullnet back propagation
        fullnet1->back_propagation_op(out_diff);
        fullnet1->update(out_diff);
        
        DoubleVector3D fn1_deltas;
        NN_Utils::resize_vector3D(fn1_deltas, out_channels, pool6_out_w, pool6_out_h);
        NN_Utils::array_to_3_dimension(fullnet1->m_deltas, fn1_deltas, out_channels, pool6_out_w, pool6_out_h);
        
        pool6->back_propagation(fn1_deltas, pool6_out_w, pool6_out_h);
        
//        conv5->back_propagation(pool6->delta, conv5_out_w, conv5_out_h);
//
//        conv4->back_propagation(conv5->delta, conv4_out_w, conv4_out_h);
//
//        pool3->back_propagation(conv4->delta, pool3_out_w, pool3_out_h);
        
        conv2->back_propagation(pool6->delta, conv2_out_w, conv2_out_h);
        
        conv1->back_propagation(conv2->delta, conv1_out_w, conv1_out_h);
        
        //NN_Utils::print_debug_data_2_file(DEBUG_FILE_NAME, conv1->kernels[0][0], conv1->k_size, conv1->k_size, "conv1_kernel", true);
        
    }
}
*/

/*
void cnn_train5()
{//3个全联接层测试
    DoubleVector4D mnist_data;
    DoubleVector label;
    int img_total = 10000;
    gen_inputs_from_mnist(mnist_data, label, img_total, "/Users/sichaoming/Desktop/CNN/CNN/data/mnist_test.csv");
    DoubleVector all_error_list;
    all_error_list.resize(img_total);
    
    // fully net
    auto fullnet1 = new NN_Layer(IMG_W*IMG_H, 300, LEARNING_RATE, (char*)"fn_1",ACTIVATION_RELU);
	auto fullnet2 = new NN_Layer(300, 100, LEARNING_RATE, (char*)"fn_2",ACTIVATION_RELU);
	auto fullnet3 = new NN_Layer(100, 10, LEARNING_RATE, (char*)"fn_3",ACTIVATION_RELU);
	
    for(int idx=0; idx<img_total; idx++)
    {
        vector<DoubleVector2D> input_data = mnist_data[idx];
        //Fullnet1
        vector<double> f1_input;
        for(int i=0; i<IMG_H; i++)
        {
            for(int j=0; j<IMG_W; j++)
            {
                    f1_input.push_back(input_data[0][i][j]);
            }
        }
        auto fullnet1_out = fullnet1->compute(f1_input);
        
        //Fullnet2
        auto fullnet2_out = fullnet2->compute(fullnet1_out);
       
        //Fullnet3
        auto fullnet3_out = fullnet3->compute(fullnet2_out);
		
		//softmax
        DoubleVector f3_softmax_out;
        f3_softmax_out.resize(fullnet3->m_neurons_num);
        NN_Utils::softmax(fullnet3_out, f3_softmax_out, fullnet3->m_neurons_num);
        
        // 反向传播 ......
        DoubleVector one_hot_label;
        one_hot_label.resize(10);
        NN_Utils::one_hot(label[idx], one_hot_label, 10);
        
        // 输出与标签的差值
        DoubleVector out_deltaY;
        out_deltaY.resize(10);
		for(int i=0; i<OUT_PUT_SIZE; i++ ){
            out_deltaY[i] -= one_hot_label[i] - f3_softmax_out[i];
            }
        }
		
        int max_idx = find_max_pos_from_array(f3_softmax_out, 10);
        if(int(label[idx]) == max_idx)
            all_error_list[idx] = 1;
        else
            all_error_list[idx] = 0;

        if(idx%100 == 0 && idx > 0)
        {
            int sum_count = 0;
            for(int i=0; i<idx; i++){
                sum_count += all_error_list[i];
            }
            double right_rate = (double)sum_count/idx;
            printf("img_idx:%d, right_rate:%f\n",idx, right_rate);
        }
        
        //fullnet3 back propagation
        fullnet3->back_propagation_op(out_deltaY);
        fullnet3->update(out_deltaY);
        //fullnet2 back propagation
        fullnet2->back_propagation_op(fullnet3->m_deltas);
        fullnet2->update(fullnet3->m_deltas);
        //fullnet1 back propagation
        fullnet1->back_propagation_op(fullnet2->m_deltas);
        fullnet1->update(fullnet2->m_deltas);
    }
}
*/