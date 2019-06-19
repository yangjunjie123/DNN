//
// Created by shuai on 3/28/2019.
//

#include "LSTM.hpp"
#include <math.h>


//int main(){
//
//    auto lstm_for_propa =  new LSTMOP<float>(10, 2, 3, 2);
//
//
//    vector<Tensor<float> *> x;
//    int *temp_dim1 = new int[2]{10, 2};
//    for(int i = 0; i < 5; i++) {
//        auto temp1 = new Tensor<float>(temp_dim1, 2);
//        float temp2[20];
//        for(int j = 0; j < 20; j++)
//            temp2[j] = j  * i;
//        temp1->setData(temp2);
//        x.push_back(temp1);
//    }
//    lstm_for_propa->forward_propagation(x);
//    Tensor<float> label(temp_dim1, 2);
//    for(int i = 0; i < 20; i++) {
//        label.data[i] = i % 4 ? 1 : 0;
//    }
//    for(int i = 0; i < 100000; i++){
//        lstm_for_propa->back_propagation(x, &label);
//        lstm_for_propa->forward_propagation(x);
//    }
//
//    for(int i = 0; i < 5; i++)
//        delete x[i];
//   return 0;
//}
