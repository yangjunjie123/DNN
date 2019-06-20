//
// Created by shuai on 3/25/2019.
//

#ifndef AUTOML_TENSOR_OP_HPP
#define AUTOML_TENSOR_OP_HPP

#include <exception>
#include <iostream>
#include "Tensor.hpp"


template<class T>
void tensor_matmul(const Tensor<T> *W, const Tensor<T> *X, Tensor<T> *output, bool w_flag = false, bool x_flag = false, bool out_flag = false) {
    auto w_dim = W->getDimension();
    auto x_dim = X->getDimension();
    int w[2], x[2];
    w[0] = w_dim[w_dim.size() - 2];
    w[1] = w_dim[w_dim.size() - 1];
    x[0] = x_dim[x_dim.size() - 2];
    x[1] = x_dim[x_dim.size() - 1];
    if (w_dim.size() != x_dim.size() || W->getSum() / w[0] != X->getSum() / x[1])
        throw ("matmul input error and program terminated in " + W->getName() + " and " + X->getName() + "\n");

    int w_warp = w[0] * w[1], x_warp = x[0] * x[1], out_warp = w[0] * x[1];
    int sum = W->getSum() / w_warp;
    T *w_data = W->data;
    if(w_flag)
        w_data = W->error;
    T *x_data = X->data;
    if(x_flag)
        x_data = X->error;
    T *out_data = output->data;
    if(out_flag)
        out_data = output->error;
    try {
        for (int i = 0; i < w[0]; i++) {
            for (int j = 0; j < x[1]; j++) {
                out_data[i * x[1] + j] = 0;
                for (int k = 0; k < x[0]; k++) {
                out_data[i * x[1] + j] += w_data[i * w[1] + k] * x_data[k * x[1] + j];
            }
        }
        }
        sum--;
        while (sum--) {
            w_data = w_data + w_warp;
            x_data = x_data + x_warp;
            out_data = out_data + out_warp;
            for (int i = 0; i < w[0]; i++) {
                for (int j = 0; j < x[1]; j++) {
                    out_data[i * x[1] + j] = 0;
                    for (int k = 0; k < x[0]; k++) {
                        out_data[i * x[1] + j] += w_data[i * w[1] + k] * x_data[k * x[1] + j];
                    }
                }
            }
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

template<class T>
void tensor_dot(const Tensor<T> *A, const Tensor<T> *B, Tensor<T> *output) {
    if (A->getSum() != B->getSum() || A->getDimension() != B->getDimension())
        throw ("mat dot input error and program terminated in " + A->getName() + " and " + B->getName() + "\n");
    int sum = B->getSum();
    T *A_data = A->data;
    T *B_data = B->data;
    T *out_data = output->data;
    try {
        out_data[0] = A_data[0] * B_data[0];
        sum--;
        while (sum--) {
            A_data++;
            B_data++;
            out_data++;
            out_data[0] = A_data[0] * B_data[0];
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

//Tensor<T> 后面加*就是加了地址，可以取出运算结果。
template<class T>
void tensor_add(const Tensor<T> *A, const Tensor<T> *B, Tensor<T> *output) {
    if (A->getSum() != B->getSum() || A->getDimension() != B->getDimension())
        throw ("mat dot input error and program terminated in " + A->getName() + " and " + B->getName() + "\n");
    int sum = B->getSum();
    T *A_data = A->data;
    T *B_data = B->data;
    T *out_data = output->data;
    try {
        out_data[0] = A_data[0] + B_data[0];
        sum--;
        while (sum--) {
            A_data++;
            B_data++;
            out_data++;
            out_data[0] = A_data[0] + B_data[0];
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

template<class T>
void tensor_minus(const Tensor<T> *A, const Tensor<T> *B, Tensor<T> *output) {
    if (A->getSum() != B->getSum() || A->getDimension() != B->getDimension())
        throw ("mat dot input error and program terminated in " + A->getName() + " and " + B->getName() + "\n");
    int sum = B->getSum();
    T *A_data = A->data;
    T *B_data = B->data;
    T *out_data = output->data;
    try {
        out_data[0] = A_data[0] - B_data[0];
        sum--;
        while (sum--) {
            A_data++;
            B_data++;
            out_data++;
            out_data[0] = A_data[0] - B_data[0];
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

template<class T>
void tensor_dot(const T b, const Tensor<T> *X, Tensor<T> *output) {
    int sum = X->getSum();
    T *x_data = X->data;
    T *out_data = output->data;
    try {
        out_data[0] = b * x_data[0];
        sum--;
        while (sum--) {
            x_data++;
            out_data++;
            out_data[0] = b * x_data[0];
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

template<class T>
void tensor_add(const T b, const Tensor<T> *X, Tensor<T> *output) {
    int sum = X->getSum();
    T *x_data = X->data;
    T *out_data = output->data;
    try {
        out_data[0] = b + x_data[0];
        sum--;
        while (sum--) {
            x_data++;
            out_data++;
            out_data[0] = b + x_data[0];
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

template<class T>
void tensor_minus(const T b, const Tensor<T> *X, Tensor<T> *output) {
    int sum = X->getSum();
    T *x_data = X->data;
    T *out_data = output->data;
    try {
        out_data[0] = x_data[0] - b;
        sum--;
        while (sum--) {
            x_data++;
            out_data++;
            out_data[0] = x_data[0] - b;
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

template <class T>
void tensor_transpose(const Tensor<T> *input, Tensor<T> *output){
    auto dim = input->getDimension();
    int trans[2];
    trans[0] = dim[dim.size() - 2];
    trans[1] = dim[dim.size() - 1];
    int warp = trans[0] * trans[1];
    int sum = input->getSum() / warp;
    T *input_data = input->data;
    T *out_data = output->data;
    try {
        for (int i = 0; i < trans[0]; i++) {
            for (int j = 0; j < trans[1]; j++) {
                out_data[j * trans[0] + i] = input_data[i * trans[1] +j];
            }
        }
        sum--;
        while (sum--) {
            input_data = input_data + warp;
            out_data = out_data + warp;
            for (int i = 0; i < trans[0]; i++) {
                for (int j = 0; j < trans[1]; j++) {
                    out_data[j * trans[0] + i] = input_data[i * trans[1] +j];
                }
            }
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

#endif //AUTOML_TENSOR_OP_HPP
