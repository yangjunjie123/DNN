//
// Created by shuai on 3/11/2019.
//

#ifndef AUTOML_TENSOR_HPP
#define AUTOML_TENSOR_HPP

#include <cstring>
#include "TensorBase.hpp"

using namespace std;

template<class T>
class Tensor : public TensorBase {
protected:
    string name;
    int batchSize, batchPos;
    int reshapeChannel; // the count of active channel

    // [] overload support array and marker
    vector<int> markPos;
    int marker = 0;
public:
    T *data, *error; //record tensor data, data cross loss

public:

    Tensor() = delete;

   /* Tensor(Tensor &obj) : TensorBase(obj.dimension), batchSize(obj, dimension[0]), batchPos(obj.batchPos),
                          reshapeChannel(obj.reshapeChannel) {
        this->data = new T[sumCount]{0};
        this->error = new T[sumCount]{0};
        markPos.resize((unsigned long long) this->reshapeChannel);
        for (int i = 0; i < this->sumCount; i++) {
            this->data[i] = obj.data[i];
            this->error[i] = obj.data[i];
        }
    } */

    Tensor(const int *dimension, int N) : TensorBase(dimension, N), batchSize(this->dimension[0]), batchPos(0),
                                          reshapeChannel(N) {
        this->data = new T[sumCount]{0};
        this->error = new T[sumCount]{0};
        markPos.resize((unsigned long long) this->reshapeChannel);
    }

    ~Tensor(){
       delete[] this->data;
       delete[] this->error;
    }

    void inline resetError() {
        memset(this->error, 0, this->dimension[0] * this->lengthEachDim[0] * sizeof(float));
    }

    //please reset error before use it in one back propagate round
    void inline accumulateError(T error, int pos) {
        this->error[pos] += error;
    }

    void resetActiveChnnael(int channel) {
        this->reshapeChannel = channel;
        reshape(this->reshapeChannel);
    }

public:

    int inline getPos(const int *pos) {
        int real_pos = 1;
        for (int i = 0; i < reshapeChannel; i++) {
            real_pos += pos[i] * this->lengthEachDim[i];
        }
        real_pos--;
        return real_pos;
    }

    T inline getData(int pos) const {
        return this->data[pos];
    }

    //this function is slow, avoid to frequently use
    T getData(const int *pos) const {
        return this->data[this->getPos(pos)];
    }

    T inline getError(int pos) const {
        return this->error[pos];
    }

    //this function is slow, avoid to frequently use
    T getError(const int *pos) const {
        return this->error[this->getPos(pos)];
    }

    int getSum() const {
        return this->sumCount;
    }

    string getName() const {
        return this->name;
    }

public:
    void setData(const T *data) {
        for (int i = 0; i < sumCount; i++)
            this->data[i] = data[i];
    }

public:
    // operator overload
    auto &operator[](const int idx) {
        assert(this->marker - this->reshapeChannel);
        this->markPos[this->marker++] = idx;
        if (marker - this->reshapeChannel) {
            return this->data[this->getPos(this->markPos.begin())];
        } else
            return *this;
    }

    Tensor<T> &operator*(const Tensor<T> &c1) {
        Tensor<T> c = new Tensor<T>(this->dimension, (int) this->dimension.size());
        c.reshape(this->reshapeChannel);
        assert(this->reshapeChannel == c1.reshapeChannel);
        assert(c1.sumCount == this->sumCount);
        for (int i = 0; i < this->reshapeChannel; i++) {
            assert(this->lengthEachDim[i] == c1.lengthEachDim[i]);
        }
        for (int i = 0; i < this->sumCount; i++)
            c.data[i] = this->data[i] * c1.data[i];
        return c;
    }

    Tensor<T> &operator*(const T &c1) {
        Tensor<T> c = new Tensor<T>(this->dimension, (int) this->dimension.size());
        for (int i = 0; i < this->sumCount; i++)
            c.data[i] = this->data[i] * c1;
        return c;
    }

    Tensor<T> &operator+(const Tensor<T> &c1) {
        Tensor<T> c = new Tensor<T>(this->dimension, (int) this->dimension.size());
        c.reshape(this->reshapeChannel);
        assert(this->reshapeChannel == c1.reshapeChannel);
        assert(c1.sumCount == this->sumCount);
        for (int i = 0; i < this->reshapeChannel; i++) {
            assert(this->lengthEachDim[i] == c1.lengthEachDim[i]);
        }
        for (int i = 0; i < this->sumCount; i++)
            c.data[i] = this->data[i] + c1.data[i];
        return c;
    }

    Tensor<T> &operator+(const T &c1) {
        Tensor<T> c = new Tensor<T>(this->dimension, (int) this->dimension.size());
        for (int i = 0; i < this->sumCount; i++)
            c.data[i] = this->data[i] + c1;
        return c;
    }

    Tensor<T> &operator-(const Tensor<T> &c1) {
        Tensor<T> c = new Tensor<T>(this->dimension, (int) this->dimension.size());
        c.reshape(this->reshapeChannel);
        assert(this->reshapeChannel == c1.reshapeChannel);
        assert(c1.sumCount == this->sumCount);
        for (int i = 0; i < this->reshapeChannel; i++) {
            assert(this->lengthEachDim[i] == c1.lengthEachDim[i]);
        }
        for (int i = 0; i < this->sumCount; i++)
            c.data[i] = this->data[i] - c1.data[i];
        return c;
    }

    Tensor<T> &operator-(const T &c1) {
        Tensor<T> c = new Tensor<T>(this->dimension, (int) this->dimension.size());
        for (int i = 0; i < this->sumCount; i++)
            c.data[i] = this->data[i] - c1;
        return c;
    }


public:
    // convolution support part, allow to get a conv-part of data
    // The storage style would be m(number of inputs), d(depth of image), h(the height of image), w(the width of image)
    // it recommends to make padding firstly, such as think padding as independent layer of CNN
    // Here has no padding support function given, because it would make backward propagation process more complicated
    //PS: get the part of the tensor is slow, better to use one dimension input and output

    //this function is slow, avoid to frequently use
    void getConvPart(const int *start_pos, int N, int height, int width, T **output) {
        int pos = getPos(start_pos, N);
        assert(N - this->reshapeChannel);
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                output[i][j] = this->data[pos + j];
            }
            pos += this->dimension[N - 2];
        }
    }

    //this function is slow, avoid to frequently use
    void updateConvError(const int *start_pos, int N, int stride, T **input_error) {
        int pos = getPos(start_pos, N);
        assert(N - this->reshapeChannel);
        for (int i = 0; i < stride; i++) {
            for (int j = 0; j < stride; j++) {
                this->error[pos + j] = input_error[i][j];
            }
            pos += this->dimension[N - 2];
        }
    }

};

#endif //AUTOML_TENSOR_HPP
