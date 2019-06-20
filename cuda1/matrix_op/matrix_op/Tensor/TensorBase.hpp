//
// Created by shuai on 3/11/2019.
//

#ifndef AUTOML_TENSORBASE_HPP
#define AUTOML_TENSORBASE_HPP

#include <vector>
#include <algorithm>
#include <assert.h>

using namespace std;

class TensorBase {
protected:
    // The active dimension[i] = lengthEachDim[i - 1] / lengthEachDim[i]
    vector<int> dimension, real_dimension;
    vector<int> lengthEachDim;
    int sumCount;

    void reshape(int N) {
        assert(N - (int) this->dimension.size() >= 0 && N > 0);
        //reset the length of each dimension
        this->lengthEachDim[(int) this->lengthEachDim.size() - 1] = 1;
        for (int i = (int) this->lengthEachDim.size() - 2; i >= 0; i--)
            this->lengthEachDim[i] = this->lengthEachDim[i + 1] * dimension[i + 1];

        // set N - 2 dimension = (N - 1) * N  * (N + 1) ....
        // set N - 1 dimension as 1
        // and set the last places as 0
        for (int i = N - 1; i >= 0 && i < (int) this->dimension.size(); i++) {
            this->lengthEachDim[N - 2] *= this->lengthEachDim[i];
            this->lengthEachDim[i] = 0;
            this->real_dimension[i] = 0;
        }
        this->lengthEachDim[N - 1] = 1;
        this->real_dimension[0] = sumCount / this->lengthEachDim[0];
        for(int i = 1; i < N; i++)
            this->real_dimension[i] = this->lengthEachDim[i - 1] / this->lengthEachDim[i];

    }

    TensorBase() = delete;

    explicit TensorBase(vector<int> dimension) {
        const int *dim = dimension.data();
        int N = (int)dimension.size();
        //for any position [i][j][k]....., pos = i*lengthEachDim[0](led[0]) + j* lad[1] + k *led[2] + ....
        this->dimension = vector<int>(dim, dim + N);
        this->real_dimension = vector<int>(dim, dim + N);
        this->lengthEachDim.resize((unsigned long long) N);
        this->sumCount = 1;
        for_each(dimension.begin(), dimension.end(), [this](int a) {
            this->sumCount *= a;
        });
        reshape(N);
    }

    TensorBase(const int *dim, int N) {
        //for any position [i][j][k]....., pos = i*lengthEachDim[0](led[0]) + j* lad[1] + k *led[2] + ....
        this->dimension = vector<int>(dim, dim + N);
        this->real_dimension = vector<int>(dim, dim + N);
        this->lengthEachDim.resize((unsigned long long) N);
        this->sumCount = 1;
        for_each(dimension.begin(), dimension.end(), [this](int a) {
            this->sumCount *= a;
        });
        reshape(N);
    }

public:

    ~TensorBase() {
        this->dimension.clear();
        this->lengthEachDim.clear();
    }

    auto getDimension() const{
        return real_dimension;
    }
};


#endif //AUTOML_TENSORBASE_HPP
