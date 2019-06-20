//
// Created by shuai on 3/22/2019.
//

#ifndef AUTOML_LSTM_HPP
#define AUTOML_LSTM_HPP

#include <vector>
#include <random>

#include "../Tensor/Tensor_op.hpp"



template<class T>
class RandomGenerator {
protected:
    default_random_engine e;
public:
    explicit RandomGenerator(unsigned int seed) {
        e.seed(seed);
    };

    inline int getNum() {
        return e();
    }

    void XavierInitializer(vector<T> &input, int dimension, int next_dimension) {
        double x = sqrt(6.0 / (dimension + next_dimension));
        uniform_real_distribution<double> u(-x, x);
        int length = (int) input.size();
        for (int i = 0; i < length; i++)
            input[i] = u(this->e);
    }

    void XavierInitializer(T *input, int dimension1, int dimension2) {
        double x = sqrt(6.0 / (dimension1 + dimension2));
        uniform_real_distribution<double> u(-x, x);
        for (int i = 0; i < dimension1 * dimension2; i++)
            input[i] = u(this->e);
		//printf("%f,T *input\n", input[1]);
    }

    void XavierInitializer(T **input, int dimension1, int dimension2) {
        for (int i = 0; i < dimension1; i++) {
            for (int j = 0; j < dimension2; j++) {
                input[i][j] = u(this->e);
            }
        }
    }

};

template<class T>
class temp_utils {
public:
    void Matmul(T **w, T **input, T **output, int dim1, int dim2, int dim3) {
        for (int i = 0; i < dim1; i++) {
            for (int k = 0; k < dim3; k++) {
                output[i][k] = 0;
                for (int j = 0; j < dim2; j++) {
                    output[i][k] += w[i][j] * input[j][k];
                }
            }
        }
    }

    void MadAdd(T **w, T **input, T **output, int dim1, int dim2) {
        for (int i = 0; i < dim1; i++) {
            for (int j = 0; j < dim2; j++) {
                output[i][j] = w[i][j] + input[i][j];
            }
        }
    }
};

template<class T>
static void sigmoid(Tensor<T> *input, Tensor<T> *output) {
    T *inputs = input->data;
    T *outputs = output->data;
    int sum = input->getSum() - 1;
    try {
        outputs[0] = 1 / (1 + exp(-inputs[0]));
        while (sum--) {
            outputs++;
            inputs++;
            outputs[0] = 1 / (1 + exp(-inputs[0]));
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

template<class T>
static void tanh(Tensor<T> *input, Tensor<T> *output) {
    T *inputs = input->data;
    T *outputs = output->data;
    int sum = input->getSum() - 1;
    try {
        outputs[0] = 2 / (1 + exp(-2 * inputs[0])) - 1;
        while (sum--) {
            outputs++;
            inputs++;
            outputs[0] = 2 / (1 + exp(-2 * inputs[0])) - 1;
        }
    } catch (exception &e) {
        cout << "Standard exception: " << e.what() << endl;
    };
}

template<class T>
void outputs(Tensor<T> *in) {
    int sum_count = in->getSum();
    for (int i = 0; i < sum_count; i++) {
        cout << in->data[i] << " ";
    }
    cout << endl;
}

template<class T>
class LSTMOP {
protected:
    int input_dimension, hidden_dimension, output_dimension, batch_size;
    vector<Tensor<T> *> gate[4], C, H, output;
    Tensor<T> *W[4], *U[4];
    T B[4], part_B[4];
    Tensor<T> *V, *C_zero, *H_zero;
    //vector<T *> f, i, t, o, C, H, output, lose_t, delta_h_t, delta_C_t;
    //T **Wf, Wi, Wt, Wo, Uf, Ui, Ut, Uo, V, C_zero, H_zero;
    T c, part_c;

    int back_pos;
public:
    LSTMOP() = delete;

    LSTMOP(int batch_size, int input_dimension, int hidden_dimension, int output_dimension) : input_dimension(
            input_dimension), hidden_dimension(hidden_dimension), output_dimension(output_dimension),
                                                                                              batch_size(batch_size) {
        auto generator = new RandomGenerator<T>(1);
        int temp1[2] = {hidden_dimension, hidden_dimension};
        for (auto &w : W) {
            w = new Tensor<T>(temp1, 2);
            generator->XavierInitializer(w->data, hidden_dimension, hidden_dimension);
        }
        /*
        Wf = new int *[hidden_dimension];
        Wi = new int *[hidden_dimension];
        Wt = new int *[hidden_dimension];
        Wo = new int *[hidden_dimension];
        for (int i = 0; i < hidden_dimension; i++) {
            Wf[i] = new int[hidden_dimension];
            Wi[i] = new int[hidden_dimension];
            Wt[i] = new int[hidden_dimension];
            Wo[i] = new int[hidden_dimension];
        } */
        int temp2[2] = {input_dimension, hidden_dimension};

        for (auto &u: U) {
            u = new Tensor<T>(temp2, 2);
            generator->XavierInitializer(u->data, input_dimension, hidden_dimension);
        }

        int temp3[2] = {hidden_dimension, output_dimension};

        V = new Tensor<T>(temp3, 2);

        int temp4[2] = {batch_size, hidden_dimension};
        C_zero = new Tensor<T>(temp4, 2);
        H_zero = new Tensor<T>(temp4, 2);
        /*
        Uf = new int *[hidden_dimension];
        Ui = new int *[hidden_dimension];
        Ua = new int *[hidden_dimension];
        Uo = new int *[hidden_dimension];
        V = new int *[output_dimension];
        C_zero = new int *[hidden_dimension];
        H_zero = new int *[hidden_dimension];
        for (int i = 0; i < hidden_dimension; i++) {
            Uf[i] = new int[input_dimension];
            Ui[i] = new int[input_dimension];
            Ua[i] = new int[input_dimension];
            Uo[i] = new int[input_dimension];
            V[i] = new int[hidden_dimension];
            C_zero[i] = new int[batch_size];
            H_zero[i] = new int[batch_size];
        }
         */
        generator->XavierInitializer(V->data, output_dimension, hidden_dimension);
        generator->XavierInitializer(C_zero->data, batch_size, hidden_dimension);
        generator->XavierInitializer(H_zero->data, batch_size, hidden_dimension);
        //C_zero = *(new int[hidden_dimension]);
        //H_zero = *(new int[hidden_dimension]);

        for (auto &b: B)
            b = 0;
        back_pos = 0;
        c = 0;

        delete generator;
    }

    ~LSTMOP(){
        delete[] W;
        delete[] U;
        delete V;
        delete C_zero;
        delete H_zero;
        for(int j = 0; j < C.size(); j++)
            delete C[j];
        for(int j = 0; j < H.size(); j++)
            delete H[j];
        for(int j = 0; j < output.size(); j++)
            delete output[j];
        for(int i = 0 ; i < 4; i++){
            for(int j = 0; j < gate[i].size(); j++)
                delete gate[i][j];
        }
    }

    void forward_propagation(vector<Tensor<T> *> inputs) {
        int temp1[2] = {batch_size, hidden_dimension};
        int temp2[2] = {batch_size, output_dimension};
        int temp3[2] = {batch_size, output_dimension};
        int input_size = (int) inputs.size();
        //init
        Tensor<T> W_h[4] = {Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2)};
        Tensor<T> U_x[4] = {Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2)};
        Tensor<T> S[4] = {Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2)};
        Tensor<T> SU[4] = {Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2)};
        Tensor<T> C_f(temp1, 2), i_a(temp1, 2), tanh_C(temp1, 2), V_h(temp2, 2), V_h_c(temp2, 2);

        for (int i = 0; i < input_size; i++) {
            back_pos++;

            if (i >= gate[0].size()) {
                for (auto &g : gate)
                    g.push_back(new Tensor<T>(temp1, 2));
                this->C.push_back(new Tensor<T>(temp1, 2));
                this->H.push_back(new Tensor<T>(temp1, 2));
                this->output.push_back(new Tensor<T>(temp3, 2));
            }

            auto *c_t = this->C_zero;
            auto *h = this->H_zero;
            if (i > 0) {
                c_t = this->C[i - 1];
                h = this->H[i - 1];
            }

            for (int j = 0; j < 4; j++) {
                tensor_matmul<T>(inputs[i], U[j], &U_x[j]);
                /*outputs(inputs[i]);
                outputs(U[j]);
                outputs(U_x[j]); */
                tensor_matmul<T>(h, W[j], &W_h[j]);
                tensor_add<T>(&U_x[j], &W_h[j], &S[j]);
                tensor_add<T>(B[j], &S[j], &SU[j]);
                sigmoid<T>(&SU[j], gate[j][i]);
            }
            tensor_dot<T>(c_t, gate[0][i], &C_f);
            tensor_dot<T>(gate[1][i], gate[2][i], &i_a);
            tensor_add<T>(&C_f, &i_a, C[i]);
            tanh<T>(C[i], &tanh_C);
            tensor_dot<T>(gate[3][i], &tanh_C, H[i]);
            tensor_matmul<T>(H[i], this->V, &V_h);
            tensor_add<T>(this->c, &V_h, &V_h_c);
            sigmoid<T>(&V_h_c, output[i]);
        }
    }

    void back_propagation(vector<Tensor<T> *> inputs, Tensor<T> *label) {
        int temp1[2] = {batch_size, hidden_dimension};
        int temp2[2] = {batch_size, output_dimension};
        int temp3[2] = {output_dimension, hidden_dimension};
        int temp4[2] = {hidden_dimension, batch_size};
        int temp5[2] = {hidden_dimension, input_dimension};
        int temp6[2] = {input_dimension, batch_size};
        int temp7[2] = {input_dimension, hidden_dimension};
        int temp8[2] = {hidden_dimension, hidden_dimension};
        Tensor<T> W_h[4] = {Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2)};
        Tensor<T> U_x[4] = {Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2)};
        Tensor<T> S[4] = {Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2)};
        Tensor<T> SU[4] = {Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2), Tensor<T>(temp1, 2)};
        Tensor<T> C_f(temp1, 2), i_a(temp1, 2), tanh_C(temp1, 2), V_h(temp2, 2);

        int sum_out = label->getSum();
        int sum_v = V->getSum();
        /*------------ last input back_propagation --------------------------*/
        part_c = 0;
        //Y =  sigmoid(Vh + c), to get part_V, part_h, part_c
        //part_c

        T temp_count = 0;
        for (int i = 0; i < sum_out; i++) {
            output[back_pos - 1]->error[i] = (output[back_pos - 1])->data[i] - label->data[i];
            if(output[back_pos - 1]->error[i] *  output[back_pos - 1]->error[i]  < 0.25)
                temp_count+= 1;
            //cout << "real:" << label ->data[i] << " prediction:"<< output[back_pos - 1]->data[i] << endl;
            V_h.error[i] = output[back_pos - 1]->error[i] * output[back_pos - 1]->data[i] *
                            (1 - output[back_pos - 1]->data[i]);
            part_c += V_h.error[i];
        }
        T temp = 0;
        for(int i = 0; i<sum_out; i++)
            temp += output[back_pos - 1]->error[i] * output[back_pos - 1]->error[i];
        cout<<"accuracy:"<< temp_count / sum_out << "; loss:"<<temp/ sum_out << endl;
        //part_h
        Tensor<T> V_T(temp3, 2);
        tensor_transpose(V, &V_T);
        tensor_matmul(&V_h, &V_T, H[back_pos - 1], true, false, true);

        //part_V
        Tensor<T> H_T(temp4, 2);
        tensor_transpose(H[back_pos - 1], &H_T);
        tensor_matmul(&H_T, &V_h, V, false, true, true);

        //part_C
        tanh<T>(C[back_pos - 1], &tanh_C);
        for (int i = 0; i < C[back_pos - 1]->getSum(); i++) {
            C[back_pos - 1]->error[i] = H[back_pos - 1]->error[i] * gate[3].at(back_pos - 1)->data[i] *
                                        (1 - tanh_C.data[i] * tanh_C.data[i]);
        }


        for (auto &b : part_B) b = 0;
        for (auto &w: W) {
            int count = w->getSum();
            for (int i = 0; i < count; i++)
                w->error[i] = 0;
        }
        for (auto &w: U) {
            int count = w->getSum();
            for (int i = 0; i < count; i++)
                w->error[i] = 0;
        }

        /* ----------- general part for back_propagation --------------------*/
        //part V because is a relation N to 1, so only count one time for partial V
        Tensor<T> h_T(temp4, 2);
        Tensor<T>  x_T (temp6, 2);
        tensor_transpose(H[back_pos - 1], &h_T);
        Tensor<T> partCt_Ct_1_GATEt_(temp1, 2);
        Tensor<T> part_H_t_1_to_H_t(temp1, 2);
        Tensor<T> part_partial_H(temp1, 2);
        Tensor<T> w_T(temp8, 2);
        Tensor<T> deltaC(temp1, 2);
        Tensor<T> part_W(temp8, 2);
        Tensor<T> part_U(temp7, 2);
        while (back_pos--) {
            Tensor<T> *true_h = nullptr, *true_c = nullptr;
            if(back_pos == 0){
                true_h = H_zero;
                true_c = C_zero;
            }
            else {
                true_h = H[back_pos - 1];
                true_c = C[back_pos - 1];
            }

            tensor_transpose(true_h, &h_T);
            tensor_transpose(inputs[back_pos], &x_T);
            tanh<T>(C[back_pos], &tanh_C);
            for (int i = 0; i < deltaC.getSum(); i++) {
                deltaC.error[i] = gate[3][back_pos]->data[i] * (1 - tanh_C.data[i] * tanh_C.data[i]);
            }


            for (int i = 0; i < true_h->getSum(); i++)
                true_h->error[i] = 0;
            //partial Wf Uf bf
            for (int i = 0; i < partCt_Ct_1_GATEt_.getSum(); i++) {
                partCt_Ct_1_GATEt_.data[i] =
                        C[back_pos]->error[i] * true_c->data[i] * gate[0][back_pos]->data[i] *
                        (1 - gate[0][back_pos]->data[i]);
                part_B[0] += partCt_Ct_1_GATEt_.data[i];
            }

            tensor_matmul(&h_T, &partCt_Ct_1_GATEt_, &part_W);
            tensor_matmul(&x_T, &partCt_Ct_1_GATEt_, &part_U);
            for(int i = 0; i < W[0]->getSum(); i++)
                W[0]->error[i] += part_W.data[i];
            for(int i = 0; i < U[0]->getSum(); i++)
                U[0]->error[i] += part_U.data[i];

            //partial Wi Ui bi
            for (int i = 0; i < partCt_Ct_1_GATEt_.getSum(); i++) {
                partCt_Ct_1_GATEt_.data[i] = C[back_pos]->error[i] * gate[2][back_pos]->data[i] * gate[1][back_pos]->data[i] *
                                              (1 - gate[1][back_pos]->data[i]);
                part_B[1] += partCt_Ct_1_GATEt_.data[i];
               // cout <<  partCt_Ct_1_GATEt_.data[i] << " ";
            }
            //cout <<endl;
            tensor_matmul(&h_T, &partCt_Ct_1_GATEt_, &part_W);
            tensor_matmul(&x_T, &partCt_Ct_1_GATEt_, &part_U);
            for(int i = 0; i < W[1]->getSum(); i++)
                W[1]->error[i] += part_W.data[i];
            for(int i = 0; i < U[1]->getSum(); i++)
                U[1]->error[i] += part_U.data[i];

            //partial Wa Ua ba
            for (int i = 0; i < partCt_Ct_1_GATEt_.getSum(); i++) {
                partCt_Ct_1_GATEt_.data[i] = C[back_pos]->error[i] * gate[1][back_pos]->data[i] * gate[2][back_pos]->data[i] *
                                              (1 - gate[2][back_pos]->data[i]);
                part_B[2] += partCt_Ct_1_GATEt_.data[i];
                //cout <<  partCt_Ct_1_GATEt_.data[i] << " ";
            }
            //cout <<endl;
            tensor_matmul(&h_T, &partCt_Ct_1_GATEt_, &part_W);
            tensor_matmul(&x_T, &partCt_Ct_1_GATEt_, &part_U);
            for(int i = 0; i < W[2]->getSum(); i++)
                W[2]->error[i] += part_W.data[i];
            for(int i = 0; i < U[2]->getSum(); i++)
                U[2]->error[i] += part_U.data[i];

            //partial Wo Uo bo
            for (int i = 0; i < partCt_Ct_1_GATEt_.getSum(); i++) {
                partCt_Ct_1_GATEt_.data[i] = H[back_pos]->error[i] * tanh_C.data[i] * gate[3][back_pos]->data[i] *
                                              (1 - gate[3][back_pos]->data[i]);
                part_B[3] += partCt_Ct_1_GATEt_.data[i];
                //cout <<  partCt_Ct_1_GATEt_.data[i] << " ";
            }
            //cout <<endl;
            tensor_matmul(&h_T, &partCt_Ct_1_GATEt_, &part_W);
            tensor_matmul(&x_T, &partCt_Ct_1_GATEt_, &part_U);
            for(int i = 0; i < W[3]->getSum(); i++)
                W[3]->error[i] += part_W.data[i];
            for(int i = 0; i < U[3]->getSum(); i++)
                U[3]->error[i] += part_U.data[i];

            if(back_pos > 0) {
                //part h(t+1) - > h(t)
                for (int i = 0; i < partCt_Ct_1_GATEt_.getSum(); i++) {
                    part_H_t_1_to_H_t.data[i] =
                            deltaC.error[i] * true_c->data[i] * gate[0][back_pos]->data[i] *
                            (1 - gate[0][back_pos]->data[i]);
                }
                tensor_transpose(W[0], &w_T);
                tensor_matmul(&part_H_t_1_to_H_t, &w_T, &part_partial_H);
                for (int i = 0; i < part_partial_H.getSum(); i++)
                    true_h->error[i] += part_partial_H.data[i];

                for (int i = 0; i < partCt_Ct_1_GATEt_.getSum(); i++) {
                    partCt_Ct_1_GATEt_.data[i] =
                            deltaC.error[i] * gate[2][back_pos]->data[i] * gate[1][back_pos]->data[i] *
                            (1 - gate[1][back_pos]->data[i]);
                }
                tensor_transpose(W[1], &w_T);
                tensor_matmul(&part_H_t_1_to_H_t, &w_T, &part_partial_H);
                for (int i = 0; i < part_partial_H.getSum(); i++)
                    true_h->error[i] += part_partial_H.data[i];

                for (int i = 0; i < partCt_Ct_1_GATEt_.getSum(); i++) {
                    partCt_Ct_1_GATEt_.data[i] =
                            deltaC.error[i] * gate[1][back_pos]->data[i] * gate[2][back_pos]->data[i] *
                            (1 - gate[2][back_pos]->data[i]);
                }
                tensor_transpose(W[2], &w_T);
                tensor_matmul(&part_H_t_1_to_H_t, &w_T, &part_partial_H);
                for (int i = 0; i < part_partial_H.getSum(); i++)
                    true_h->error[i] += part_partial_H.data[i];

                for (int i = 0; i < partCt_Ct_1_GATEt_.getSum(); i++) {
                    partCt_Ct_1_GATEt_.data[i] = tanh_C.data[i] * gate[3][back_pos]->data[i] *
                                                 (1 - gate[3][back_pos]->data[i]);
                }
                tensor_transpose(W[3], &w_T);
                tensor_matmul(&part_H_t_1_to_H_t, &w_T, &part_partial_H);
                for (int i = 0; i < part_partial_H.getSum(); i++)
                    true_h->error[i] += part_partial_H.data[i];

                for (int i = 0; i < true_h->getSum(); i++)
                    true_h->error[i] = true_h->error[i] * H[back_pos]->error[i];

                // part Ct+1 to Ct

                tanh<T>(true_c, &tanh_C);
                for (int i = 0; i < true_c->getSum(); i++)
                    true_c->error[i] = C[back_pos]->error[i] * gate[0][back_pos]->data[i] +
                                                true_h->error[i] * gate[3][back_pos - 1]->data[i] *
                                                (1 - tanh_C.data[i] * tanh_C.data[i]);
            }
        }
        T rate = 1;
        for(int i = 0; i < 4; i++){
            for(int j = 0 ; j < W[i]->getSum(); j++)
                W[i]->data[j] -= rate * W[i]->error[j];
            for(int j = 0 ; j < U[i]->getSum(); j++)
                U[i]->data[j] -= rate * U[i]->error[j];
            B[i] -= part_B[i] * rate;
        }
        for(int j = 0 ; j < V->getSum(); j++)
            V->data[j] -= rate * V->error[j];
        c -= part_c * rate;
    }


    vector<Tensor<T>> GetOutput() {
        return this->output;
    }

public:
    void update_w(Tensor<T> **W) {
        this->W = W;
    }

    void update_U(Tensor<T> **U) {
        this->U = U;
    }

    void update_b(T *b) {
        this->B = b;
    }

    void update_V(Tensor<T> *V) {
        this->V = V;
    }

    void update_C0(T c) {
        this->c = c;
    }

    void update_C0(Tensor<T> *C0) {
        this->C_zero = C0;
    }

    void update_H0(Tensor<T> *H0) {
        this->H_zero = H0;
    }

};

#endif //AUTOML_LSTM_HPP
