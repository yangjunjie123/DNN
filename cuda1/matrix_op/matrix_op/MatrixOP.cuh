/*
#ifndef MATRIX_OP_CUH
#define MATRIX_OP_CUH

#define OP_PARA(TYPE) std::vector<TYPE *> &inputs, TYPE *output, int dim_m, int dim_k, int dim_n, int N
#define OP_PARA_IN(TYPE) inputs, output, dim_m, dim_k, dim_n, N

extern template<T>
void mat_mul(OP_PARA(T));

//general condition
#define DECLARE_FUNC(NAME, TYPE, P, P_IN)      \
    extern "C" void func##NAME##TYPE(P(TYPE)){ \
        NAME<TYPE>(P_IN(TYPE));                \
    }
#define DECLARE_TEMPLATE(NAME, P, P_IN)        \
    DECLARE_FUNC(NAME, short, P, P_IN)         \
    DECLARE_FUNC(NAME, int, P, P_IN)           \
    DECLARE_FUNC(NAME, float, P, P_IN)         \
    DECLARE_FUNC(NAME, double, P, P_IN)        \
    DECLARE_FUNC(NAME, long, P, P_IN)          \

DECLARE_TEMPLATE(mat_mul, OP_PARA, OP_PARA_IN)

#endif
*/