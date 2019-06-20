//
//#ifndef FUNC_DECLARE
//#define FUNC_DECLARE
//
//// template for the definition of inputs of function
//#define OP_PARA(TYPE) std::vector<TYPE *> &inputs, TYPE *output, int dim_m, int dim_k, int dim_n, int N
//#define OP_PARA_IN(TYPE) inputs, output, dim_m, dim_k, dim_n, N
///*********************************fixed area *****************************************/
////general condition
//#define DECLARE_FUNC(NAME, TYPE, P, P_IN)      \
//    extern "C" void func##NAME##TYPE(P(TYPE)); \
//    template<>                                 \
//    void NAME<TYPE>(P(TYPE)){                  \
//        func##NAME##TYPE(P_IN(TYPE));          \
//    }
//
//#define DECLARE_TEMPLATE(NAME, P, P_IN)        \
//    template <typename T>                      \
//    void NAME(P(T)) = delete;                  \
//    DECLARE_FUNC(NAME, float, P, P_IN)         \
//    /*DECLARE_FUNC(NAME, short, P, P_IN)         \
//    DECLARE_FUNC(NAME, int, P, P_IN)           \
//    DECLARE_FUNC(NAME, double, P, P_IN)        \
//    DECLARE_FUNC(NAME, long, P, P_IN)       */   
///************************************* end *****************************************/
//
//DECLARE_TEMPLATE(mat_mul, OP_PARA, OP_PARA_IN)
//
//#endif
//
// 
