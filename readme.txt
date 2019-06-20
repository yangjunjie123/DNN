1.  /cuda1/matrix_op是一个矩阵乘法，用了c++模板的cuda程序和普通cpu计算做对比，
计算时间大概是十倍的差距。

2.tensor文件夹是一个基于c指针内存作为tensor的，实现了全连接层分类minst。

3.dnn文件夹是一个基于c++的vector作为tensor的，实现全连接层分类minst。
