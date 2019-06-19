
//  frame.hpp
//  automl
//
//  Created by Jack on 2019/2/24.
//  Copyright © 2019年 PA. All rights reserved.
//


#ifndef nn_frame_hpp
#define nn_frame_hpp

#include <stdio.h>
#include <vector>
#include <math.h>
#include "utils/nn_utils.hpp"

using namespace std;
using namespace pa_nn;


// point
struct StPoint{
    int x;
    int y;
};



extern void cnn_train_2();

#define DEBUG_FILE_NAME "/Users/jack/Projects/automl/automl/debug.txt"

#endif /* frame_hpp */
