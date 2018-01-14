
#ifndef __Image_Manip__
#define __Image_Manip__

#include <iostream>
#include <cmath>
#include <Eigen/Dense>

using Eigen::MatrixXd;


class Image_Manip{

    std::default_random_engine generator;
    std::normal_distribution<double> norm(0, 0.3);


public:
    //randomly subsamples a matrix at regular intervals
    int sub_sampler( int src[], int out_dim);
    
     //Provides a normal dist random value between 0-1 mean 0.5
    double norm_resampler();


}

#endif