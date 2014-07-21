/* 
 * File:   SoftMax.h
 * Author: labuser
 *
 * Created on September 18, 2013, 2:00 PM
 */

#ifndef SOFTMAX_H
#define	SOFTMAX_H

class SoftMax {
public:
    SoftMax();
    SoftMax(int inputSize, int numClasses, float lambda);
    SoftMax(const SoftMax& orig);
    virtual ~SoftMax();
    void computeCostAndGradient(float* data, int col, float& cost, float* &thetaGrad, float* groundTruth);
    float* feedForward(float* data, int col);
    void softMaxTrain(float* data, int col, int* labels, int maxIter,float learningRate, float precision);
    void setTheta(float* theta);
    void initialTheta(int numClasses, int inputSize);
    float normal_rand();
    void printParameters();
    
    float* theta; //the parameter to be optimized

    int inputSize; //the dimension of input data
    int numClasses; //number of classes
    float lambda;
};

#endif	/* SOFTMAX_H */

