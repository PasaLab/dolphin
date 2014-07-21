/* 
 * File:   Autoencoder.h
 * Author: labuser
 *
 * Created on September 16, 2013, 10:57 AM
 */

#ifndef AUTOENCODER_H
#define	AUTOENCODER_H

class Autoencoder {
public:
    Autoencoder();
    Autoencoder(int visibleSize, int hiddenSize, float sparsityParam, float lambda, float beta);
    Autoencoder(const Autoencoder& orig);
    void computeCostAndGradient(float& cost, float* &gradientW1, float* &gradientW2, float* &gradientB1, float* &gradientB2, float* data, int col);
    void initialWeights(int hiddenSize, int visibleSize);
    void setWeights(float* w1, float* w2, float* b1, float* b2);
    float* feedForwardAutoencoder(float* data, int col);
    void autoencoderTrain(float* data, int col, int maxIter, float precision, float learningRate);
    void printParameter();
    float stepLength(int iteration, float f_new, float f_old, const float* &gradientW1, const float* 
    &gradientW2, const float* &gradientB1, const float* &gradientB2);
    int setWeightMatrix1(float* w);//set the weightMatrix1, return the size of array weightMatrix1
    
    virtual ~Autoencoder();

    int visibleSize; //number of input and output units,about data size->hiddenSize->hiddenSize,1000,100000,....
    int hiddenSize; //number of hidden units,about the same with visibleSize
    float* weightMatrix1; //its size is of hiddenSize*visibleSize
    float* weightMatrix2; //its size is of visibleSize*hiddenSize
    float* b1; //it is an array of size hiddenSize*1
    float* b2; //it is an array of size visibleSize*1
    float sparsityParam;
    float lambda;
    float beta;
};

#endif	/* AUTOENCODER_H */

