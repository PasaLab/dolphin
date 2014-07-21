/* 
 * File:   StackedAutoencoder.h
 * Author: labuser
 *
 * Created on September 16, 2013, 10:22 AM
 */

#ifndef STACKEDAUTOENCODER_H
#define	STACKEDAUTOENCODER_H
#include "Autoencoder.h"
#include "SoftMax.h"

class StackedAutoencoder {
public:
    StackedAutoencoder();
    StackedAutoencoder(int* layerInfo, int length, int inputSize, float sparsityParam, float lambda, float beta, int numClasses);
    StackedAutoencoder(const StackedAutoencoder& orig);
    virtual ~StackedAutoencoder();
    void computeCostAndGradient(float* data, int col, float& cost, float** &gradient, float* groundTruth);
    void stackedAutoencoderFineTune(float* data, int col, int* labels, int maxIter, float precision, float learningRate);
    void updateParameter(float** gradient, float learningRate);
    void StackedAutoencoderTrain(float* data, int col, int maxIter, int* labels, float precision, float learningRate);
    int* predictLabels(float* data, int col);
    float getAccuracy(int* predictLables, int* labels, int col);
    void printParameters();
    //store the number of nodes of each layer,does not include the input layer
    int* layerInfo;
    int numLayers;
    int numClasses;
    //the dimension of the input data
    int inputSize;
    //desired average activation of hidden units
    Autoencoder** autoencoders;
    SoftMax* softMax;
    float sparsityParam;
    float lambda; //weight decay parameter
    float beta; //weight of sparsity penalty term
    int length;
};

#endif	/* STACKEDAUTOENCODER_H */

