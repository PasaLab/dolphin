/* 
 * File:   StackedAutoencoder.cpp
 * Author: labuser
 * 
 * Created on September 16, 2013, 10:22 AM
 */

#include "StackedAutoencoder.h"
#include "math.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "time.h"
#include "consts.h"
using namespace std;


StackedAutoencoder::StackedAutoencoder() {
}

StackedAutoencoder::StackedAutoencoder(int* layerInfo, int length, int inputSize, 
        float sparsityParam, float lambda, float beta, int numClasses){
    this->layerInfo = new int[length];
    for(int i = 0; i < length; i++){
        this->layerInfo[i] = layerInfo[i];
    }
    this->inputSize = inputSize;
    this->sparsityParam = sparsityParam;
    this->lambda = lambda;
    this->beta = beta; 
    this->numLayers = length;
    this->autoencoders = new Autoencoder * [length];
    autoencoders[0] = new Autoencoder(inputSize, layerInfo[0], sparsityParam, lambda, beta);
    for(int i = 1; i < length; i++){
        autoencoders[i] = new Autoencoder(layerInfo[i-1], layerInfo[i], sparsityParam, lambda, beta);
    }
    this->numClasses = numClasses;
    this->softMax = new SoftMax(layerInfo[length - 1], numClasses, lambda);
    this->length = length;
}

StackedAutoencoder::StackedAutoencoder(const StackedAutoencoder& orig) {
}

StackedAutoencoder::~StackedAutoencoder() {
    delete[] layerInfo;
    for(int i = 0; i < length;i++)
	delete autoencoders[i];
    delete[] this->autoencoders;
    delete this->softMax;
}

void StackedAutoencoder::computeCostAndGradient(float* data, int col, float& cost, float** &gradient, float* groundTruth){
   
    float** output = new float*[numLayers];
    output[0] = this->autoencoders[0]->feedForwardAutoencoder(data, col);
    for(int i = 1; i < numLayers; i++){
        output[i] = this->autoencoders[i]->feedForwardAutoencoder(output[i-1], col);
    }

    float* result = this->softMax->feedForward(output[numLayers-1], col);
   
    float costMatrix = 0.0;
    float thetaCost = 0.0;  
    float weightSquare = 0.0;

    #pragma omp sections
    {//sections begin
    #pragma omp section
    {
    #pragma omp parallel for reduction(+:costMatrix) num_threads(NUM_THREADS / 4)
    for(int i = 0; i < numClasses * col; i++){
        costMatrix += groundTruth[i] * log(result[i] + 10e-37);
    }
    }
 
    #pragma omp section
    {
    #pragma omp parallel for reduction(+:thetaCost) num_threads(NUM_THREADS / 4)
    for(int i = 0;i < numClasses * layerInfo[numLayers - 1]; i++){
        thetaCost += softMax->theta[i] * softMax->theta[i];
    }
    }

    #pragma omp section
    {  
    #pragma omp parallel for reduction(+:weightSquare) num_threads(NUM_THREADS / 4)
    for(int i = 0; i < layerInfo[0] * inputSize; i++){
        weightSquare += (this->autoencoders[0]->weightMatrix1[i]) * (this->autoencoders[0]->weightMatrix1[i]);
    }
    for(int i = 1; i < numLayers; i++){
        #pragma omp parallel for reduction(+:weightSquare) num_threads(NUM_THREADS / 4)
        for(int j = 0; j < layerInfo[i] * layerInfo[i-1]; j++){
            weightSquare += (this->autoencoders[i]->weightMatrix1[j]) * (this->autoencoders[i]->weightMatrix1[j]);
        }
    }
    }
    }//sections end

    cost = -1 * costMatrix / col + 0.5 * lambda * (thetaCost + weightSquare);
    
    float* tmpArray;
    malloc_float(tmpArray, numClasses, col);
    #pragma omp sections
    {
    #pragma omp section
    {
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < numClasses * col; i++){
        tmpArray[i] = groundTruth[i] - result[i];
    }
    }
    #pragma omp section
    {  
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < numClasses * layerInfo[numLayers-1]; i++){
        gradient[numLayers][i] = lambda * softMax->theta[i];
    }
    }
    }

    float** delta = new float*[numLayers];
    #pragma omp sections
    {//omp sections begin
    #pragma omp section
    {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		numClasses, layerInfo[numLayers - 1], col,
		-1.0 / col, tmpArray, col,
		output[numLayers - 1], col, 1.0, 
		gradient[numLayers], layerInfo[numLayers - 1]);
    }
    
    #pragma omp section
    {
    malloc_float(delta[numLayers - 1], layerInfo[numLayers-1], col);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		layerInfo[numLayers - 1], col, numClasses,
		-1.0, softMax->theta, layerInfo[numLayers-1],
		tmpArray, col, 0.0, 
		delta[numLayers - 1], col);
    #pragma omp parallel for num_threads(NUM_THREADS) 
    for(int i = 0; i < layerInfo[numLayers-1] * col; i++){
        delta[numLayers - 1][i] = delta[numLayers - 1][i] * output[numLayers - 1][i] * (1 - output[numLayers - 1][i]);
    }
    
  
    for(int l = numLayers-2;l >= 0; l--){
        malloc_float(delta[l], layerInfo[l], col);

        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		autoencoders[l+1]->visibleSize, col, layerInfo[l + 1],
		1.0, autoencoders[l+1]->weightMatrix1, autoencoders[l+1]->visibleSize,
		delta[l+1], col, 0.0, 
		delta[l], col);
        #pragma omp parallel for num_threads(NUM_THREADS) 
        for(int i = 0; i < layerInfo[l] * col; i++){
            delta[l][i] = delta[l][i] * output[l][i] * (1 - output[l][i]);
        }
    }
    }

    }//omp sections end  
    #pragma omp sections
    {
    #pragma omp section
    {
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i <layerInfo[0] * inputSize; i++){
        gradient[0][i] = lambda * autoencoders[0]->weightMatrix1[i];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		layerInfo[0], inputSize, col,
		1.0 / col, delta[0], col,
		data, col, 1.0, 
		gradient[0], inputSize);  
    }

    #pragma omp section
    {
    float* bOffSet = &(gradient[0][layerInfo[0] * inputSize]);
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < layerInfo[0]; i++){
	float tmp =.0;
        for(int j = 0; j < col; j++){
            tmp += delta[0][i*col+j];
        } 
        bOffSet[i] = tmp / col;
    }
    }
    }

    //compute the gradient of the rest of the layers
    for(int l = 1; l < numLayers; l++){
        #pragma omp sections
        {// pragma section begin
        #pragma omp section
        {//section1 begin
        #pragma omp parallel for num_threads(NUM_THREADS / 2)
        for(int i = 0; i <layerInfo[l] * layerInfo[l-1]; i++){
            gradient[l][i] = lambda * autoencoders[l]->weightMatrix1[i];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		    layerInfo[l], layerInfo[l - 1], col,
		    1.0/col, delta[l], col,
		    output[l - 1], col, 1.0, 
		    gradient[l], layerInfo[l - 1]);
        }//section1 end

        #pragma omp section
        {//section2 begin 
        float* bOffSet2 = &(gradient[l][layerInfo[l] * layerInfo[l-1]]);
        #pragma omp parallel for num_threads(NUM_THREADS / 2)
        for(int i = 0; i < layerInfo[l]; i++){
	    float tmp = 0.0;
            for(int j = 0; j < col; j++){
                tmp += delta[l][i * col + j];
            }
            bOffSet2[i] = tmp / col;
        }
        }//section2 end
        }//pragma sections end
    }

    mkl_free(result);
    for(int i = 0;i < numLayers; i++){
        mkl_free(output[i]);
        mkl_free(delta[i]);
    }
    mkl_free(tmpArray);
    delete []delta;
    delete []output;
}

void StackedAutoencoder::stackedAutoencoderFineTune(float* data, int col, int* labels, int maxIter, float precision, float learningRate){
    int iter = 0;
    float cost_old, cost_new, change;
    float** gradient = new float*[numLayers + 1];
    gradient[0] = (float*)mkl_malloc(sizeof(float) * (layerInfo[0] * inputSize + layerInfo[0]), 64);
    for(int i = 1; i < numLayers; i++){
        gradient[i] = (float*)mkl_malloc(sizeof(float) * (layerInfo[i] * layerInfo[i-1] + layerInfo[i]), 64);
    }
    gradient[numLayers] = (float*)mkl_malloc(sizeof(float) * (numClasses * layerInfo[numLayers - 1]), 64);

    float* groundTruth = new float[numClasses * col];
    memset(groundTruth, 0, sizeof(float) * numClasses * col);
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int i = 0; i < col; i++){
        groundTruth[(labels[i] - 1) * col + i] = 1;
    }

    cerr << "in fine tune" << endl;
    computeCostAndGradient(data, col, cost_old, gradient, groundTruth);
    cerr << "after compute gradient" << endl;
    //update parameter;
    updateParameter(gradient, learningRate);
    cerr << "after update parameter" <<endl;
    do{
        iter++;
        cout << "finetune iteration:" << iter <<endl;
        // compute gradient and update the gradient and change
        computeCostAndGradient(data, col, cost_new, gradient, groundTruth);
        updateParameter(gradient, learningRate);
        cout << cost_new << endl;
        change = cost_new - cost_old;
        cost_old = cost_new;
    }while(iter < maxIter);

    for(int i = 0; i <= numLayers; i++){
        mkl_free(gradient[i]);
    }
    delete[] gradient;
    delete[] groundTruth;
}

void StackedAutoencoder::updateParameter(float** gradient, float learningRate){
    #pragma omp sections
    {  
    #pragma omp section
    {
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < numLayers; i++){
        for(int j = 0; j < autoencoders[i]->hiddenSize * autoencoders[i]->visibleSize; j++){
            autoencoders[i]->weightMatrix1[j] -= learningRate * gradient[i][j];
        }
    }
    }
    #pragma omp section
    {
    #pragma omp parallel for num_threads(NUM_THREADS/2)
    for(int j = 0; j < softMax->numClasses * softMax->inputSize; j++){
        softMax->theta[j] -= learningRate * gradient[numLayers][j];
    }
    }
    }
}

void StackedAutoencoder::StackedAutoencoderTrain(float* data, int col, int maxIter, int* labels, float precision, float learningRate){
    cerr << "start sda train" << endl;
    this->autoencoders[0]->autoencoderTrain(data, col, maxIter, precision, learningRate);
    float* output[numLayers + 1];
    output[0] = this->autoencoders[0]->feedForwardAutoencoder(data, col);
    for(int i = 1; i < numLayers; i++){
        this->autoencoders[i]->autoencoderTrain(output[i - 1], col, maxIter, precision, learningRate);
        output[i]  = this->autoencoders[i]->feedForwardAutoencoder(output[i - 1], col);
        mkl_free(output[i - 1]);
    }
    cerr << "start softmax train at:" << clock() <<endl;
    this->softMax->softMaxTrain(output[numLayers - 1], col, labels, maxIter, learningRate, precision);
    cerr << "finish one softmax train" <<endl;
    mkl_free(output[numLayers - 1]);
    cerr << "finish sda train" <<endl;
}

int* StackedAutoencoder::predictLabels(float* data, int col){
    float* output[numLayers + 1];
    output[0] = this->autoencoders[0]->feedForwardAutoencoder(data, col);
    for(int i = 1; i < numLayers; i++){
        output[i] = this->autoencoders[i]->feedForwardAutoencoder(output[i - 1], col);
    }
    output[numLayers] = this->softMax->feedForward(output[numLayers - 1], col);
    
    int *labels = new int[col];
    float* max = new float[col];
    for(int i = 0; i < col; i++){
        max[i] = 0.0;
        labels[i] = 1;
    }
    for(int i = 0; i < softMax->numClasses; i++){
        for(int j = 0; j < col; j++){
            if(output[numLayers][i * col + j] >= max[j]){
                max[j] = output[numLayers][i * col + j];
                labels[j] = i + 1;
            }
        }
    }
    delete[] max;
    for(int i = 0; i <= numLayers; i++)
	mkl_free(output[i]);
    return labels;
}

float StackedAutoencoder::getAccuracy(int* predictLabels, int* labels, int col){
    int right = 0;
    for(int i = 0; i < col; i++){
        if(predictLabels[i] == labels[i])
            right++;
    }
    return (float)right / col;
}

void StackedAutoencoder::printParameters(){
    cout << "numLayers" << numLayers <<endl;
    cout << "numClasses" << numClasses <<endl;
    cout << "inputSize" << inputSize <<endl;
    cout << "sparsityParam" << sparsityParam <<endl;
    cout << "lambda" << lambda <<endl;
    cout << "beta" << beta <<endl;
    
    for(int i = 0; i < numLayers; i++){
        cout << "the " << i+1 << " autoencoder info:" <<endl;
        this->autoencoders[i]->printParameter();
    }
    
    cout << "the softmax info:" <<endl;
    softMax->printParameters();
}
