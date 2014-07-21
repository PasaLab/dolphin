/* 
 * File:   Autoencoder.cpp
 * Author: labuser
 * 
 * Created on September 16, 2013, 10:57 AM
 */

#include <iostream>
#include "Autoencoder.h"
#include "math.h"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include "matrixm.h"
#include "consts.h"
#include <mkl.h>
#include <time.h>
using namespace std;


Autoencoder::Autoencoder() {
}

Autoencoder::Autoencoder(int visibleSize, int hiddenSize, float sparsityParam, float lambda, float beta){
    this->hiddenSize = hiddenSize;
    this->visibleSize = visibleSize;
    this->sparsityParam = sparsityParam;
    this->lambda = lambda;
    this->beta = beta;
    initialWeights(hiddenSize,visibleSize);
}

Autoencoder::Autoencoder(const Autoencoder& orig) {
}

Autoencoder::~Autoencoder() {
    mkl_free(this->weightMatrix1);
    mkl_free(this->weightMatrix2);
    mkl_free(this->b1);
    mkl_free(this->b2);
}

int Autoencoder::setWeightMatrix1(float* w)
{
  memcpy(this->weightMatrix1, w, sizeof(float)*hiddenSize*visibleSize);
  return hiddenSize * visibleSize;
}

void Autoencoder::initialWeights(int hiddenSize, int visibleSize){
    malloc_float(b1, hiddenSize, 1);
    malloc_float(b2, visibleSize, 1);

    memset(b1, 0, sizeof(float) * hiddenSize);
    memset(b2, 0, sizeof(float) * visibleSize);

    malloc_float(weightMatrix1, hiddenSize, visibleSize);
    malloc_float(weightMatrix2, visibleSize, hiddenSize);
    unsigned seed = time(NULL);
    srand(seed);
    cerr << "autoencoder seed:" << seed << endl;
    float r = sqrt(6) / sqrt(hiddenSize + visibleSize + 1);
    for(int i = 0; i < hiddenSize * visibleSize; i++){
            weightMatrix1[i] = (rand() / (float)(RAND_MAX)) * 2 * r - r;
            weightMatrix2[i] = (rand() / (float)(RAND_MAX)) * 2 * r - r;
    }
}

void Autoencoder::computeCostAndGradient(float& cost, float* &gradientW1, float* &gradientW2, float* &gradientB1, float* &gradientB2, float* data, int col){  
    float* z2;
    float* a2;
    malloc_float(z2, hiddenSize, col);
    malloc_float(a2, hiddenSize, col);

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		hiddenSize, col, visibleSize,
		1.0, weightMatrix1, visibleSize,
		data, col, 0.0, 
		z2, col);
 
    float *rho;
    malloc_float(rho, hiddenSize, 1);
    memset(rho, 0, sizeof(float) * hiddenSize);
    float KLdivergence = 0.0;
    float *sparsityDelta;
    malloc_float(sparsityDelta, hiddenSize, 1);
     
    #pragma omp parallel for num_threads(NUM_THREADS)
    for(int i = 0; i < hiddenSize; i++){
        for(int j = 0;j < col; j++){
            a2[i * col + j] = 1 / (1 + exp(-1 * (z2[i * col + j] + b1[i])));
            rho[i] += a2[i * col + j];
        }
        rho[i] /= col;
        sparsityDelta[i] = -1 * sparsityParam / (rho[i] + 10e-37) + (1 - sparsityParam) / (1 - rho[i] + 10e-37);
    }

    for(int i = 0; i < hiddenSize; i++){
        KLdivergence += sparsityParam * log(sparsityParam / (rho[i] + 10e-37) + 10e-37) + (1 - sparsityParam) * log((1 - sparsityParam) / (1 - rho[i] + 10e-37) + 10e-37);
    }

    float* z3;
    float* a3;
    malloc_float(a3, visibleSize, col);
    malloc_float(z3, visibleSize, col);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		visibleSize, col, hiddenSize,
		1.0, weightMatrix2, hiddenSize,
		a2, col, 0.0, 
		z3, col);

    
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < visibleSize * col; i++){
       a3[i] = 1 / (1 + exp(-1 * (z3[i] + b2[i / col])));
    }
   
    float square = 0.0;
    float wSquare = 0.0; 
    float* delta3;
    float* delta2;
    #pragma omp sections
    {//omp sections begin
    #pragma omp section
    {
    #pragma omp parallel for reduction(+:square) num_threads(NUM_THREADS / 2)
    for(int i = 0; i < visibleSize * col; i++){
       square += (a3[i] - data[i]) * (a3[i] - data[i]);
    }
    }
    
    #pragma omp section
    {
    //compute the regularization
    #pragma omp parallel for reduction(+:wSquare) num_threads(NUM_THREADS / 3)
    for(int i = 0; i < hiddenSize * visibleSize; i++){
       wSquare += weightMatrix1[i] * weightMatrix1[i];
       wSquare += weightMatrix2[i] * weightMatrix2[i];
    }
    }
    
    #pragma omp section
   {
    //compute delta3 and delta2
    malloc_float(delta3, visibleSize, col);
    malloc_float(delta2, hiddenSize, col);
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < visibleSize * col; i++){
        delta3[i] = -1 * (data[i] - a3[i]) * a3[i] * (1 - a3[i]);
    }

    float* tmpArray;
    malloc_float(tmpArray, hiddenSize, col);
    memset(tmpArray, 0, sizeof(float) * hiddenSize * col);
    	
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		hiddenSize, col, visibleSize,
		1.0, weightMatrix2, hiddenSize,
		delta3, col, 0.0, 
		tmpArray, col);

    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < hiddenSize * col; i++){
        delta2[i] = (tmpArray[i] + beta * sparsityDelta[i / col]) * a2[i] * (1 - a2[i]);
    }
    mkl_free(tmpArray);
    } 
    }//omp sections end
    //compute the gradient
    cost = 0.5 * square / col + 0.5 * lambda * wSquare + beta * KLdivergence;
 
    #pragma omp sections
    {//sections begin
    #pragma omp section
    {
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < visibleSize * hiddenSize; i++){
        gradientW2[i] = lambda * weightMatrix2[i];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		visibleSize, hiddenSize, col,
		1.0/col, delta3, col,
		a2, col, 1.0, 
		gradientW2, hiddenSize);
    }
    
    #pragma omp section
    {
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < visibleSize; i++){
        gradientB2[i] = 0.0;
        for(int k = 0; k < col; k++){
            gradientB2[i] += delta3[i * col + k];
        }
        gradientB2[i] /= col;
    }
    }

    #pragma omp section
    {
    #pragma omp parallel for num_threads(NUM_THREADS/2)
    for(int i = 0;i < visibleSize * hiddenSize; i++){
        gradientW1[i] = lambda * weightMatrix1[i];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		hiddenSize, visibleSize, col,
		1.0/col, delta2, col,
		data, col, 1.0, 
		gradientW1, visibleSize);
    
    }
     
    #pragma omp section
    { 
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
     for(int i = 0; i < hiddenSize; i++){
        gradientB1[i] = 0.0;
        for(int k = 0;k < col; k++){
            gradientB1[i] += delta2[i * col + k];
        }
        gradientB1[i] /= col;
    }
    }
    }//sections end

    mkl_free(z2);
    mkl_free(a2);
    mkl_free(rho);
    mkl_free(z3);
    mkl_free(a3);
    mkl_free(sparsityDelta);
    mkl_free(delta3);
    mkl_free(delta2);
}


float* Autoencoder::feedForwardAutoencoder(float* data, int col){;
    float* output;
    malloc_float(output, hiddenSize, col);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		hiddenSize, col, visibleSize,
		1.0, weightMatrix1, visibleSize,
		data, col, 0.0, 
		output, col);
    #pragma omp parallel for num_threads(NUM_THREADS / 2)
    for(int i = 0; i < hiddenSize * col; i++){
        output[i] = 1 / (1 + exp(-1 * (output[i] + b1[i / col])));
    }
    return output;
}

void Autoencoder::setWeights(float* w1, float* w2, float* b1, float* b2){

    for(int i = 0; i < hiddenSize; i++){
        for(int j = 0; j < visibleSize; j++){
            this->weightMatrix1[i * visibleSize + j] = w1[i * visibleSize + j];
	}
    }
    for(int j = 0; j < visibleSize; j++){
	for(int i = 0; i < hiddenSize; i++){
            this->weightMatrix2[j * hiddenSize + i] = w2[j * hiddenSize + i];
        }
    }
    
    for(int i = 0; i < hiddenSize; i++){
        this->b1[i] = b1[i];
    }
    
    for(int i = 0; i < visibleSize; i++){
        this->b2[i] = b2[i];
    }
}

void Autoencoder::printParameter(){
    cout  << "W1:" << endl;
    for(int i = 0; i < hiddenSize; i++){
        for(int j = 0; j < visibleSize; j++){
            cout << this->weightMatrix1[i * visibleSize + j] << ",";
        }
        cout << endl;
    }
    cout << "b1:" << endl;
    for(int i = 0; i < hiddenSize; i++){
        cout << this->b1[i] << endl;
    } 
    cout << "W2:" <<endl;
    for(int i = 0; i < visibleSize; i++){
        for(int j = 0; j < hiddenSize; j++){
            cout << this->weightMatrix2[i * hiddenSize + j] << ",";
        }
        cout << endl;
    }
    cout << "b2:" << endl;
    for(int i = 0; i < visibleSize; i++){
        cout << this->b2[i] << endl;
    }
}

void Autoencoder::autoencoderTrain(float* data, int col, int maxIter, float precision, float learningRate){
    cout << "start autoencoder train" << endl;
    int iter = 0;
    float cost_old, cost_new, change;
    float *gradientW1, *gradientW2, *gradientB1, *gradientB2;
    malloc_float(gradientW1, hiddenSize, visibleSize);
    malloc_float(gradientW2, visibleSize, hiddenSize);
    malloc_float(gradientB1, hiddenSize, 1);
    malloc_float(gradientB2, visibleSize, 1);
    
    computeCostAndGradient(cost_old, gradientW1, gradientW2, gradientB1, gradientB2, data, col);
    cout << "cost_old:" << cost_old << endl;
    
    for(int i = 0; i < visibleSize * hiddenSize; i++){
        this->weightMatrix1[i] -= gradientW1[i] * learningRate;
        this->weightMatrix2[i] -= gradientW2[i] * learningRate;
    }

    for(int i = 0; i < visibleSize; i++){
        this->b2[i] -= gradientB2[i] * learningRate;
    }

    for(int i = 0; i < hiddenSize; i++){
        this->b1[i] -= gradientB1[i] * learningRate;
    }
    do{
        iter++;
        cout << "autoencoder iteration:" << iter <<endl;
        // compute gradient and update the gradient and change
        computeCostAndGradient(cost_new, gradientW1, gradientW2, gradientB1, gradientB2, data, col);
        for(int i = 0; i < visibleSize * hiddenSize; i++){
            this->weightMatrix1[i] -= gradientW1[i] * learningRate;
            this->weightMatrix2[i] -= gradientW2[i] * learningRate;
        }
        for(int i = 0; i < visibleSize; i++){
            this->b2[i] -= gradientB2[i] * learningRate;
        }
        for(int i = 0; i < hiddenSize; i++){
            this->b1[i] -= gradientB1[i] * learningRate;
        }
        cout << "cost_new:" << cost_new << endl;
        change = cost_new - cost_old;
        cost_old = cost_new;
    }while(iter < maxIter);
    cout << "finish autoencoder train" <<endl;
    mkl_free(gradientW1);
    mkl_free(gradientW2);
    mkl_free(gradientB1);
    mkl_free(gradientB2);
}
