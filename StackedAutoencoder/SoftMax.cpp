/* 
 * File:   SoftMax.cpp
 * Author: labuser
 * 
 * Created on September 18, 2013, 2:00 PM
 */
#include <iostream>
#include "SoftMax.h"
#include <math.h>
#include <float.h>
#include <cstdlib>
#include <cstring>
#include "consts.h"
#include <time.h>
using namespace std;

SoftMax::SoftMax() {
}

SoftMax::SoftMax(int inputSize, int numClasses, float lambda){
    this->inputSize = inputSize;
    this->numClasses = numClasses;
    this->lambda = lambda;
    
    this->theta = new float[numClasses * inputSize];
    initialTheta(numClasses, inputSize);
}

void SoftMax::initialTheta(int numClasses, int inputSize){
    unsigned seed = time(NULL);
    srand(seed);
    cerr << "softmax seed:" << seed << endl;
    for(int i = 0; i < numClasses * inputSize; i++){
        this->theta[i] = normal_rand() * 0.005;
    }
}

float SoftMax::normal_rand(){
    static float V1, V2, S;
    static int phase = 0;
    float X;
    if(phase == 0){
        do{
            float U1 = (float)rand() / RAND_MAX;
            float U2 = (float)rand() / RAND_MAX;
            
            V1 = 2 * U1-1;
            V2 = 2 * U2-1;
            S = V1 * V1 + V2 * V2;
        }while(S >= 1 || S == 0);
        
        X = V1 * sqrt(-2 * log(S) / S);      
    } else
        X = V2 * sqrt(-2 * log(S) / S);
    
    phase = 1 - phase;
    return X;
}

SoftMax::SoftMax(const SoftMax& orig) {
}

SoftMax::~SoftMax() {
    delete[] theta;
}

void SoftMax::computeCostAndGradient(float* data, int col, float& cost, float* &thetaGrad, float* groundTruth){
    //compute theta*data theta numClasses*inputSize  data: inputSize * col
    float* M;
    malloc_float(M, numClasses, col);
    float* sum = new float[col];
    memset(sum, 0, sizeof(float) * col);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		numClasses, col, inputSize,
		1.0, theta, inputSize,
		data, col, 0.0, 
		M, col);
    //to prevent overflow , we compute the maximum value of every column
    float* max = new float[col];
    memset(max, 0, sizeof(float) * col);
    #pragma omp parallel for num_threads(NUM_THREADS / 6)
    for(int i = 0; i < numClasses * col; i++){
       if(M[i] >= max[i % col])
          max[i % col] = M[i];
    }
    #pragma omp parallel for num_threads(NUM_THREADS / 6)
    for(int i = 0; i < numClasses * col; i++){
       M[i] -= max[i % col];
       M[i] = exp(M[i]);
       sum[i % col] += M[i];
    }
    #pragma omp parallel for num_threads(NUM_THREADS / 6)
    for(int i = 0; i < numClasses * col; i++){
        M[i] /= sum[i % col];
    }

    float costMatrix = 0.0;
    #pragma omp parallel for num_threads(NUM_THREADS / 6)
    for(int i = 0; i < numClasses * col; i++){
        costMatrix += groundTruth[i] * log(M[i] + 10e-37);
    }
    
    float thetaCost = 0.0;
    #pragma omp parallel for num_threads(NUM_THREADS / 6)
    for(int i = 0; i < numClasses * inputSize; i++){
        thetaCost += theta[i] * theta[i];
    }
    
    cost = -1 * costMatrix / col + 0.5 * lambda * thetaCost;
    float* tmpArray;
    malloc_float(tmpArray, numClasses, col);
    #pragma omp sections
    {
    #pragma omp section
    {
    #pragma omp parallel for num_threads(NUM_THREADS / 4)
    for(int i = 0; i < numClasses * col; i++){
        tmpArray[i] = groundTruth[i] - M[i];
    }
    }
    #pragma omp section
    {  
    #pragma omp parallel for num_threads(NUM_THREADS / 6)
    for(int i = 0; i < numClasses * inputSize; i++){
        thetaGrad[i] = lambda * theta[i];
    }
    }
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		numClasses, inputSize, col,
		-1.0 / col, tmpArray, col,
		data, col, 1.0, 
		thetaGrad, inputSize);
    mkl_free(tmpArray);
    mkl_free(M);
    delete[] sum;
    delete[] max;
}

void SoftMax::setTheta(float* theta){
     memcpy(this->theta, theta, sizeof(float) * numClasses * inputSize);
}

float* SoftMax::feedForward(float* data, int col){
    float* M;
    malloc_float(M, numClasses, col);
    float* sum = new float[col];
    memset(sum, 0, sizeof(float) * col);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		numClasses, col, inputSize,
		1.0, theta, inputSize,
		data, col, 0.0, 
		M, col);

    //to prevent overflow , we compute the maximum value of every column
    float* max = new float[col];
    memset(max, 0, sizeof(float) * col);
    #pragma omp parallel for num_threads(NUM_THREADS / 6)
    for(int i = 0; i < numClasses * col; i++){
       if(M[i] >= max[i % col])
          max[i % col] = M[i];
    }
    #pragma omp parallel for num_threads(NUM_THREADS / 6)
    for(int i = 0; i < numClasses * col; i++){
       M[i] -= max[i % col];
       M[i] = exp(M[i]);
       sum[i % col] += M[i];
    }
    #pragma omp parallel for num_threads(NUM_THREADS / 4)
    for(int i = 0;i < numClasses * col; i++){
        M[i] /= sum[i % col];
    }
    delete []sum;
    delete []max;
    return M;
}

void SoftMax::softMaxTrain(float* data, int col, int * labels, int maxIter, float learningRate, float precision){
    int iter = 0;
    float cost_old, cost_new, change;
    float* gradient;
    malloc_float(gradient, numClasses, inputSize);
    float* groundTruth = new float[numClasses * col];
    memset(groundTruth, 0, sizeof(float)*numClasses * col);
    for(int i = 0; i < col; i++){
        groundTruth[(labels[i] - 1) * col + i] = 1;
    }

    computeCostAndGradient(data, col, cost_old, gradient, groundTruth);
    for(int i = 0; i < numClasses * inputSize; i++){
        this->theta[i] -= gradient[i] * learningRate; 
    }
    do{
        iter++;
        cout << "softmax iteration: " << iter << endl;
        // compute gradient and update the gradient and change
        computeCostAndGradient(data, col, cost_new, gradient, groundTruth);
        for(int i = 0;i < numClasses * inputSize; i++){
            this->theta[i] -= gradient[i] * learningRate;
        }
        change = cost_new - cost_old;
        cost_old = cost_new;
        cout << cost_new <<endl;
    }while(iter < maxIter);
    mkl_free(gradient);
    delete[] groundTruth;
}

void SoftMax::printParameters(){
    cout << "inputSize:" << inputSize << endl;
    cout << "numClasses:" << numClasses <<endl;
    cout << "lambda:" << lambda <<endl;
    
    cout << "theta:" << endl;
    for(int i = 0; i < numClasses;i++){
        for(int j = 0; j< inputSize;j++){
            cout << theta[i * inputSize + j] <<",";
        }
        cout << endl;
    }
}

