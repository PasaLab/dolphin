#include "mkl.h"
#include "stdlib.h"
#include <iostream>
#include <math.h>
#include <string.h>
#include "Rbm.h"
#include "consts.h"

using namespace std;
Rbm::Rbm(float momentum, float alpha, int visibleSize, int hiddenSize){
  this->momentum = momentum;
  this->alpha = alpha;
  this->visibleSize = visibleSize;
  this->hiddenSize = hiddenSize;
  
  W = (float*)mkl_malloc(sizeof(float) * hiddenSize * visibleSize, 64);
  vW = (float*)mkl_malloc(sizeof(float) * hiddenSize * visibleSize, 64);
  memset(W, 0, sizeof(float) * hiddenSize * visibleSize);
  b = (float*)mkl_malloc(sizeof(float) * visibleSize, 64);
  vb = (float*)mkl_malloc(sizeof(float) * visibleSize, 64);
  memset(b, 0, sizeof(float) * visibleSize);
  memset(vb, 0, sizeof(float) * visibleSize);
  c = (float*)mkl_malloc(sizeof(float) * hiddenSize, 64);
  vc = (float*)mkl_malloc(sizeof(float) * hiddenSize, 64);
  memset(c, 0, sizeof(float) * hiddenSize);
  memset(vc, 0, sizeof(float) * hiddenSize);
  random = (float*)mkl_malloc(sizeof(float) * (hiddenSize >= visibleSize ? hiddenSize:visibleSize), 64);
  int max = hiddenSize >= visibleSize ? hiddenSize : visibleSize;
  for(int i = 0; i < max; i++){
    random[i] = rand() / RAND_MAX;
  }
}

Rbm::~Rbm(){
  mkl_free(W);
  mkl_free(vW);
  mkl_free(b);
  mkl_free(vb);
  mkl_free(c);
  mkl_free(vc);
  mkl_free(random);
}
float Rbm::computeCostAndGradient(float* &data, int batchSize){
  float error = 0.0;  //v1 = data
  //h1 = sigmrnd(v1*W'+c) batchSize * hiddenSize
  /*float* h1 = (float*)mkl_malloc(sizeof(float) * batchSize * hiddenSize, 64);
  memset(h1, 0, sizeof(float) * batchSize * hiddenSize);*/
  //printf("h1:%p\n",h1);
  //printf("before the first matrix\n");
  //printf("1\n");
  /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		batchSize, hiddenSize, visibleSize,
		1.0, data, visibleSize,
		W, visibleSize, 0.0, 
		h1, hiddenSize);*/
  for(int i = 0; i < batchSize; i++){
    for(int j = 0; j < hiddenSize; j++){
        h1[i * hiddenSize + j] = 0;
        for(int k = 0; k < visibleSize; k++){
            h1[i * hiddenSize + j] += data[i * visibleSize + k] * W[j * visibleSize + k];
        }
    }
  }
  //printf("2\n");
  //#pragma omp parallel for num_threads(NUM_THREADS/8)
  //#pragma ivdep
  for(int i = 0; i < batchSize * hiddenSize; i++){
    h1[i] = 1 / (1 + exp(-1 * (h1[i] + c[i % hiddenSize])));
    if(h1[i] > random[i % hiddenSize])
      h1[i] = 1;
    else
      h1[i] = 0;
  }
//  printf("3\n");
  //v2 = sigmrnd(h1*W+b) batchSize * visibleSize
  /*float* v2 = (float*)mkl_malloc(sizeof(float) * batchSize * visibleSize, 64);
  memset(v2, 0, sizeof(float) * batchSize * visibleSize);*/
  //printf("v2:%p\n",v2);
  /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
		batchSize, visibleSize, hiddenSize,
		1.0, h1, hiddenSize,
		W, visibleSize, 0.0, 
		v2, visibleSize);*/
  for(int i = 0; i < batchSize; i++){
    for(int j = 0; j < visibleSize; j++){
        v2[i * visibleSize + j] = 0.0;
        for(int k = 0; k < hiddenSize; k++){
            v2[i * visibleSize + j] += h1[i * hiddenSize + k] * W[k * visibleSize + j];
        }
    }
  }
  //printf("4\n");
  //#pragma omp parallel for num_threads(NUM_THREADS/8)
 // #pragma ivdep
  for(int i = 0; i < batchSize * visibleSize; i++){
    v2[i] = 1 / (1 + exp(-1 * (v2[i] + b[i % visibleSize])));
    if(v2[i] > random[i % visibleSize])
      v2[i] = 1;
    else
      v2[i] = 0;
  }

  //printf("%d",2);
  //h2 = sigm(v2*W'+c)
  /*float* h2 = (float*)mkl_malloc(sizeof(float) * batchSize * hiddenSize, 64);
  memset(h2, 0, sizeof(float) * batchSize * hiddenSize);*/
  //printf("h2:%p\n",h2);
  /*cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		batchSize, hiddenSize, visibleSize,
		1.0, v2, visibleSize,
		W, visibleSize, 0.0, 
		h2, hiddenSize);*/
  for(int i = 0;i < batchSize; i++){
    for(int j = 0; j < hiddenSize; j++){
        h2[i * hiddenSize + j] = 0.0;
        for(int k = 0; k < visibleSize; k++){
            h2[i * hiddenSize + j] += v2[i * visibleSize + k] * W[j * visibleSize + k];
        } 
    }
  }


  //#pragma omp parallel for num_threads(NUM_THREADS/8)
  //#pragma ivdep
  for(int i = 0; i < batchSize * hiddenSize; i++){
    h2[i] = 1 / (1 + exp(-1 * (h2[i] + c[i % hiddenSize])));
  }

  //c1 = h1'*v1 c2=h2'*v2
  /*float* c1 = (float*)mkl_malloc(sizeof(float) * hiddenSize * visibleSize, 64);
  float* c2 = (float*)mkl_malloc(sizeof(float) * hiddenSize * visibleSize, 64);
  memset(c1, 0, sizeof(float) * hiddenSize * visibleSize);
  memset(c2, 0, sizeof(float) * hiddenSize * visibleSize);*/
  //printf("c1:%p\n",c1);
  //printf("c2:%p\n",c2);
  //#pragma omp sections
  //{
  //#pragma omp section
  //{
// printf("5\n");
  /*cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		hiddenSize, visibleSize, batchSize,
		1.0, h1, hiddenSize,
		data, visibleSize, 0.0, 
		c1, visibleSize);*/
  for(int i = 0; i < hiddenSize; i++){
    for(int j = 0; j < visibleSize; j++){
        c1[i * visibleSize + j] = 0.0;
        for(int k = 0; k < batchSize; k++){
            c1[i * visibleSize + j] += h1[k * hiddenSize + i] * data[k * visibleSize + j];
        }
    }
  }
 //printf("6\n");
  //}
  //#pragma omp section
  //{
  /*cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
		hiddenSize, visibleSize, batchSize,
		1.0, h2, hiddenSize,
		v2, visibleSize, 0.0, 
		c2, visibleSize);*/
  for(int i = 0; i < hiddenSize; i++){
    for(int j = 0; j < visibleSize; j++){
        c2[i * visibleSize + j] = 0.0;
        for(int k = 0; k < batchSize; k++){
            c2[i * visibleSize + j] += h2[k * hiddenSize + i] * v2[k * visibleSize + j];
        }
    }
  }
 // printf("7\n");
  //}
  //}		
  //compute gradient
 // #pragma omp sections
  //{
  //#pragma omp section
  //{
  //#pragma omp parallel for num_threads(NUM_THREADS/10)
  //#pragma ivdep
  for(int i = 0; i < hiddenSize * visibleSize; i++){
    vW[i] = momentum * vW[i] + alpha * (c1[i] - c2[i]) / batchSize;
  }
  //printf("7.1\n");
  //}

  //#pragma omp section
  //{

  //#pragma omp parallel for num_threads(NUM_THREADS/8)
  //#pragma ivdep
  for(int i = 0; i < visibleSize; i++){
    vb[i] = momentum * vb[i];
  }
 
  /*#pragma omp parallel for num_threads(NUM_THREADS/10)
  for(int i = 0; i < batchSize; i++){
    for(int j = 0; j < visibleSize; j++){
      vb[j] += alpha * (data[i * visibleSize + j] - v2[i * visibleSize + j]) / batchSize;
      error += (data[i * visibleSize + j] - v2[i * visibleSize + j]) * (data[i * visibleSize + j] - v2[i * visibleSize + j]);
    }
  }*/
  
  int tmp = batchSize * visibleSize - 1;
  //#pragma omp parallel for num_threads(NUM_THREADS / 10)
  for(int i = 0; i < batchSize * visibleSize; i++){
     vb[i % visibleSize] += alpha * (data[i] - v2[i]);
     error += (data[i] - v2[i]) * (data[i] - v2[i]);
     if(i == tmp)
       vb[i % visibleSize] /= batchSize;
  }
  //printf("7.2\n");
 // }

  //#pragma omp section
  //{

  //#pragma omp parallel for num_threads(NUM_THREADS/10)
  //#pragma ivdep
  for(int i = 0; i < hiddenSize; i++){
    vc[i] = momentum * vc[i];
  }
  //cerr << "7.3.1" <<endl;
  //#pragma omp parallel for num_threads(NUM_THREADS/10)
  for(int i = 0; i < batchSize; i++){
    //#pragma ivdep
    for(int j = 0; j < hiddenSize; j++){
      vc[j] += alpha * (h1[i * hiddenSize + j] - h2[i * hiddenSize + j]) / batchSize;
    }
  }
 // printf("7.3.2\n");
  //}
  //}
 //printf("8\n");
  error /= batchSize;  
 // mkl_free(h1);

 // mkl_free(v2);
 
 // mkl_free(h2);
 
 // mkl_free(c1);

  //mkl_free(c2);
 
  return error;
}

void Rbm::updateWeight(){
  //#pragma omp parallel for num_threads(NUM_THREADS/8)
  //#pragma ivdep
  for(int i = 0; i < hiddenSize * visibleSize; i++){
    W[i] += vW[i];
    if(i < visibleSize)
      b[i] += vb[i];
    if(i < hiddenSize)
      c[i] += vc[i];
  }
}

void Rbm::train(float* &data, int iter, int batchSize){
  int count = 0;
  float error = 0.0;
  h1 = (float*)mkl_malloc(sizeof(float) * batchSize * hiddenSize, 64);
  h2 = (float*)mkl_malloc(sizeof(float) * batchSize * hiddenSize, 64);
  v2 = (float*)mkl_malloc(sizeof(float) * batchSize * visibleSize, 64);
  c1 = (float*)mkl_malloc(sizeof(float) * hiddenSize * visibleSize, 64);
  c2 = (float*)mkl_malloc(sizeof(float) * hiddenSize * visibleSize, 64);
  while(count < iter){
    error = computeCostAndGradient(data, batchSize);
    printf("iter %d, reconstruction error: %f\n", count, error);
    updateWeight();
    count++;
  }
  mkl_free(h1);
  mkl_free(h2);
  mkl_free(v2);
  mkl_free(c1);
  mkl_free(c2);
}





