/* 
 * File:   main.cpp
 * Author: labuser
 *
 * Created on September 16, 2013, 10:09 AM
 */

#include <cstdlib>
#include <iostream>
#include "Autoencoder.h"
#include "SoftMax.h"
#include "StackedAutoencoder.h"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include "time.h"
#include "consts.h"
using namespace std;

char tstr[1000000];

///////////// Some configurations //////////////////
// the configurations will be read from file in the future //

//////////About train & test data set
const int FEATURE_NUMBER = 784;//how many features does a sample have?
const int CLASS_NUMBER = 10;// how many classes are there? The number of classes in the classification task
const char* TRAIN_FILE_PATH = "train5.txt";
const char* TRAIN_LABEL_FILE_PATH = "trainLabels5.txt";
const int TRAIN_SAMPLE_NUMBER = 5000;//how many samples are there in the train set?
const char* TEST_FILE_PATH = "test.txt";
const char* TEST_LABEL_FILE_PATH = "testLabels.txt";
const int TEST_SAMPLE_NUMBER = 10000;//how many samples are there in the test set?
//////////About Network 
//Set the hidden layer size info of the Deep Neural Network.
const int LAYER_DEPTH = 2;//how many hidden layers?
int layerInfo[LAYER_DEPTH] = {200,200}; //how many nodes are there in each hidden layer?


/*
 * 
 */
int main(int argc, char** argv) {
    timespec b_time,e_time;
    clock_gettime(CLOCK_REALTIME,&b_time);
    ifstream fin(TRAIN_FILE_PATH,ios::in);
    if(fin == NULL)
        printf("error\n");
    string str;
    float *data;
	//Allocate space for train data set
		//Use MKL_MALLOC to allocate the memory in alignment
    data = (float*)mkl_malloc( FEATURE_NUMBER * TRAIN_SAMPLE_NUMBER * sizeof(float), 64 );
    int *trainLabels = new int[TRAIN_SAMPLE_NUMBER];
    float *test = new float[FEATURE_NUMBER * TEST_SAMPLE_NUMBER];
    int *testLabels = new int[TEST_SAMPLE_NUMBER];
    int i = 0;
    string substr;
    char *p;
    
    const char *d = " ";
    // load train data set
    while(getline(fin, str)){
        strcpy(tstr, str.c_str());
        char* tmp = tstr;
        p = strtok(tmp, d);
        while(p){
            sscanf(p, "%f", &data[i]);
            p = strtok(NULL, d);
            i++;
        }
    }
    fin.close();
	// load train labels
    fin.open(TRAIN_LABEL_FILE_PATH,ios::in);
    if(!fin){
       printf("error2\n");
       exit(2);
    }
    i = 0;
    while(getline(fin, str)){
        sscanf(str.c_str(), "%d", &trainLabels[i]);     
        i++;
    }
    fin.close();
    cout << "NUM_THREADS=" << NUM_THREADS << endl;
	// load test set
    fin.open(TEST_FILE_PATH, ios::in);
    i = 0;
    while(getline(fin, str)){
        strcpy(tstr, str.c_str());
        char* tmp = tstr;
        p = strtok(tmp, d);
        while(p){
            sscanf(p, "%f", &test[i]);
            p = strtok(NULL, d);
            i++;
        }
    }
    fin.close();
	
	// load test label
    fin.open(TEST_LABEL_FILE_PATH,ios::in);
    i = 0;
    while(getline(fin, str)){
        sscanf(str.c_str(), "%d", &testLabels[i]);
        i++;
    }
    fin.close();
    cout << "finished" <<endl;

	//Set the random seed. Change it as you want
    srand((unsigned)time(NULL));
	/* Make a new StackedAutoencoder, the parameters of the constructor function are:
	1. layer Size Info
	2. hidden layer number
	3. training data set size (number of traning samples)
	4. sparsity parameter
	5. lambda parameter
	6. beta parameter
	7. number of classes in the classification task
    */
    StackedAutoencoder *s = new StackedAutoencoder(layerInfo, LAYER_DEPTH,  FEATURE_NUMBER, 0.1, 
            0.003, 3, CLASS_NUMBER);
    clock_t start, finish;
    start = clock();
    cout << "start sda train at:" << start << endl;
	
	/* 
	* Start trainning.
	* The parameters of the Training function are:
	* 1. training data set
	* 2. number of samples
	* 3. maximum iteration times
	* 4. training sample labels
	* 5. precision threshhold to stop the iteration
	* 6. learning rate
	*/
    s->StackedAutoencoderTrain(data, TRAIN_SAMPLE_NUMBER, 500, trainLabels, 0.0000000001, 0.2);
	
    clock_gettime(CLOCK_REALTIME,&e_time);
    cout << " Start time :" << b_time.tv_sec << "." << b_time.tv_nsec/(double)1e9 << endl;
    cout << " End time :" << e_time.tv_sec << "." << e_time.tv_nsec/(double)1e9 << endl;
    finish = clock();
    cerr << "finish sda train at:" << finish << endl;
	
	//Predict for the test set
    int *predictLabels = s->predictLabels(test, TEST_SAMPLE_NUMBER);
	//Calculate classification accuracy on test set
    float acc = s->getAccuracy(predictLabels, testLabels, TEST_SAMPLE_NUMBER);
    delete[] predictLabels;
	
	//Start fine tune process
    cerr << "start finetune at" << clock() << endl;
	/*
	* Start fine tune.
	* The parameters of the FindTune function are:
	* 1. training set
	* 2. number of samples
	* 3. lebels of training samples
	* 4. maximum iteration count
	* 5. precision threshhold to stop the iteration
	* 6. learning rate
	*/
    s->stackedAutoencoderFineTune(data, TRAIN_SAMPLE_NUMBER, trainLabels, 100, 0.000001, 1.0);
    cerr << "finish finetune at" << clock() << endl;
    cerr << "before fine tune" << endl;
    cerr << "the accuracy is: " << acc << endl;
	//Predict the labels again.
    int *predictLabels2 = s->predictLabels(test, TEST_SAMPLE_NUMBER);
    cerr << "after fine tune" <<endl;
	//Get the classification accuracy on test set.
    cerr << "the accuracy is: " << s->getAccuracy(predictLabels2, testLabels,TEST_SAMPLE_NUMBER) << endl;
    delete[] predictLabels2;
    cerr << "finish all the procedure" <<endl;
    mkl_free(data);
    delete[] trainLabels;
    delete[] test;
    delete[] testLabels;
    delete s;
}

