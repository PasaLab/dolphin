#include <iostream>
#include "mkl.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include <fstream>
#include <sstream>
#include <pthread.h>
#include "Rbm.h"

using namespace std;
char tstr[1000000];
float* chunk;
pthread_mutex_t mutex[10];
pthread_cond_t cond[10];
bool full[10];
int iter;
extern void readData(float* &data, string filename);

extern void* trainingThread(void*);
extern void* loadingThread(void*);
extern string itos(int& i);
/*
 * 
 */
int main(int argc, char** argv) { 
    float *data;
    data = (float*)mkl_malloc( 576 * 100000 * sizeof(float), 64 );
    memset(data, 0, sizeof(float) * 576 * 100000);
    Rbm* rbm = new Rbm(0.1, 1, 576, 1024);
    string path = "./DataSet/";
    readData(data, "./DataSet/576.txt");
    cerr << "finish reading data" << endl;
    float* p;
    for(int i = 0; i < 500; i++){
	p = &data[(i%50) * 200 * 576];
	//printf("hereh\n");
        rbm->train(p, 1, 200);
	printf("iter:%d\n",i);
    }
    //rbm->train(data, 20, 10000);
    mkl_free(data);
    delete rbm;
}

/*int main(int argc, char** argv){
    iter = 10;
    for(int i = 0; i < 10; i++){
        pthread_mutex_init(&mutex[i], NULL);
        pthread_cond_init(&cond[i], NULL);
        full[i] = false;
    }
    
    pthread_t lThread, tThread;
    long t1 = 1, t2 = 2;
    chunk = (float*)mkl_malloc(sizeof(float) * 576 * 100000, 64);
    if(chunk == NULL)
       printf("malloc failure\n");
    pthread_create(&lThread, NULL, loadingThread, (void*)t1);
    pthread_create(&tThread, NULL, trainingThread, (void*)t2);
    pthread_join(lThread, NULL);
    pthread_join(tThread, NULL);
    
    pthread_exit(0);
    mkl_free(chunk);
    return 0;
}*/

void* loadingThread(void*){
    string prefix = "./DataSet/";
    for(int i = 0; i < iter; i++){
       int j = i + 1;
       string suffix = itos(j);
       string path = prefix + suffix + ".txt";
       pthread_mutex_lock(&mutex[i % 10]);
       printf("loading: enter critical section! %d\n", i % 10);
       if(full[i % 10]){
           printf("loading: block %d is still full, wait for use!\n", i % 10);
           pthread_cond_wait(&cond[i % 10], &mutex[i % 10]);
       }
       printf("loading: block %d is empty, we need to load!\n", i % 10);
       float* data = &chunk[(i % 10) * 10000];
       readData(data, path);
       printf("finish reading %d.txt\n", j);
       full[i % 10] = true;
       printf("loading: begin to exit critical section! %d\n", i % 10);
       pthread_mutex_unlock(&mutex[i % 10]);
       printf("loading: begin to signal cond %d\n", i % 10);
       pthread_cond_signal(&cond[i % 10]);
    }
    printf("finish loading all the data\n");
    return NULL;
}

void* trainingThread(void*){
    int i = 0;
    Rbm* rbm = new Rbm(0.1, 1, 576, 1024);
    while(i < iter){
        pthread_mutex_lock(&mutex[i % 10]);
        printf("training: enter critical section! %d\n", i % 10);
        if(!full[i % 10]){
            printf("training: block %d is empty, wait for loading!\n", i % 10);
            pthread_cond_wait(&cond[i % 10], &mutex[i % 10]);
        }
        //get the data and train
        printf("training: block %d is full, begin to train\n", i % 10);
        float* data = &chunk[(i % 10) * 10000];
        float* p;
        for(int j = 0; j < 50; j++){
           p = &data[j * 200];
           rbm->train(p, 1, 200);
	   printf("iter:%d\n",j);
        }
        printf("finish training %d.txt\n", (i+1));
        full[i % 10] = false;
        printf("training: begin to exit critical section %d\n", i % 10);
        pthread_mutex_unlock(&mutex[i % 10]);
        printf("training: begin to signal loading thread to load %d\n", i % 10);
        pthread_cond_signal(&cond[i % 10]);
        i++;
    }
    printf("finish training all the data\n");
    return NULL;
}

string itos(int &i){
    stringstream s;
    s << i;
    return s.str();
}

void readData(float* &data, string filename){
	ifstream fin(filename.c_str(),ios::in);
	if(fin == NULL){
	   printf("error!\n");
           cerr << filename << endl;
        }
        
        memset(tstr, 0, sizeof(char) * 1000000);
	string str="";
	char *p;
	const char *d = " ";

	int i = 0;
	while(getline(fin, str)){
		strcpy(tstr,str.c_str());
		char *tmp = tstr;
		p = strtok(tmp,d);
		while(p){
			sscanf(p,"%f",&data[i]);
			p = strtok(NULL,d);
			i++;
		}
	}
        printf("read %d data\n", i);
	fin.close();

}
