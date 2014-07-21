#ifdef __MIC__
	#define NUM_THREADS  240
	#define NESTED_THREADS 1
#else
	#define NUM_THREADS 32 
	#define NESTED_THREADS 2
#endif


#include <omp.h>
#include "mkl.h"
#include <stdlib.h>
#include <stdio.h>

#define set_num_threads(x) {} // {if ((x)<=NUM_THREADS) omp_set_num_threads(NUM_THREADS/2); else omp_set_num_threads(NUM_THREADS);}
#define set_nested_num_threads(x) {} //{if (omp_get_num_threads() <= NUM_THREADS) omp_set_num_threads(2);}
#define malloc_float(A,row, col){ A = (float*)mkl_malloc(sizeof(float)*row*col,64);\
    if(A == NULL){\
    fprintf(stderr,"Allocated error!!!!!!!!!!!!!!!!!!!!!!!!!");\
    exit(0);\
    }}
