#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <immintrin.h>
#include <assert.h>


//此处M,N,K的理解和下面的理解不一样
#define BLOCK_M 2 
#define BLOCK_N 1
#define BLOCK_K 2

#define M_INT 4 
#define K_INT 4 
#define N_INT 4 

#ifdef __MIC__
#define THREAD_NUM 240
#else
#define THREAD_NUM 16
#endif

float A[M_INT*K_INT],B[K_INT*N_INT],C[M_INT*N_INT];
float A1[M_INT*K_INT],B1[K_INT*N_INT],C1[M_INT*N_INT],C2[M_INT*N_INT];

/**
 * For a matrix `a`(the size of `a` is m*n), get the start address of submatrix(i,j). i & j start from 0.
 * ASSUME: that m%BLOCK_M == 0 && n%BLOCK_N ==0
 */
inline float* getStartAddress(float* a,int i,int j, int m, int n)
{
	assert(m%BLOCK_M + n%BLOCK_N == 0);
	return &a[i*BLOCK_M*n + j*BLOCK_N*BLOCK_M];
}

/**
 * Convert the normal matrix src(size is m*n) to the matrix presented in sub matrix formal.
 * dst should be allocated outside.
 */
float* NormalToBlockMatrix(float* dst, float* src,int m, int n)
{
	int i,j,k;
	for(i = 0; i < m/BLOCK_M; i++)
		for(j = 0; j < n/BLOCK_N; j++)
		{
			float* dstp = getStartAddress(dst,i,j,m,n);
			float* srcp = src + i*BLOCK_M*n + j*BLOCK_N;
			//Convert SubMatrix src(i,j)
			for(k = 0; k < BLOCK_M;k++)//Copy the k-th line of matrix to the sub-block-matrix
			{
				memcpy(dstp,srcp,sizeof(float)*BLOCK_N);
				//go to the next line
				dstp += BLOCK_N;
				srcp += n;
			}
		}
	return dst;
}

float* BlockToNormal(float* dst, float* src, int m,int n)
{
	int i,j,k;
	for(i = 0; i < m/BLOCK_M; i++)
		for(j = 0; j < n/BLOCK_N; j++)
		{
			 float* srcp = getStartAddress(src,i,j,m,n);
			 float* dstp = dst + i*BLOCK_M*n + j*BLOCK_N;
			//Convert SubMatrix src(i,j)
			for(k = 0; k < BLOCK_M;k++)//Copy the k-th line of matrix to the sub-block-matrix
			{
				memcpy(dstp,srcp,sizeof(float)*BLOCK_N);
				//go to the next line
				srcp += BLOCK_N;
				dstp += n;
			}
		}
	return dst;
}


/**
 * Multiply Matrices A & B to the Matrix C.
 * the sizes of A,B and C are:
 * -----------------
 * A		M * K
 * B		K * N
 * C		M * N
 * -----------------
 * Matrix C should be allocated outside.
 * Matrix C will not be cleared in this function, thus you can add new result to the Matrix C.
 */
void highEfficentMatrixMultiply(float* C, float* A, float* B, int M, int K, int N)
{
	int i,j,k;
	float tmp[16] __attribute__((align(64)));
//#pragma omp parallel for private(j,k) num_threads(THREAD_NUM)
	for(i=0; i < M; i++)
	{
		/* Code Run on Xeon Phi */
		#ifdef __MIC__
		__m512 _A,_B,_C;
		for(k = 0; k < K; k++)
		{
			_A = _mm512_set_1to16_ps(A[i*K + k]);
			/*
			_mm512_packstorelo_ps((void*)&tmp,_A);
			_mm512_packstorehi_ps((char*)&tmp + 64,_A);
			for(int s = 0 ; s < 16; s++)
				fprintf(stderr,"%f ",tmp[s]);
			*/
			//for(j = 0; j < N/16; j += 16)
			for(j = 0; j+16 < N; j += 16)
			{
				//fprintf(stderr,"[i,k,j,A[i,k]]=[%d,%d,%d,%f]\n",i,k,j,A[i*K+k]);
				_B = _mm512_loadunpacklo_ps(_B,(void*)(&B[k*N + j]));
				_B = _mm512_loadunpackhi_ps(_B,(void*)(&B[k*N + j + 16]));
				_C = _mm512_loadunpacklo_ps(_C,(void*)(&C[i*N + j]));
				_C = _mm512_loadunpackhi_ps(_C,(void*)(&C[i*N + j + 16]));

				_mm512_packstorelo_ps((void*)&tmp,_C);
				_mm512_packstorehi_ps((char*)&tmp + 64,_C);
				
				/*
				fprintf(stderr,"_C=\n");
				for(int s = 0 ; s < 4; s++)
					fprintf(stderr,"%f ",tmp[s]);
				fprintf(stderr,"\n");
				*/
				

				_C = _mm512_add_ps(_C,_mm512_mul_ps(_A,_B));
				_mm512_packstorelo_ps((void*)(&C[i*N+j]),_C);
				_mm512_packstorehi_ps((void*)(&C[i*N+j+16]),_C);

				
				/*
				_mm512_packstorelo_ps((void*)&tmp,_C);
				_mm512_packstorehi_ps((char*)&tmp + 64,_C);
				
				for(int s = 0 ; s < 4; s++)
					fprintf(stderr,"%f ",tmp[s]);
				fprintf(stderr,"\n");
				*/
				
			}
			if (j+16 > N)
			{
				
//				fprintf(stderr,"[j=%d]\n",j);
				//We should deal with the tail in each row
				float temp = A[i*K + k];
				#pragma ivdep 
				for(; j < N; j++)
					C[i*N + j] += temp * B[k*N + j];
					
			}



		}	
		
		#else
		/* Code Run On Xeon */
		for(k = 0; k < K; k++)
		{
			float temp = A[i*K + k];
			#pragma ivdep
			for(j = 0; j < N; j++)
			{
				C[i*N + j] += temp * B[k*N + j];
			}
		}
	#endif
	}
}

int MyTest(void)
{
	int i,j,k;
	//init
	for(i = 0 ; i < M_INT; i++)
		for(j = 0; j < K_INT; j++)
			A[i*K_INT + j] = (float)i;

	for(i = 0 ; i < K_INT; i++)
		for(j = 0; j < N_INT; j++)
			B[i*N_INT + j] = (float)j;

	memset(C,0,sizeof(float)*M_INT*N_INT);
	memset(C1,0,sizeof(float)*M_INT*N_INT);
	memset(C2,0,sizeof(float)*M_INT*N_INT);

	NormalToBlockMatrix(A1,A,M_INT,K_INT);
	NormalToBlockMatrix(B1,B,K_INT,N_INT);
	
	fprintf(stderr,"%d\n",getStartAddress(C1,0,1,M_INT,N_INT) - C1);

	for(i = 0; i < M_INT/BLOCK_M; i++)
		for(j = 0; j < N_INT/BLOCK_N; j++)
			for(k = 0; k < K_INT/BLOCK_K; k++)
			{
				//Calculate C1(i,j) = A1(i,k) * B1(k,j)
				highEfficentMatrixMultiply(
						getStartAddress(C1,i,j,M_INT,N_INT),
						getStartAddress(A1,i,k,M_INT,K_INT),
						getStartAddress(B1,k,j,K_INT,N_INT),
						BLOCK_M,BLOCK_K,BLOCK_N);
			}
	

	highEfficentMatrixMultiply(C, A, B, M_INT,K_INT,N_INT);

	BlockToNormal(C2,C1,M_INT,N_INT);

	for(i = 0; i < M_INT; i++)
	{
		for(j = 0; j < N_INT ; j++){
			fprintf(stderr,"%f|%f ",C[i*N_INT+j],C2[i*N_INT+j]);
	//		assert(C1[i*N_INT+j] == C[i*N_INT+j]);
		}
		printf("\n");
	}
			
	return 0;
}
