#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <sys/time.h>
#include <cuda.h>
#include <chrono>


cudaEvent_t startm, stopm;

using namespace std;
double sigma = 1.;
double epsilon = 1.;

const int MAXPART=20001;
//  Position
double r[MAXPART*3] __attribute__((aligned (32)));
double a[MAXPART*3] __attribute__((aligned (32)));
int N = 20000;


__device__ double ourAtomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void computeAccelerationsDivision(double *rcuda, double *replicationcuda, int triplo){
    int i = blockIdx.x * blockDim.x * 3 + threadIdx.x;
    int aux = blockIdx.x * blockDim.x * 3;
    double ai0 = 0, ai1 = 0, ai2 = 0;
    for (int j = i+3; j < triplo; j+=3) {
        double rij0  = rcuda[i]   - rcuda[j];
        double rij1  = rcuda[i+1] - rcuda[j+1];
        double rij2  = rcuda[i+2] - rcuda[j+2];
        double rSqd  = 1 / (rij0 * rij0 + rij1 * rij1 + rij2 * rij2);
        double rSqd3 = rSqd * rSqd * rSqd;
        double f     = rSqd3 * rSqd * (2 * rSqd3 - 1);
        rij0  = rij0 * f;
        rij1  = rij1 * f;
        rij2  = rij2 * f;
        replicationcuda[aux+j]   -= rij0;
        replicationcuda[aux+j+1] -= rij1;
        replicationcuda[aux+j+2] -= rij2;
        ai0 += rij0;
        ai1 += rij1;
        ai2 += rij2;
    }
	replicationcuda[aux+i]   += ai0;
    replicationcuda[aux+i+1] += ai1;
    replicationcuda[aux+i+2] += ai2;
}


void divide_work(double *replication_matrix, int number_threads, int triplo, int num_blocks, int num_threads_per_block) {
    double *rcuda,*replicationcuda;
    int bytesR   = N * 3 * sizeof(double);
    int bytesRep = MAXPART*3*number_threads* sizeof(double);
    cudaMalloc ((void**) &rcuda, bytesR);
    cudaMalloc ((void**) &replicationcuda, bytesRep);
    cudaMemcpy (rcuda, r, bytesR, cudaMemcpyHostToDevice);
    cudaMemcpy (replicationcuda, replication_matrix, bytesRep, cudaMemcpyHostToDevice);
    computeAccelerationsDivision<<<num_threads_per_block,num_blocks>>>(rcuda,replicationcuda,N*3);
    cudaMemcpy (replication_matrix, replicationcuda, bytesRep, cudaMemcpyHostToDevice);
    cudaFree(replicationcuda);
    cudaFree(rcuda);
}

double sum_column(double *replication_matrix, int number_threads, int column) {
    double res = 0;
    // Soma dos valores das colunas
    for(int j = 0 ; j < number_threads; j++)
        res += replication_matrix[j * number_threads + column];
    return res;
}

void reduce_matrix_values(double *replication_matrix, int number_threads,int lim) {
    for (int i = 0; i < lim; i+=3) {
        a[i]   = 24 * sum_column(replication_matrix,number_threads,i);
        a[i+1] = 24 * sum_column(replication_matrix,number_threads,i+1);
        a[i+2] = 24 * sum_column(replication_matrix,number_threads,i+2);
    }
}

void computeAccelerations() {
    int triplo = 3*N;
    int num_blocks = 300;
    int num_threads_per_block = 64;
    int number_threads = num_blocks * num_threads_per_block;
    double *replication_matrix = (double*)calloc(MAXPART*3*number_threads, sizeof(double));
    divide_work(replication_matrix,number_threads,triplo,num_blocks,num_threads_per_block);   
    reduce_matrix_values(replication_matrix,number_threads,triplo);
    free(replication_matrix);
}



void computeAccelerationsSeq() {
    int i, j, triplo = 3*N;
    double rij0, rij1, rij2, rSqd, rSqd3, f;
    for (i = 0; i < triplo; i++)
        a[i] = 0;
    for (i = 0; i < triplo-3; i+=3) {
        for (j = i+3; j < triplo; j+=3) {
            rij0  = r[i] - r[j];
            rij1  = r[i+1] - r[j+1];
            rij2  = r[i+2] - r[j+2];
            rSqd  = 1 / (rij0 * rij0 + rij1 * rij1 + rij2 * rij2);
            rSqd3 = rSqd * rSqd * rSqd;
            f     = rSqd3 * rSqd * (2 * rSqd3 - 1);
            rij0  = rij0 * f;
            rij1  = rij1 * f;
            rij2  = rij2 * f;
            a[j]   -= rij0;
            a[j+1] -= rij1;
            a[j+2] -= rij2;

            a[i]   += rij0;
            a[i+1] += rij1;
            a[i+2] += rij2;
        }
    }
    for (i = 0; i < triplo; i+=3) {
        a[i] *= 24;
        a[i+1] *= 24;
        a[i+2] *= 24;
    }
}

int main() {
    for(int  i = 0; i < 3*N; i++)
    {
        r[i] = (double)rand() / 50;
    }
    chrono::steady_clock::time_point begin_seq = chrono::steady_clock::now();
    computeAccelerationsSeq();
    chrono::steady_clock::time_point begin_cuda = chrono::steady_clock::now();
    computeAccelerations();
    chrono::steady_clock::time_point end_cuda = chrono::steady_clock::now();
    cout << endl << "Sequential CPU execution: " << std::chrono::duration_cast<std::chrono::microseconds>(begin_cuda - begin_seq).count()  *0.001 << " miliseconds" << endl << endl;
    cout << endl << "Cuda           execution: " << std::chrono::duration_cast<std::chrono::microseconds>(end_cuda   - begin_cuda).count() *0.001 << " miliseconds" << endl << endl;
}