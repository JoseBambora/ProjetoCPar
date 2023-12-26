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

const int MAXPART=30001;
//  Position
double r[MAXPART*3] __attribute__((aligned (32)));
double a[MAXPART*3] __attribute__((aligned (32)));
int N = 30000;


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

__device__ double ourAtomicMinus(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val -
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

__global__ void computeAccelerationsDivision(double *rcuda, double *acuda, int triplo){
    int i = blockIdx.x * blockDim.x * 3 + threadIdx.x;
    if (i < triplo) {
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
            ourAtomicMinus(&acuda[j],rij0);
            ourAtomicMinus(&acuda[j+1],rij1);
            ourAtomicMinus(&acuda[j+2],rij2);
            ai0 += rij0;
            ai1 += rij1;
            ai2 += rij2;
        }
        ourAtomicAdd(&acuda[i],ai0);
        ourAtomicAdd(&acuda[i+1],ai1);
        ourAtomicAdd(&acuda[i+2],ai2);
    }
}


void computeAccelerations() {
    int triplo = 3 * N;
    for (int i = 0; i < triplo; i++)
        a[i] = 0;
    int num_blocks = 256;
    int num_threads_per_block = 256;
    double *rcuda,*acuda;
    int bytes   = N * 3 * sizeof(double);
    cudaMalloc ((void**) &rcuda, bytes);
    cudaMalloc ((void**) &acuda, bytes);
    cudaMemcpy (rcuda, r, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy (acuda, a, bytes, cudaMemcpyHostToDevice);
    computeAccelerationsDivision<<<num_threads_per_block,num_blocks>>>(rcuda,acuda,N*3);
    cudaMemcpy (a, acuda, bytes, cudaMemcpyDeviceToHost);
    cudaFree(acuda);
    cudaFree(rcuda);
    for (int i = 0; i < triplo; i+=3) {
        a[i] *= 24;
        a[i+1] *= 24;
        a[i+2] *= 24;
    }
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
    cout << endl << a[0] << ", " << a[1] << ", " << a[2] << endl;
    chrono::steady_clock::time_point begin_cuda = chrono::steady_clock::now();
    computeAccelerations();
    cout << endl << a[0] << ", " << a[1] << ", " << a[2] << endl;
    chrono::steady_clock::time_point end_cuda = chrono::steady_clock::now();
    cout << endl << "Sequential CPU execution: " << std::chrono::duration_cast<std::chrono::microseconds>(begin_cuda - begin_seq).count()  *0.001 << " miliseconds" << endl << endl;
    cout << endl << "Cuda           execution: " << std::chrono::duration_cast<std::chrono::microseconds>(end_cuda   - begin_cuda).count() *0.001 << " miliseconds" << endl << endl;

}