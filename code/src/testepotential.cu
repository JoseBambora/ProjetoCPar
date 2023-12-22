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

__device__ double PotentialMath(double sub1, double sub2, double sub3,double sigma)
{
    double quot = sigma * sigma / (sub1 * sub1 + sub2 * sub2 + sub3 * sub3);
    double quot6 = quot * quot * quot;
    return quot6 * quot6 - quot6;
}

__global__ void PotentialDivision(double *rcuda, double *result, int triplo,double sigma) {
    double Pot = 0;
    // i = 0, j = 0
    // i = 0, j = 1

    int i = blockIdx.x * blockDim.x + threadIdx.x * 3;
    if( i < triplo)
    {
     	for (int j=0; j < i; j+=3)
                Pot += PotentialMath(rcuda[i]-rcuda[j],rcuda[i+1]-rcuda[j+1],rcuda[i+2]-rcuda[j+2],sigma);
        for (int j=i+3; j < triplo; j+=3)
                Pot += PotentialMath(rcuda[i]-rcuda[j],rcuda[i+1]-rcuda[j+1],rcuda[i+2]-rcuda[j+2],sigma);
        ourAtomicAdd(result, Pot);
    }
}


double Potential() {
    int num_blocks = 256;
    int num_threads_per_block = 256;
    double *rcuda, *potcuda, Pot;
    int bytes = N * 3 * sizeof(double);
    cudaMalloc ((void**) &rcuda, bytes);
    cudaMalloc ((void**) &potcuda, sizeof(double));
    cudaMemcpy (rcuda, r, bytes, cudaMemcpyHostToDevice);
    cudaMemset (potcuda, 0, sizeof(double));
    PotentialDivision<<<num_blocks,num_threads_per_block>>>(rcuda,potcuda,3*N,sigma);
    cudaMemcpy(&Pot, potcuda, sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(rcuda);
    cudaFree(potcuda);
    return 4 * epsilon * Pot;
}




double PotentialAuxSeq(double sub1, double sub2, double sub3)
{
    double quot = sigma * sigma / (sub1 * sub1 + sub2 * sub2 + sub3 * sub3);
    double quot6 = quot * quot * quot;
    return quot6 * quot6 - quot6;
}

// Function to calculate the potential energy of the system
double PotentialSeq() {
    int j, i, triplo = 3*N;
    double Pot = 0.0;
    for (i=0; i<triplo; i+=3) {
        for (j=0; j<i; j+=3)
            Pot += PotentialAuxSeq(r[i]-r[j],r[i+1]-r[j+1],r[i+2]-r[j+2]);
        for (j=i+3; j < triplo; j+=3)
            Pot += PotentialAuxSeq(r[i]-r[j],r[i+1]-r[j+1],r[i+2]-r[j+2]);
    }
    return 4 * epsilon * Pot;
}

int main() {
    for(int  i = 0; i < 3*N; i++)
    {
        r[i] = (double)rand() / 50;
    }
    chrono::steady_clock::time_point begin_seq = chrono::steady_clock::now();
    double seq = PotentialSeq();
    chrono::steady_clock::time_point begin_cuda = chrono::steady_clock::now();
    double cuda = Potential();
    chrono::steady_clock::time_point end_cuda = chrono::steady_clock::now();
    cout << endl << "Sequential CPU execution: " << std::chrono::duration_cast<std::chrono::microseconds>(begin_cuda - begin_seq).count()  *0.001 << " miliseconds" << endl << endl;
    cout << endl << "Cuda           execution: " << std::chrono::duration_cast<std::chrono::microseconds>(end_cuda   - begin_cuda).count() *0.001 << " miliseconds" << endl << endl;
    printf("%.30f\n",seq);
    printf("%.30f\n",cuda);
}