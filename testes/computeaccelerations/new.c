#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int N = 2160;

const int MAXPART=5001;

void computeAccelerations() {
    int i, j;
    double a[MAXPART*3] __attribute__((aligned (32)));
    double r[MAXPART*3] __attribute__((aligned (32)));
    double rij0, rij1, rij2, rSqd, rSqd3, f;
    for (i = 0; i < 3*N; i++)
        a[i] = 0;
    for (i = 0; i < 3*(N-1); i+=3) {
        for (j = i+3; j < 3*N; j+=3) {
            rij0  = r[i] - r[j];
            rij1  = r[i+1] - r[j+1];
            rij2  = r[i+2] - r[j+2];
            rSqd  = 1 / (rij0 * rij0 + rij1 * rij1 + rij2 * rij2);
            rSqd3 = rSqd * rSqd * rSqd;
            f     = 24 * (2 * rSqd3 * rSqd * rSqd3 - rSqd3 * rSqd);
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
}


int main()
{
    computeAccelerations();
    return 0;
}