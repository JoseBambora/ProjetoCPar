#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int N = 2160;

double sigma = 1.;
double epsilon = 1.;
double r[5001*3] __attribute__((aligned (32)));

double PotentialAux(double sub1, double sub2, double sub3)
{
    double quot = sigma / sqrt(sub1 * sub1 + sub2 * sub2 + sub3 * sub3);
    double quot6 = quot * quot * quot * quot * quot * quot;
    double quot12 = quot6*quot6;
    return 4 * epsilon * (quot12 - quot6);
}


// Function to calculate the potential energy of the system
double Potential() {
    int j, i;
    double Pot[N] __attribute__((aligned (32)));
    for (int i = 0; i < N; i++) {
        Pot[i] = 0.0;
    }
    for (i=0; i<3*N; i+=3) {
        for (j=0; j<i; j+=3)
        {
            Pot[j%3] += PotentialAux(r[i]-r[j],r[i+1]-r[j+1],r[i+2]-r[j+2]);
        }
        for (j=i+3; j < N*3; j+=3)
        {
            Pot[j%3] += PotentialAux(r[i]-r[j],r[i+1]-r[j+1],r[i+2]-r[j+2]);
        }
    }
    double res = 0;
    for(int i = 0; i < N; i++)
    {
        res += Pot[i];
    }
    return res;
}


int main()
{
    for(int i = 0; i < 5001*3; i++)
        r[i] = 1;
    printf("%f\n",Potential());
    return 0;
}