#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int N = 2160;


double r[5001*3] __attribute__((aligned (32)));
double sigma = 1.;
double epsilon = 1.;

double PotentialAux(double sub1, double sub2, double sub3)
{
    double quot = sigma / sqrt(2);
    double quot6 = quot * quot * quot * quot * quot * quot;
    return quot6 * quot6 - quot6;
}


// Function to calculate the potential energy of the system
double Potential() {
    int j, i;
    double Pot[N] __attribute__((aligned (32)));
    for (int i = 0; i < N; i++)
        Pot[i] = 0.0;
    for (i=0; i<3*N; i+=3) {
        int aux = 0;
        for (j=0; j<i; j+=3, aux++)
            Pot[aux] += PotentialAux(r[i]-r[j],r[i+1]-r[j+1],r[i+2]-r[j+2]);
        for (j+=3; j < N*3; j+=3, aux++)
            Pot[aux] += PotentialAux(r[i]-r[j],r[i+1]-r[j+1],r[i+2]-r[j+2]);
    }
    
    double res = 0;
    for(int i = 0; i < N; i++)
        res += Pot[i];
    return 4 * epsilon * res;
}

int main()
{
    printf("%f\n",Potential());
    return 0;
}