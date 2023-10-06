#include <stdio.h>
#include <stdlib.h>
int N = 2160;

double PotentialAux(int i, int j)
{
    return 1;
}

double sum(double a[])
{
    double res = 0;
    for(int i = 0; i < N; i++)
    {
        res += a[i];
    }
    return res;
}

// Function to calculate the potential energy of the system
double Potential() {
    int j, i;
    double Pot[N] __attribute__((aligned (32)));
    for (int i = 0; i < N; i++) {
        Pot[i] = 0.0;
    }
    for (i=0; i<N; i++) {
        for (j=0; j<i; j++)
            Pot[j] += PotentialAux(i,j);
        for (j=i+1; j < N; j++)
            Pot[j] += PotentialAux(i,j);
    }
    double res = sum(Pot);
    return res;
}

int main()
{
    printf("%f\n",Potential());
    return 0;
}